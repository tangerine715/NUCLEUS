from collections import OrderedDict

import torch
import torch.nn.functional as F
from diffusers import UNet2DModel, DDIMScheduler
import hydra
from hydra.utils import get_original_cwd
from lightning import seed_everything
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nucleus.data import InMemForecastDataset
from nucleus.data.batching import collate
from nucleus.data.layout import convert_layout
from nucleus.data.normalize import get_normalizer
from nucleus.models import get_model
from nucleus.utils.set_fp32_precision import set_fp32_precision
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from einops import rearrange
from pathlib import Path


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    set_fp32_precision()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(cfg.seed)

    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))
    layout = cfg.model_cfg.layout
    T = cfg.history_time_window
    C = 4

    # ---------- pre-trained model (frozen) ----------
    ckpt = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    hp = ckpt.get("hyper_parameters", {})
    pt_cfg = hp.get("model_cfg", OmegaConf.to_container(cfg.model_cfg, resolve=True))
    model_kwargs = dict(pt_cfg["params"])
    for key in ("load_balance_loss_weight", "z_loss_weight", "pushforward_prob",
                "pushforward_start_step", "pushforward_decay_rate", "num_windows"):
        model_kwargs.pop(key, None)
    model_kwargs["input_fields"] = len(cfg.data_cfg.input_fields)
    model_kwargs["output_fields"] = len(cfg.data_cfg.output_fields)
    pretrained = get_model(pt_cfg["name"], **model_kwargs).to(device)
    state = OrderedDict()
    for k, v in ckpt["state_dict"].items():
        name = k[6:] if k.startswith("model.") else k
        state[name] = v
    pretrained.load_state_dict(state)
    pretrained.eval()
    pretrained.requires_grad_(False)

    # ---------- diffusion dataset ----------
    train_dataset = InMemForecastDataset(
        filenames=cfg.data_cfg.train_paths,
        input_fields=cfg.data_cfg.input_fields,
        output_fields=cfg.data_cfg.output_fields,
        future_time_window=cfg.future_time_window,
        history_time_window=cfg.history_time_window,
        time_step=cfg.time_step,
        start_time=cfg.start_time,
        normalizer=normalizer,
        augment=True,
        layout=layout,
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate,
    )

    # ---------- diffusion UNet (trained) ----------
    unet = UNet2DModel(
        sample_size=None,
        in_channels=3 * T * C,
        out_channels=T * C,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
    ).to(device)

    scheduler = DDIMScheduler(num_train_timesteps=100, beta_start=0.0001, beta_end=0.02)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    def _flatten(x):
        if layout == "t h w c":
            return rearrange(x, "b t h w c -> b (t c) h w")
        return rearrange(x, "b t c h w -> b (t c) h w")

    def _unflatten(x):
        if layout == "t h w c":
            return rearrange(x, "b (t c) h w -> b t h w c", t=T)
        return rearrange(x, "b (t c) h w -> b t c h w", t=T)

    # ======================== TRAINING ========================
    unet.train()
    global_step = 0
    for epoch in range(1000):
        for batch in train_dataloader:
            batch = batch.to(device)

            inp_flat = _flatten(batch.input)
            tgt_flat = _flatten(batch.target)

            with torch.no_grad():
                raw = pretrained(batch.get_input())
                pred_raw = raw[0] if isinstance(raw, tuple) else raw
            pred_flat = _flatten(pred_raw)

            noise = torch.randn_like(tgt_flat)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (inp_flat.shape[0],), device=device
            ).long()
            noisy_tgt = scheduler.add_noise(tgt_flat, noise, timesteps)

            model_input = torch.cat([inp_flat, pred_flat, noisy_tgt], dim=1)
            pred_noise = unet(model_input, timesteps).sample

            loss = F.mse_loss(pred_noise, noise)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 100 == 0:
                print(f"Step {global_step}: loss = {loss.item():.6f}")

            if global_step >= cfg.max_steps:
                break
        if global_step >= cfg.max_steps:
            break

    # ======================== INFERENCE ========================
    unet.eval()
    num_inference_steps = 20
    scheduler.set_timesteps(num_inference_steps)
    save_dir = Path(cfg.log_dir) / "diffusion_rollout" if cfg.get("log_dir") else Path("diffusion_rollout")
    save_dir.mkdir(parents=True, exist_ok=True)

    for test_file_path in cfg.data_cfg.test_paths:
        test_dataset = InMemForecastDataset(
            filenames=[test_file_path],
            input_fields=cfg.data_cfg.input_fields,
            output_fields=cfg.data_cfg.output_fields,
            future_time_window=cfg.future_time_window,
            history_time_window=cfg.history_time_window,
            time_step=1,
            start_time=cfg.start_time,
            normalizer=normalizer,
            augment=False,
            layout=layout,
        )

        skip_itrs = test_dataset.future_time_window
        preds_list = []
        targets_list = []

        with torch.no_grad():
            for itr in range(0, len(test_dataset), skip_itrs):
                data = test_dataset[itr]
                batch = data.to_collated_batch().to(device)

                bulk_temp = normalizer.unnormalize_params(
                    [batch.fluid_params_dict[0]]
                )[0]["bulk_temp"]

                if len(preds_list) > 0:
                    last_pred = preds_list[-1].unsqueeze(0).to(device)
                    batch.input = normalizer.normalize(last_pred, bulk_temp, layout=layout)

                inp_flat = _flatten(batch.input)

                raw = pretrained(batch.get_input())
                pred_raw = raw[0] if isinstance(raw, tuple) else raw
                pred_flat = _flatten(pred_raw)

                noise = torch.randn(1, T * C, *inp_flat.shape[2:], device=device)
                noisy = noise.clone()

                for t in scheduler.timesteps:
                    model_input = torch.cat([inp_flat, pred_flat, noisy], dim=1)
                    pn = unet(model_input, torch.full((1,), t, device=device, dtype=torch.long)).sample
                    noisy = scheduler.step(pn, t, noisy).prev_sample

                pred = _unflatten(noisy)
                pred = normalizer.unnormalize(pred, bulk_temp, layout=layout)
                tgt = normalizer.unnormalize(batch.target, bulk_temp, layout=layout)

                pred = pred.to(torch.float32).squeeze(0).detach().cpu()
                tgt = tgt.to(torch.float32).squeeze(0).detach().cpu()

                if not pred.isfinite().all() or not tgt.isfinite().all():
                    print(f"Hit NaN at iter {itr}")
                    break

                for t_idx in range(pred.shape[0]):
                    if layout == "t h w c":
                        pred[t_idx, :, :, 0] = sdf_reinit_sussman(pred[t_idx, :, :, 0], dx=1 / 4)
                    else:
                        pred[t_idx, 0, :, :] = sdf_reinit_sussman(pred[t_idx, 0, :, :], dx=1 / 4)

                preds_list.append(pred)
                targets_list.append(tgt)

        preds = torch.cat(preds_list, dim=0)[None, ...]
        targets = torch.cat(targets_list, dim=0)[None, ...]

        preds = convert_layout(preds, target_layout="t h w c", source_layout=layout)
        targets = convert_layout(targets, target_layout="t h w c", source_layout=layout)

        result_path = save_dir / f"{Path(test_file_path).stem}_results.pt"
        torch.save({"preds": preds, "targets": targets}, result_path)
        print(f"Saved rollout to {result_path}")


if __name__ == "__main__":
    main()
