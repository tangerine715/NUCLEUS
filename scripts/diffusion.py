import matplotlib
matplotlib.use("Agg")
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
from nucleus.utils.physical_metrics import PhysicalMetrics, BubbleMetrics
from nucleus.test import TestResults
from nucleus.plot.plotting import plot_rollout
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

    def _dummy_metrics(B, T_rollout, H, W):
        pm = PhysicalMetrics(
            eikonal=torch.zeros(B, T_rollout),
            heatflux=torch.zeros(B, T_rollout),
            heatflux_at_heater=torch.zeros(B, T_rollout),
            liquid_divergence=torch.zeros(B, T_rollout),
            mean_liquid_temperature=torch.zeros(B, T_rollout),
            liquid_temperature_at_heater=torch.zeros(B, T_rollout, W),
            vapor_volume=torch.zeros(B, T_rollout),
            vapor_volume_at_height=torch.zeros(B, T_rollout, H),
            temperature_distribution=torch.zeros(B, 100),
            velx_distribution=torch.zeros(B, 100),
            vely_distribution=torch.zeros(B, 100),
            mean_liquid_x_velocity=torch.zeros(B, T_rollout),
            mean_liquid_y_velocity=torch.zeros(B, T_rollout),
            mean_vapor_x_velocity=torch.zeros(B, T_rollout),
            mean_vapor_y_velocity=torch.zeros(B, T_rollout),
            mean_interface_x_velocity=torch.zeros(B, T_rollout),
            mean_interface_y_velocity=torch.zeros(B, T_rollout),
        )
        bm = BubbleMetrics(
            bubble_labels=torch.zeros(B, T_rollout, H, W, dtype=torch.long),
            bubble_count=torch.zeros(B, T_rollout),
            bubble_volume=[[] for _ in range(B)],
            bubble_x_velocity=[[] for _ in range(B)],
            bubble_y_velocity=[[] for _ in range(B)],
        )
        return pm, bm

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
    noise_level_ratio = 0.3

    pretrained_rollout = None
    if cfg.get("rollout_path"):
        data = torch.load(cfg.rollout_path, map_location="cpu", weights_only=False)
        if isinstance(data, list):
            pretrained_rollout = data[0].preds.squeeze(0)
        elif isinstance(data, dict):
            pretrained_rollout = data["preds"].squeeze(0)
        else:
            pretrained_rollout = data.preds.squeeze(0)
        print(f"Loaded pretrained rollout with shape {pretrained_rollout.shape}")

    save_dir = Path(get_original_cwd()) / "diffusion_rollout"
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
        noise_level = int(scheduler.config.num_train_timesteps * noise_level_ratio)
        prev_clean_flat = None

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

                if pretrained_rollout is not None:
                    start = itr // skip_itrs * T
                    pred_raw = pretrained_rollout[start:start + T].unsqueeze(0).to(device)
                    if layout != "t h w c":
                        pred_raw = convert_layout(pred_raw, target_layout=layout, source_layout="t h w c")
                    pred_raw = normalizer.normalize(pred_raw, bulk_temp, layout=layout)
                else:
                    raw = pretrained(batch.get_input())
                    pred_raw = raw[0] if isinstance(raw, tuple) else raw
                pred_flat = _flatten(pred_raw)

                if prev_clean_flat is None:
                    noise = torch.randn(1, T * C, *inp_flat.shape[2:], device=device)
                    noisy = noise.clone()
                else:
                    noise = torch.randn_like(prev_clean_flat)
                    noisy = scheduler.add_noise(prev_clean_flat, noise, torch.full((1,), noise_level, device=device))

                for t in scheduler.timesteps:
                    if prev_clean_flat is not None and t > noise_level:
                        continue
                    model_input = torch.cat([inp_flat, pred_flat, noisy], dim=1)
                    pn = unet(model_input, torch.full((1,), t, device=device, dtype=torch.long)).sample
                    noisy = scheduler.step(pn, t, noisy).prev_sample

                pred_flat_clean = noisy
                prev_clean_flat = pred_flat_clean

                pred = _unflatten(pred_flat_clean)
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

        fluid_params = test_dataset.fluid_params[0]
        B, T_rollout, H, W, _ = preds.shape
        pred_pm, pred_bm = _dummy_metrics(B, T_rollout, H, W)
        tgt_pm, tgt_bm = _dummy_metrics(B, T_rollout, H, W)
        case_name = f"{fluid_params['setup']}_{fluid_params['liquid']}_{fluid_params['heater']['wallTemp']}"
        test_results = TestResults(
            case_name=case_name,
            preds=preds,
            targets=targets,
            pred_physical_metrics=pred_pm,
            target_physical_metrics=tgt_pm,
            pred_bubble_metrics=pred_bm,
            target_bubble_metrics=tgt_bm,
            moe_outputs=[],
            fluid_params=fluid_params,
        )

        case_dir = save_dir / case_name
        case_dir.mkdir(parents=True, exist_ok=True)
        plot_rollout(
            save_dir=str(case_dir),
            rollout=preds,
            test_results=test_results,
            step_size=5,
            include_ground_truth=True,
        )

        result_path = save_dir / f"{Path(test_file_path).stem}_results.pt"
        torch.save(test_results, result_path)
        print(f"Saved rollout to {result_path}")


if __name__ == "__main__":
    main()
