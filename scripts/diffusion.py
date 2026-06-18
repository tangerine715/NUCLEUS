import matplotlib
from collections import OrderedDict

import torch
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_flow_match_euler_discrete import FlowMatchEulerDiscreteScheduler
import hydra
from lightning import seed_everything
from lightning import LightningModule
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from nucleus.data import InMemForecastDataset
from nucleus.data.batching import collate
from nucleus.data.layout import convert_layout
from nucleus.data.normalize import get_normalizer
from nucleus.models import get_model
from nucleus.utils.set_fp32_precision import set_fp32_precision
from nucleus.utils.sdf_reinit import sdf_reinit_sussman
from nucleus.utils.physical_metrics import PhysicalMetrics, BubbleMetrics, physical_metrics, bubble_metrics
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
    C = 4
    T = cfg.history_time_window  # time-window length used by _flatten/_unflatten below

    model_data = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
    hp = model_data.get("hyper_parameters", {})
    pt_cfg = hp.get("model_cfg", OmegaConf.to_container(cfg.model_cfg, resolve=True))
    model_kwargs = dict(pt_cfg["params"])
    for key in ("load_balance_loss_weight", "z_loss_weight", "pushforward_prob",
                "pushforward_start_step", "pushforward_decay_rate", "num_windows"):
        model_kwargs.pop(key, None)
    model_kwargs["input_fields"] = len(cfg.data_cfg.input_fields)
    model_kwargs["output_fields"] = len(cfg.data_cfg.output_fields)
    model = get_model(pt_cfg["name"], **model_kwargs).to(device)


    weight_state_dict = OrderedDict()
    for key, val in model_data["state_dict"].items():
        if isinstance(model, LightningModule):
            name = key
        else:
            name = key[6:]
        weight_state_dict[name] = val
    del model_data
    model.load_state_dict(weight_state_dict)
    model.eval()
    model.requires_grad_(False)

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

    unet = UNet2DModel(
        sample_size=None,
        in_channels=3 * cfg.history_time_window * C,
        out_channels=cfg.history_time_window * C,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
        norm_num_groups=32,
    ).to(device)

    scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=100)
    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # pushforward: fraction of steps where the UNet conditions on the surrogate's
    # own fed-back rollout (instead of ground-truth history) so it sees realistic,
    # imperfect conditioning like it will at inference time.
    pushforward_prob = cfg.get("pushforward_prob", 0.5)
    pushforward_steps = cfg.get("pushforward_steps", 2)


    def _flatten(x):
        if layout == "t h w c":
            return rearrange(x, "b t h w c -> b (t c) h w")
        return rearrange(x, "b t c h w -> b (t c) h w")

    def _unflatten(x):
        if layout == "t h w c":
            return rearrange(x, "b (t c) h w -> b t h w c", t=T)
        return rearrange(x, "b (t c) h w -> b t c h w", t=T)


    #training
    unet.train()
    global_step = 0
    for epoch in range(1000):
        for batch in train_dataloader:
            batch = batch.to(device)

            tgt_flat = _flatten(batch.target)

            with torch.no_grad():
                if (pushforward_steps > 1 and torch.rand(()) < pushforward_prob):
                    fluid_params = normalizer.unnormalize_params(batch.fluid_params_dict)
                    bulk_temps = [fp["bulk_temp"] for fp in fluid_params]
                    for _ in range(pushforward_steps - 1):
                        raw = model(batch.get_input())
                        step_pred = raw[0] if isinstance(raw, tuple) else raw
                        batch.input = torch.stack([
                            normalizer.normalize(step_pred[i:i + 1], bulk_temps[i], layout=layout)[0]
                            for i in range(step_pred.shape[0])
                        ])
                raw = model(batch.get_input())
                pred_raw = raw[0] if isinstance(raw, tuple) else raw
            inp_flat = _flatten(batch.input)
            pred_flat = _flatten(pred_raw)

            #sampling
            noise = torch.randn_like(tgt_flat)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (inp_flat.shape[0],), device=device
            ).long()
            sigmas = scheduler.sigmas.to(device)[timesteps]
            while sigmas.dim() < tgt_flat.dim():
                sigmas = sigmas.unsqueeze(-1)
            noisy_tgt = (1.0 - sigmas) * tgt_flat + sigmas * noise

            #predicting
            model_input = torch.cat([inp_flat, pred_flat, noisy_tgt], dim=1)
            pred_noise = unet(model_input, timesteps).sample
            loss = torch.nn.functional.mse_loss(pred_noise, noise - tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            global_step += 1

            if global_step % 100 == 0:
                print(f'Step {global_step}: loss = {loss.item():.6f}')

            if global_step >= cfg.max_steps:
                break
        if global_step >= cfg.max_steps:
            break

    #flow matching inference
    unet.eval()
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
    noise_level_ratio = 0.2
    first_step = (scheduler.sigmas[:-1] - noise_level_ratio).abs().argmin().item()
    sigma_noise = scheduler.sigmas[first_step].item()

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

    save_dir = Path(cfg.log_dir) / "diffusion_rollout"
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
        prev_clean_flat = None
        prev_pretrained_pred = None

        with torch.no_grad():
            for itr in range(0, len(test_dataset), skip_itrs):
                data = test_dataset[itr]
                batch = data.to_collated_batch().to(device)

                bulk_temp = normalizer.unnormalize_params(
                    [batch.fluid_params_dict[0]]
                )[0]["bulk_temp"]

                if len(preds_list) > 0 and prev_pretrained_pred is not None:
                    batch.input = normalizer.normalize(prev_pretrained_pred, bulk_temp, layout=layout)

                inp_flat = _flatten(batch.input)

                if pretrained_rollout is not None:
                    start = itr // skip_itrs * cfg.history_time_window
                    pred_raw = pretrained_rollout[start:start + cfg.history_time_window].unsqueeze(0).to(device)
                    if layout != "t h w c":
                        pred_raw = convert_layout(pred_raw, target_layout=layout, source_layout="t h w c")
                    pred_raw = normalizer.normalize(pred_raw, bulk_temp, layout=layout)
                else:
                    raw = model(batch.get_input())
                    pred_raw = raw[0] if isinstance(raw, tuple) else raw
                prev_pretrained_pred = pred_raw.detach().clone()
                pred_flat = _flatten(pred_raw)

                if prev_clean_flat is None:
                    noise = torch.randn(1, cfg.history_time_window * C, *inp_flat.shape[2:], device=device)
                    noisy = noise.clone()
                else:
                    sig_noise = torch.full((1,), sigma_noise, device=device, dtype=torch.float32)
                    while sig_noise.dim() < prev_clean_flat.dim():
                        sig_noise = sig_noise.unsqueeze(-1)
                    noise = torch.randn_like(prev_clean_flat)
                    noisy = (1.0 - sig_noise) * prev_clean_flat + sig_noise * noise

                for idx, t in enumerate(scheduler.timesteps):
                    if prev_clean_flat is not None and idx < first_step:
                        continue
                    model_input = torch.cat([inp_flat, pred_flat, noisy], dim=1)
                    pn = unet(model_input, torch.full((1,), t, device=device, dtype=torch.long)).sample
                    sigma_t = scheduler.sigmas[idx]
                    sigma_next = scheduler.sigmas[idx + 1]
                    noisy = noisy + (sigma_next - sigma_t) * pn

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

        dx = 1/4
        dy = dx
        bulk_temp = fluid_params["bulk_temp"]
        heater_temp = fluid_params["heater"]["wallTemp"]
        pred_pm = physical_metrics(
            preds[..., 0], preds[..., 1], preds[..., 2], preds[..., 3],
            heater_min=-5.25, heater_max=5.25,
            bulk_temp=bulk_temp, heater_temp=heater_temp,
            xcoords=torch.arange(-8, 8, dx) + dx / 2,
            dx=dx, dy=dy,
        )
        pred_bm = bubble_metrics(preds[..., 0], preds[..., 2], preds[..., 3], dx=dx, dy=dy)
        tgt_pm = physical_metrics(
            targets[..., 0], targets[..., 1], targets[..., 2], targets[..., 3],
            heater_min=-5.25, heater_max=5.25,
            bulk_temp=bulk_temp, heater_temp=heater_temp,
            xcoords=torch.arange(-8, 8, dx) + dx / 2,
            dx=dx, dy=dy,
        )
        tgt_bm = bubble_metrics(targets[..., 0], targets[..., 2], targets[..., 3], dx=dx, dy=dy)
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
        print(f"Plotting rollout to {case_dir}")
        plot_rollout(
            save_dir=str(case_dir),
            rollout=preds,
            test_results=test_results,
            step_size=5,
            include_ground_truth=True,
        )

if __name__ == "__main__":
    main()