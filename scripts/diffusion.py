import math
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

    lr = cfg.get("lr", 2.5e-4)
    lr_warmup_steps = cfg.get("lr_warmup_steps", 150)
    min_lr_ratio = cfg.get("min_lr_ratio", 0.001)  # decay down to this fraction of `lr` by max_steps
    optimizer = torch.optim.AdamW(unet.parameters(), lr=lr)

    def _lr_lambda(step):
        if step < lr_warmup_steps:
            return (step + 1) / max(1, lr_warmup_steps)
        progress = (step - lr_warmup_steps) / max(1, cfg.max_steps - lr_warmup_steps)
        progress = min(progress, 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, _lr_lambda)

    # pushforward: fraction of steps where the UNet conditions on the surrogate's
    # own fed-back rollout (instead of ground-truth history) so it sees realistic,
    # imperfect conditioning like it will at inference time.
    pushforward_prob = cfg.get("pushforward_prob", 0.5)
    pushforward_steps = cfg.get("pushforward_steps", 2)

    # warm-start: fraction of training steps where the diffusion target is noised
    # starting from the previous window (inp_flat) instead of pure Gaussian noise.
    # The noise *level* for this is no longer a fixed low-sigma ceiling -- it mirrors
    # the adaptive_ratio formula used at inference exactly (see the inference loop
    # below), computed per-sample from how much pred_flat disagrees with inp_flat in
    # the SDF channel. Without this, training only ever pairs inp_flat with low noise,
    # while inference routinely pushes the warm start up near sigma=1.0 whenever
    # something topological is happening -- exactly the frames where bubble shape
    # matters most, and exactly the frames this mismatch was leaving untrained.
    warmstart_prob = cfg.get("warmstart_prob", 0.75)
    noise_level_ratio = cfg.get("noise_level_ratio", 0.2)
    nucleation_sensitivity = cfg.get("nucleation_sensitivity", 5.0)
    train_sigmas = scheduler.sigmas.to(device)[: scheduler.config.num_train_timesteps]


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
                fluid_params = normalizer.unnormalize_params(batch.fluid_params_dict)
                bulk_temps = [fp["bulk_temp"] for fp in fluid_params]
                if (pushforward_steps > 1 and torch.rand(()) < pushforward_prob):
                    for _ in range(pushforward_steps - 1):
                        raw = model(batch.get_input())
                        step_pred = raw[0] if isinstance(raw, tuple) else raw
                        batch.input = torch.stack([
                            normalizer.normalize(step_pred[i:i + 1], bulk_temps[i], layout=layout)[0]
                            for i in range(step_pred.shape[0])
                        ])
                raw = model(batch.get_input())
                pred_raw = raw[0] if isinstance(raw, tuple) else raw
                # pred_raw comes back in physical units (same as step_pred above) -- normalize
                # it so it's on the same scale as inp_flat/tgt_flat before it gets concatenated
                # into the UNet's input. Without this, one third of the UNet's input channels
                # sits at a totally different numeric scale than the other two thirds.
                pred_raw = torch.stack([
                    normalizer.normalize(pred_raw[i:i + 1], bulk_temps[i], layout=layout)[0]
                    for i in range(pred_raw.shape[0])
                ])
            inp_flat = _flatten(batch.input)
            pred_flat = _flatten(pred_raw)

            #sampling
            use_warmstart = torch.rand(()) < warmstart_prob
            if use_warmstart:
                # Noise from the previous window, like prev_clean_flat does at inference.
                # The noise level is adaptive per-sample, mirroring adaptive_ratio at
                # inference: computed from how much pred_flat (this window's raw surrogate
                # prediction) disagrees with inp_flat (the warm-start base) in the SDF
                # channel specifically. Bubbles that are nucleating, merging, or detaching
                # show up as large SDF disagreement, so those samples get pushed toward a
                # high warm-start noise level too, instead of only ever training on the
                # calm low-noise regime the old fixed pool was restricted to.
                base = inp_flat
                pred_unflat_dis = _unflatten(pred_flat)
                inp_unflat_dis = _unflatten(inp_flat)
                if layout == "t h w c":
                    sdf_diff = (pred_unflat_dis[..., 0] - inp_unflat_dis[..., 0]).abs()
                else:
                    sdf_diff = (pred_unflat_dis[:, :, 0] - inp_unflat_dis[:, :, 0]).abs()
                sdf_disagreement = sdf_diff.flatten(1).mean(dim=1)  # (B,) -- per-sample, not a scalar
                adaptive_ratio = (noise_level_ratio + nucleation_sensitivity * sdf_disagreement).clamp(max=1.0)
                # nearest available training sigma to each sample's target ratio
                timesteps = (train_sigmas.unsqueeze(0) - adaptive_ratio.unsqueeze(1)).abs().argmin(dim=1)
            else:
                # original cold-start behavior: noise from pure Gaussian, full schedule
                base = tgt_flat
                timesteps = torch.randint(
                    0, scheduler.config.num_train_timesteps, (inp_flat.shape[0],), device=device
                ).long()
            noise = torch.randn_like(tgt_flat)
            sigma_1d = scheduler.sigmas.to(device)[timesteps]
            # unet_timesteps uses the scheduler's actual convention (sigma * num_train_timesteps),
            # the same thing `scheduler.timesteps` holds and what the inference loop feeds in --
            # NOT the raw sampling index `timesteps`, which runs in the opposite direction
            # (idx=0 -> sigma=1.0, but the proper conditioning value for sigma=1.0 is
            # num_train_timesteps, not 0).
            unet_timesteps = sigma_1d * scheduler.config.num_train_timesteps
            sigmas = sigma_1d
            while sigmas.dim() < tgt_flat.dim():
                sigmas = sigmas.unsqueeze(-1)
            noisy_tgt = (1.0 - sigmas) * base + sigmas * noise

            #predicting
            model_input = torch.cat([inp_flat, pred_flat, noisy_tgt], dim=1)
            pred_noise = unet(model_input, unet_timesteps).sample
            loss = torch.nn.functional.mse_loss(pred_noise, noise - tgt_flat)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            global_step += 1

            if global_step % 100 == 0:
                print(f'Step {global_step}: loss = {loss.item():.6f}, lr = {lr_scheduler.get_last_lr()[0]:.2e}')

            if global_step >= cfg.max_steps:
                break
        if global_step >= cfg.max_steps:
            break

    #flow matching inference
    unet.eval()
    num_inference_steps = 50
    scheduler.set_timesteps(num_inference_steps)
    # noise_level_ratio and nucleation_sensitivity (both set above, now also used to
    # drive the training-time warm-start sampling) together define the *floor* of a
    # per-window adaptive budget rather than a single fixed value.

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

        with torch.no_grad():
            for itr in range(0, len(test_dataset), skip_itrs):
                data = test_dataset[itr]
                batch = data.to_collated_batch().to(device)

                bulk_temp = normalizer.unnormalize_params(
                    [batch.fluid_params_dict[0]]
                )[0]["bulk_temp"]

                if prev_clean_flat is not None:
                    # Feed the diffusion-corrected prediction back as next window's history,
                    # instead of the raw (uncorrected) surrogate output. Otherwise the
                    # surrogate model drifts exactly as it would with no diffusion correction
                    # at all -- the correction never gets a chance to slow down error
                    # accumulation over a long rollout.
                    batch.input = _unflatten(prev_clean_flat)

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
                    # normalize, matching the pretrained_rollout branch above and the
                    # (now fixed) training loop -- pred_raw comes back in physical units.
                    pred_raw = normalizer.normalize(pred_raw, bulk_temp, layout=layout)
                pred_flat = _flatten(pred_raw)

                if prev_clean_flat is None:
                    noise = torch.randn(1, cfg.history_time_window * C, *inp_flat.shape[2:], device=device)
                    noisy = noise.clone()
                    step_first_step = 0
                else:
                    # Adaptive warm-start budget: widen the noise budget for this window
                    # when the raw surrogate's prediction (pred_flat) disagrees a lot with
                    # where the diffusion trajectory currently sits (prev_clean_flat), in
                    # the SDF channel specifically -- that disagreement is the signal that
                    # something topological (nucleation, detachment, merging) is trying to
                    # happen, which a small fixed warm-start budget structurally can't
                    # represent (it's built to stay close to the previous frame). Ordinary
                    # slowly-evolving windows keep the old small budget; windows where the
                    # surrogate is "shouting" about a new bubble get a bigger one.
                    pred_unflat = _unflatten(pred_flat)
                    prev_unflat = _unflatten(prev_clean_flat)
                    if layout == "t h w c":
                        sdf_disagreement = (pred_unflat[..., 0] - prev_unflat[..., 0]).abs().mean().item()
                    else:
                        sdf_disagreement = (pred_unflat[:, :, 0] - prev_unflat[:, :, 0]).abs().mean().item()

                    adaptive_ratio = min(1.0, noise_level_ratio + nucleation_sensitivity * sdf_disagreement)
                    step_first_step = (scheduler.sigmas[:-1] - adaptive_ratio).abs().argmin().item()
                    sigma_noise_step = scheduler.sigmas[step_first_step].item()
                    print(f"  itr {itr}: sdf_disagreement={sdf_disagreement:.4f} -> "
                          f"adaptive_ratio={adaptive_ratio:.3f} (floor={noise_level_ratio})")

                    sig_noise = torch.full((1,), sigma_noise_step, device=device, dtype=torch.float32)
                    while sig_noise.dim() < prev_clean_flat.dim():
                        sig_noise = sig_noise.unsqueeze(-1)
                    noise = torch.randn_like(prev_clean_flat)
                    noisy = (1.0 - sig_noise) * prev_clean_flat + sig_noise * noise

                for idx, t in enumerate(scheduler.timesteps):
                    if idx < step_first_step:
                        continue
                    model_input = torch.cat([inp_flat, pred_flat, noisy], dim=1)
                    pn = unet(model_input, torch.full((1,), t, device=device, dtype=torch.float32)).sample
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