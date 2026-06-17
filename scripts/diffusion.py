import torch
import torch.nn.functional as F
from diffusers.models.unets.unet_2d import UNet2DModel
from diffusers.schedulers.scheduling_ddim import DDIMScheduler
import hydra
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from lightning import seed_everything
from nucleus.data import InMemForecastDataset
from nucleus.data.batching import collate
from nucleus.data.normalize import get_normalizer
from nucleus.data.layout import convert_layout
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
        in_channels=2 * T * C,
        out_channels=T * C,
        block_out_channels=(64, 128, 256, 512),
        layers_per_block=2,
    ).to(device)

    scheduler = DDIMScheduler(
        num_train_timesteps=100, beta_start=0.0001, beta_end=0.02
    )

    optimizer = torch.optim.AdamW(unet.parameters(), lr=1e-4)

    # Training
    unet.train()
    global_step = 0
    for epoch in range(1000):
        for batch in train_dataloader:
            batch = batch.to(device)

            if layout == "t h w c":
                inp = rearrange(batch.input, "b t h w c -> b (t c) h w")
                tgt = rearrange(batch.target, "b t h w c -> b (t c) h w")
            else:
                inp = rearrange(batch.input, "b t c h w -> b (t c) h w")
                tgt = rearrange(batch.target, "b t c h w -> b (t c) h w")

            noise = torch.randn_like(tgt)
            timesteps = torch.randint(
                0, scheduler.config.num_train_timesteps, (inp.shape[0],), device=device
            ).long()
            noisy_tgt = scheduler.add_noise(tgt, noise, timesteps)

            model_input = torch.cat([inp, noisy_tgt], dim=1)
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

    # Inference rollout
    unet.eval()
    save_dir = Path("diffusion_rollout")
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
            for itr in range(0, 1000, skip_itrs):
                data = test_dataset[itr]
                batch = data.to_collated_batch().to(device)

                bulk_temp = normalizer.unnormalize_params(
                    [batch.fluid_params_dict[0]]
                )[0]["bulk_temp"]

                if len(preds_list) > 0:
                    last_pred = preds_list[-1].unsqueeze(0).to(device)
                    batch.input = normalizer.normalize(
                        last_pred, bulk_temp, layout=layout
                    )

                if layout == "t h w c":
                    inp_flat = rearrange(batch.input, "b t h w c -> b (t c) h w")
                else:
                    inp_flat = rearrange(batch.input, "b t c h w -> b (t c) h w")

                tgt = batch.target

                noise = torch.randn(1, T * C, *inp_flat.shape[2:], device=device)
                noisy = noise.clone()

                for t in range(scheduler.config.num_train_timesteps - 1, -1, -1):
                    model_input = torch.cat([inp_flat, noisy], dim=1)
                    pred_noise = unet(
                        model_input,
                        torch.full((1,), t, device=device, dtype=torch.long),
                    ).sample
                    noisy = scheduler.step(pred_noise, t, noisy).prev_sample

                pred_flat = noisy

                if layout == "t h w c":
                    pred = rearrange(
                        pred_flat, "b (t c) h w -> b t h w c", t=T
                    )
                else:
                    pred = rearrange(
                        pred_flat, "b (t c) h w -> b t c h w", t=T
                    )

                pred = normalizer.unnormalize(pred, bulk_temp, layout=layout)
                tgt = normalizer.unnormalize(tgt, bulk_temp, layout=layout)

                pred = pred.to(torch.float32).squeeze(0).detach().cpu()
                tgt = tgt.to(torch.float32).squeeze(0).detach().cpu()

                if not pred.isfinite().all() or not tgt.isfinite().all():
                    print(f"Hit NaN at iter {itr}")
                    break

                for t_idx in range(pred.shape[0]):
                    if layout == "t h w c":
                        pred[t_idx, :, :, 0] = sdf_reinit_sussman(
                            pred[t_idx, :, :, 0], dx=1 / 4
                        )
                    else:
                        pred[t_idx, 0, :, :] = sdf_reinit_sussman(
                            pred[t_idx, 0, :, :], dx=1 / 4
                        )

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
