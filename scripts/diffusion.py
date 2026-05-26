import os
import pathlib
from numpy import true_divide
import torch
from collections import OrderedDict
from nucleus.baseline.moe_dpot import DataLoader
from nucleus.models import get_model
import hydra
from omegaconf import DictConfig, OmegaConf
from nucleus.data.normalize import get_normalizer
from nucleus.test import run_test, TestResults
from nucleus.plot.plotting import (
    plot_rollout, 
    plot_rollout_stability, 
    plot_rollout_moe_overlay,
)
from nucleus.plot.plot_metrics import (
    plot_simple_metrics,
    plot_vapor_volume_at_height,
    plot_bubble_counts,
)
from nucleus.utils.set_fp32_precision import set_fp32_precision


import torch
from PIL import Image
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

import torch.nn.functional as F

from diffusers.optimization import get_cosine_schedule_with_warmup

from diffusers.pipelines.ddpm.pipeline_ddpm import DDPMPipeline
from diffusers.utils.pil_utils import make_image_grid
import os

from accelerate import Accelerator
from huggingface_hub import create_repo, upload_folder
from tqdm.auto import tqdm
from pathlib import Path
import os


#for now I'm hardcoding this only to work with neighbor_moe
#load training set

train_dataloader = DataLoader(
    train_dataset,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=10,
    pin_memory=True,
    prefetch_factor=2,
    persistent_workers=True,
    collate_fn=collate,
)


#noise schedule and loss

noise_scheduler = DDPMScheduler(num_train_timesteps=1000)
noise = torch.randn(sample_image.shape)
timesteps = torch.LongTensor([50])
noisy_image = noise_scheduler.add_noise(sample_image, noise, timesteps)

Image.fromarray(((noisy_image.permute(0, 2, 3, 1) + 1.0) * 127.5).type(torch.uint8).numpy()[0])

#predict noise added

noise_pred = model(noisy_image, timesteps).sample
loss = F.mse_loss(noise_pred, noise)

#training the model!
optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
lr_scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=config.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * config.num_epochs),
)

#evaluating the model!

def evaluate(config, epoch, pipeline):
    # Sample some images from random noise (this is the backward diffusion process).
    # The default pipeline output type is `List[PIL.Image]`
    images = pipeline(
        batch_size=config.eval_batch_size,
        generator=torch.Generator(device='cpu').manual_seed(config.seed), # Use a separate torch generator to avoid rewinding the random state of the main training loop
    ).images

    # Make a grid out of the images
    image_grid = make_image_grid(images, rows=4, cols=4)

    # Save the images
    test_dir = os.path.join(config.output_dir, "samples")
    os.makedirs(test_dir, exist_ok=True)
    image_grid.save(f"{test_dir}/{epoch:04d}.png")

#training loop

def train_loop(config, model, noise_scheduler, optimizer, train_dataloader, lr_scheduler):
    # Initialize accelerator and tensorboard logging
    accelerator = Accelerator(
        mixed_precision=config.mixed_precision,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        log_with="tensorboard",
        project_dir=os.path.join(config.output_dir, "logs"),
    )
    if accelerator.is_main_process:
        if config.output_dir is not None:
            os.makedirs(config.output_dir, exist_ok=True)
        if config.push_to_hub:
            repo_id = create_repo(
                repo_id=config.hub_model_id or Path(config.output_dir).name, exist_ok=True
            ).repo_id
        accelerator.init_trackers("train_example")

    # Prepare everything
    # There is no specific order to remember, you just need to unpack the
    # objects in the same order you gave them to the prepare method.
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )

    global_step = 0

    # Now you train the model
    for epoch in range(config.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader), disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        for step, batch in enumerate(train_dataloader):
            clean_images = batch["images"]
            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape, device=clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps, (bs,), device=clean_images.device,
                dtype=torch.int64
            )

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                noise_pred = model(noisy_images, timesteps, return_dict=False)[0]
                loss = F.mse_loss(noise_pred, noise)
                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            progress_bar.update(1)
            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0], "step": global_step}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
            global_step += 1

        # After each epoch you optionally sample some demo images with evaluate() and save the model
        if accelerator.is_main_process:
            pipeline = DDPMPipeline(unet=accelerator.unwrap_model(model), scheduler=noise_scheduler)

            if (epoch + 1) % config.save_image_epochs == 0 or epoch == config.num_epochs - 1:
                evaluate(config, epoch, pipeline)

            if (epoch + 1) % config.save_model_epochs == 0 or epoch == config.num_epochs - 1:
                if config.push_to_hub:
                    upload_folder(
                        repo_id=repo_id,
                        folder_path=config.output_dir,
                        commit_message=f"Epoch {epoch}",
                        ignore_patterns=["step_*", "epoch_*"],
                    )
                else:
                    pipeline.save_pretrained(config.output_dir)


@hydra.main(version_base=None, config_path="../config", config_name="default")
def main(cfg: DictConfig):
    set_fp32_precision()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_name = cfg.model_cfg.name

    # model_kwargs = OmegaConf.to_container(cfg.model_cfg.params, resolve=True)


    model_kwargs = {
        "input_fields": 4,
        "output_fields": 4,
        "patch_size": cfg.model_cfg.params.patch_size,
        "embed_dim": cfg.model_cfg.params.embed_dim,
        "processor_blocks": cfg.model_cfg.params.processor_blocks,
        "num_heads": cfg.model_cfg.params.num_heads,
        "num_fluid_params": cfg.model_cfg.params.num_fluid_params,
    }

    # model_kwargs = OmegaConf.to_container(cfg.model_cfg.params, resolve=True)

    if cfg.model_cfg.params.get("num_experts", None) is not None:
        model_kwargs["num_experts"] = cfg.model_cfg.params.num_experts
        model_kwargs["topk"] = cfg.model_cfg.params.topk


    model = get_model(model_name, **model_kwargs)
    model = model.to(device)
    model_data = torch.load(cfg.checkpoint_path, map_location=device, weights_only=False)
            
    weight_state_dict = OrderedDict()
    for key, val in model_data["state_dict"].items():
        print(key, val.shape)
        if isinstance(model, LightningModule):
            name = key
        else:
            name = key[6:]
        weight_state_dict[name] = val
    del model_data
    model.load_state_dict(weight_state_dict)
    model.eval()


    normalizer = get_normalizer(OmegaConf.to_container(cfg.normalizer_cfg, resolve=True))
    
    # Rollouts are saved in the directory containing the checkpoint
    save_root = pathlib.Path(cfg.checkpoint_path).parent / "rollouts"
    save_root.mkdir(parents=True, exist_ok=True)
    all_test_results = []
    for test_file_path in cfg.data_cfg.test_paths:
        test_results: TestResults = run_test(cfg, model, normalizer, test_file_path, max_timesteps=1000)
        all_test_results.append(test_results)

        save_dir = save_root / f"{test_results.case_name}"
        save_dir.mkdir(parents=True, exist_ok=True)
        plot_rollout(
           save_dir=save_dir,
           rollout=test_results.preds,
           test_results=test_results,
           step_size=5,
            include_ground_truth=True,
        )
        plot_distribution(
            save_dir=save_dir,
            rollout=test_results.preds,
            test_results=test_results,
        )
        
    torch.save(all_test_results, save_root / "test_results_reinit.pt")
if __name__ == "__main__":
    # pylint: disable=no-value-for-parameter
    main()