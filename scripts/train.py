import os
import argparse

import hydra
import wandb
from omegaconf import OmegaConf, DictConfig
from lightning import seed_everything
from lightning.pytorch.loggers import CSVLogger


@hydra.main(version_base=None, config_path="../src/config", config_name="default")
def main(cfg: DictConfig) -> None:
    seed_everything(cfg.seed)

    params = {}
    params["distributed"] = cfg.distributed
    params["resuming"] = cfg.resuming
    params["checkpoint_path"] = cfg.checkpoint_path
    params["data_cfg"] = OmegaConf.to_container(cfg.data_cfg, resolve=True)
    params["model_cfg"] = OmegaConf.to_container(cfg.model_cfg, resolve=True)
    params["optim_cfg"] =  OmegaConf.to_container(cfg.optim_cfg, resolve=True)
    params["scheduler_cfg"] =  OmegaConf.to_container(cfg.scheduler_cfg, resolve=True)
    
    log_id = (
        cfg.model_cfg.model_name.lower() + "_"
        + cfg.data_cfg.dataset.lower() + "_"
        + os.getenv("SLURM_JOB_ID")
    )
    params["log_dir"] = os.path.join(cfg.log_dir, log_id)
    os.makedirs(params["log_dir"], exist_ok=True)

    local_logger = CSVLogger(
        save_dir=params["log_dir"],
        flush_logs_every_n_steps=10
    )
    wandb_run = None
    if cfg.use_wandb:
        try:
            wandb_key_path = "src/config/wandb_api_key.txt"
            with open(wandb_key_path, "r") as f:
                wandb_key = f.read().strip()
            wandb.login(key=wandb_key)
            wandb_run = wandb.init(
                project="bubbleformer",
                name=log_id,
                tags=cfg.wandb_tags,
                config=params
            )
        except Exception as e:
            print(e)
            print("Valid wandb API key not found at path src/config/wandb_api_key.txt")




if __name__ == "__main__":
    main()
