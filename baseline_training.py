import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# https://gymnasium.farama.org/environments/mujoco/
os.environ["MUJOCO_GL"] = "disable"

import time
from pathlib import Path

import hydra
import torch
import wandb
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf

from algorithms.BaselinePPOTrainer import PPOTrainer
from envs import dmc_gymnasium
from envs.mujoco_gym import create_vector_env as create_gym_vector_env
from utils.utils import set_seed_everywhere

# Use with: Python 3.8

ENV_REGISTRY = {
    "gym": create_gym_vector_env,
    "dmc": dmc_gymnasium.create_vector_env,
}

TRAINER_REGISTRY = {
    "ppo": PPOTrainer,
}


def resolve_device(requested: str) -> torch.device:
    """Fall back to CPU instead of crashing when CUDA was requested but is absent."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: '{requested}' requested but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@hydra.main(config_path="configs/ppo_baseline", config_name="config")
def main(cfg: DictConfig) -> None:
    start = time.time()
    work_dir = Path.cwd()
    print(f"Workspace: {work_dir}")

    set_seed_everywhere(cfg.seed)
    device = resolve_device(str(cfg.device))
    print(f"Using device: {device}")

    algo_name = str(cfg.algorithm).lower()
    if algo_name not in TRAINER_REGISTRY:
        raise ValueError(
            f"Algorithm '{algo_name}' is not registered. Available: {list(TRAINER_REGISTRY.keys())}"
        )
    TrainerClass = TRAINER_REGISTRY[algo_name]

    save_path = Path(cfg.trainer_cfg.save_path)
    if not save_path.is_absolute():
        cfg.trainer_cfg.save_path = str(work_dir / save_path)

    wandb_config = OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)

    env_suite = str(cfg.get("env_suite", "gym")).lower()
    if env_suite not in ENV_REGISTRY:
        raise ValueError(f"Env suite '{env_suite}' is not registered. Available: {list(ENV_REGISTRY.keys())}")
    create_vector_env = ENV_REGISTRY[env_suite]

    envs = None
    try:
        envs = create_vector_env(
            cfg.env_name,
            cfg.num_envs,
            cfg.seed,
            gamma=cfg.gamma,
        )

        exp_name = f"{cfg.env_name}_{algo_name}_baseline"
        wandb.init(
            project=cfg.project,
            entity=None,  # This has to be your WandB user-institution
            name=exp_name,
            config=wandb_config,
            settings=wandb.Settings(_disable_stats=True),
            mode="online" if cfg.use_wandb else "offline",
            notes=cfg.notes,
        )

        trainer = TrainerClass(envs, cfg.trainer_cfg)
        trainer.train()
    finally:
        if envs is not None:
            envs.close()
        # wandb.init() always runs (offline when disabled), so always finish it.
        if wandb.run is not None:
            wandb.finish()
        print(f"Total execution time: {time.time() - start:.2f} seconds")


if __name__ == "__main__":
    main()