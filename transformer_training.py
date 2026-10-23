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
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf

from algorithms.TransformerPPOTrainer import TransformerPPOTrainer
from data.env_transform_loader import EnvTransformLoader
from envs import dmc_gymnasium
from envs.mujoco_gym_context import create_vector_env as create_gym_vector_env
from utils.utils import set_seed_everywhere

# Use with: Python 3.8

ENV_REGISTRY = {
    "gym": create_gym_vector_env,
    "dmc": dmc_gymnasium.create_vector_env,
}

TRAINER_REGISTRY = {
    "ppo": TransformerPPOTrainer,
}


def resolve_device(requested: str) -> torch.device:
    """Fall back to CPU instead of crashing when CUDA was requested but is absent."""
    if requested.startswith("cuda") and not torch.cuda.is_available():
        print(f"WARNING: '{requested}' requested but CUDA is unavailable. Falling back to CPU.")
        return torch.device("cpu")
    return torch.device(requested)


@hydra.main(config_path="configs/ppo_transformer", config_name="config")
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
            context_len=cfg.context_len,
        )

        # Start the env normalizers from the pretraining statistics so a pretrained
        # model sees the same input distribution it was trained on.
        env_transforms_path = cfg.get("env_transforms_path")
        if env_transforms_path:
            env_transforms_path = to_absolute_path(env_transforms_path)
            EnvTransformLoader.load_into_env(
                env_transforms_path, envs,
                freeze=cfg.get("freeze_env_transforms", True),
                prior_count=cfg.get("env_transforms_prior_count"),
            )
            print(f"Loaded env transforms from {env_transforms_path}")

        model_cfg = cfg.trainer_cfg.get("model")
        if model_cfg is not None and model_cfg.get("pretrained_path"):
            model_cfg.pretrained_path = to_absolute_path(model_cfg.pretrained_path)

        exp_name = f"{cfg.env_name}_{algo_name}_transformer"
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