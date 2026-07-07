import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# https://gymnasium.farama.org/environments/mujoco/
os.environ["MUJOCO_GL"] = "disable" # egl doesn't work on Peteroa :(

from pathlib import Path

import hydra
import torch
import wandb
import omegaconf
from pprint import pprint
from utils.logger import Logger
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from data.replay_buffer import make_replay_loader
from utils.utils import set_seed_everywhere, Until, Every, Timer

torch.backends.cudnn.benchmark = True

@hydra.main(version_base=None, config_path="configs", config_name="eval_online")
def main(cfg: DictConfig):
    work_dir = Path.cwd()
    # Print the actual directory determined by hydra config
    print(f"Workspace: {work_dir}")

    # Set seed for random, numpy and torch
    set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")
    
    # Since we may work with dmc, mtm_dmc or future environment standards
    env_cls = instantiate(
        cfg.env_cls_route
    )

    # Create environment for specified task
    env = env_cls.make(cfg.task, seed=cfg.seed)
    # Create agent. 
    agent = instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
    )
    """
    TODO:
    logger = instantiate(cfg.eval_logger)
    total_reward = 0
    rewards = []
    for i in range(cfg.eval_steps):
        action = agent.act(env.state)
        reward, _, _, _ = env.step(action)
        total_reward += reward
        rewards.append(reward)
    std = utils.get_std(rewards)
    logger.log(total_reward/cfg.eval_steps, std)
    """
    


if __name__ == "__main__":
    main()
