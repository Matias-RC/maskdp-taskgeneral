import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
# https://gymnasium.farama.org/environments/mujoco/
os.environ["MUJOCO_GL"] = "disable" # egl doesn't work on Peteroa :(

from pathlib import Path

import hydra
# import numpy as np
import torch
# from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader
# from video import VideoRecorder
import omegaconf
from pprint import pprint

torch.backends.cudnn.benchmark = True


def get_dir(cfg):
    resume_dir = Path(cfg.resume_dir)
    snapshot = resume_dir / str(cfg.seed) / f"snapshot_{cfg.resume_step}.pt"
    print("loading from", snapshot)
    return snapshot


def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1

# This links it to pretrain.yaml
@hydra.main(config_path=".", config_name="pretrain")
def main(cfg):
    work_dir = Path.cwd()
    # Print the actual directory determined by hydra config
    print(f"Workspace: {work_dir}")

    # Set seed for random, numpy and torch
    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)
    print(f"Using device: {device}")

    # Create DeepMindControl environment for specified task
    env = dmc.make(cfg.task, seed=cfg.seed)

    # Create agent. Utils will instantiate a class
    # Since cfg.agent is 'mdp'. The class will be specified by the mdp.yaml
    # in the agent folder
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
    )

    # Create a snapshot directory for the task
    domain = get_domain(cfg.task)
    snapshot_dir = work_dir / Path(cfg.snapshot_dir) / domain / str(cfg.seed)
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # Create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    exp_name = "_".join([cfg.agent.name, domain, str(cfg.seed)])

    # This script is just for testing, do not use WandB
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=False)

    replay_train_dir = Path(cfg.replay_buffer_dir) / domain
    print("Using dataset:", replay_train_dir)
    train_loader = make_replay_loader(
        env,
        replay_train_dir,
        cfg.replay_buffer_size,
        cfg.batch_size,
        cfg.replay_buffer_num_workers,
        cfg.discount,
        domain,
        cfg.agent.transformer_cfg.traj_length,
        relabel=False,
    )

    # See replay_buffer.py - OfflineReplayBuffer._sample for details
    # This is a dataloader, calling 'next' on it returns a list of 6 tensors
    # (batch_size, traj_length, observation_dim)
    # (batch_size, traj_length, action_dim)
    # (batch_size, traj_length, reward_dim)      # Typically 1
    # (batch_size, traj_length, discount_dim)    # Typically 1
    # (batch_size, traj_length, observation_dim) # Next state
    # (batch_size,)                              # This is just 0s
    train_iter = iter(train_loader) 

    timer = utils.Timer()

    global_step = cfg.resume_step

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    # print("----- Pretest -----")
    # testy = next(train_iter)
    # print(testy[0].shape)
    # print(testy[1].shape)
    # print(testy[2].shape)
    # print(testy[3].shape)
    # print(testy[4].shape)
    # print(testy[5].shape)
    # print("-------------------")

    # True until global_step gets to cfg.num_grad_steps
    while train_until_step(global_step):
        # try to evaluate
        # Train on a single batch and permform a gradient step
        metrics = agent.update(train_iter, global_step)
        # Log each metric using the "Train meter group" on the logger
        logger.log_metrics(metrics, global_step, ty="train")
        # Log just registers the metrics on a MetersGroup instance
        # inside the logger
        if log_every_step(global_step):
            elapsed_time, total_time = timer.reset()
            with logger.log_and_dump_ctx(global_step, ty="train") as log:
                log("fps", cfg.log_every_steps / elapsed_time)
                log("total_time", total_time)
                log("step", global_step)

        if global_step in cfg.snapshots:
            snapshot = snapshot_dir / f"snapshot_{global_step}.pt"
            payload = {
                "model": agent.model.state_dict(),
                "cfg": cfg.agent.transformer_cfg,
            }
            with snapshot.open("wb") as f:
                torch.save(payload, f)

        global_step += 1
    print("Done!")


if __name__ == "__main__":
    main()
