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
import wandb
import omegaconf
from pprint import pprint

torch.backends.cudnn.benchmark = True


def get_dir(cfg):
    resume_dir = Path(cfg.resume_dir)
    snapshot = resume_dir / f"snapshot_{cfg.resume_step}.pt"
    print("loading from", snapshot)
    return snapshot

def get_domain(task):
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


# This links it to pretrain.yaml
@hydra.main(config_path=".", config_name="finetune")
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

    if cfg.resume:
        pretrained = torch.load(get_dir(cfg), map_location=cfg.device)
        agent.model.load_state_dict(pretrained["model"], strict=True)
        print(f"Loaded pretrained weights from: {get_dir(cfg)}")

    # Create a snapshot directory for the task
    domain = get_domain(cfg.task)
    snapshot_dir = work_dir / domain / str(cfg.seed) / cfg.name
    snapshot_dir.mkdir(exist_ok=True, parents=True)

    # Create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    exp_name = "_".join([
        cfg.agent.name, domain, str(cfg.seed), str(cfg.name)])
    # Create wandb_config from Hydra's omegaconf
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=cfg.project,
        # This has to be your WandB user-institution
        entity="bibarelusedfly-cenia", 
        name=exp_name,
        config=wandb_config,
        settings=wandb.Settings(_disable_stats=True,),
        mode="online" if cfg.use_wandb else "offline",
        notes=cfg.notes,
    )
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # We're training on eval data
    finetune_data_path = "_".join([cfg.replay_buffer_dir, cfg.task])
    replay_train_dir = Path(finetune_data_path) / f"{cfg.data_split}percent"
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

    train_iter = iter(train_loader)

    timer = utils.Timer()

    global_step = 0

    train_until_step = utils.Until(cfg.num_grad_steps)
    eval_every_step = utils.Every(cfg.eval_every_steps)
    log_every_step = utils.Every(cfg.log_every_steps)

    print("Num workers is", cfg.replay_buffer_num_workers)

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
            # Upon exiting the context manager "LogAndDumpCtx", the logged
            # data is actually dumped to WandB

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