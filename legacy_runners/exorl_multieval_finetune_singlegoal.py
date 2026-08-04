import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os
import csv
import re

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "disable"

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import make_replay_loader

import wandb
import omegaconf

torch.backends.cudnn.benchmark = True


def get_domain(task):
    '''Get name of task'''
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]

def get_all_snapshots(cfg):
    '''Get path to all model weights in folder'''
    snapshot_base_dir = Path(cfg.snapshot_base_dir)
    snapshot_dir = snapshot_base_dir / get_domain(cfg.task) / str(1) / cfg.algorithm
    return sorted(snapshot_dir.glob("snapshot_*.pt"))

def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


def eval_mdp(
    agent, # agent.mdp_goal.MDPGoalAgent
    env: dmc.ExtendedTimeStepWrapper,
    logger: Logger,
    goal_iter: torch.utils.data.dataloader._MultiProcessingDataLoaderIter,
    device: torch.device,
    num_eval_episodes: int,
    seed: int,                 # Only for logging
    replan: bool=False,        # True means closed-loop control
    snapshot_name: str="",     # For naming purposes
    csv_path: Path=None, # Save data to a CSV for easier plotting later
    model_name: str=""   # This is to identify which line belongs to which alg
):

    step, episode, total_dist2goal = 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    batch = next(goal_iter)
    start_obs, start_physics, goal_obs, goal_physics, timestep = utils.to_torch(
        batch, device
    )

    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(start_physics[episode].cpu())
        dist2goal = 1e6
        if replan is False:
            with torch.no_grad(), utils.eval_mode(agent):
                actions = agent.act(
                    start_obs[episode].unsqueeze(0),
                    goal_obs[episode].unsqueeze(0),
                    timestep[episode],
                )

            for a in actions:
                time_step = env.step(a)
                step += 1
                dist = np.linalg.norm(
                    time_step.observation - goal_obs[episode].cpu().numpy()
                )
                dist2goal = min(dist2goal, dist)

            episode += 1
            total_dist2goal.append(dist2goal)
        else:
            obs = start_obs[episode]
            for t in range(timestep[episode]):
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(
                        obs.unsqueeze(0),
                        goal_obs[episode].unsqueeze(0),
                        timestep[episode] - t,
                    )[0, ...]
                time_step = env.step(action)
                obs = np.asarray(time_step.observation)
                obs = torch.as_tensor(obs, device=device)
                dist = np.linalg.norm(
                    time_step.observation - goal_obs[episode].cpu().numpy()
                )
                dist2goal = min(dist2goal, dist)
                step += 1

            episode += 1
            total_dist2goal.append(dist2goal)

    mean_dist = np.mean(total_dist2goal)
    std_dist = np.std(total_dist2goal)
    stderr_dist = std_dist / np.sqrt(len(total_dist2goal))
    episode_length = step / episode

    # 1) Log to WandB
    with logger.log_and_dump_ctx(step=0, ty="eval") as log:
        log("distance2goal", mean_dist)
        log("std", std_dist)
        log("stderr", stderr_dist)
        log("episode_length", episode_length)
        if snapshot_name:
            # Logger only accepts numeric values
            snapshot_name_num = int(re.search(r'\d+', snapshot_name).group(0))
            log("snapshot", snapshot_name_num)

    # 2) Log to CSV
    if csv_path is not None:
        file_exists = csv_path.exists()
        with csv_path.open("a", newline="") as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow([
                    "model_name",
                    "snapshot",
                    "replan",
                    "seed",
                    "num_eval_episodes",
                    "distance2goal",
                    "std",
                    "stderr",
                    "episode_length",
                ])
            writer.writerow([
                model_name,
                snapshot_name,
                replan,
                seed,
                num_eval_episodes,
                mean_dist,
                std_dist,
                stderr_dist,
                episode_length,
            ])

# This links it to eval_exorl.yaml
@hydra.main(config_path=".", config_name="eval_exorl")
def main(cfg):

    # This is simply because when initially pretraining I forgot to add
    # the _{dataset_split} at the end of the algorithm names
    # Those without that are all 100% split though
    if not cfg.algorithm[-1].isnumeric():
        algorithm_name = cfg.algorithm + "_100"
    else:
        algorithm_name = cfg.algorithm

    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed)

    # create agent (for config only)
    snapshot_paths = get_all_snapshots(cfg)
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
        path=snapshot_paths[-1],
    )

    # create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    cfg.agent.transformer_cfg = agent.config
    
    exp_name = "_".join([cfg.agent.name, cfg.task, str(cfg.replan),
                    str(cfg.seed), str(cfg.algorithm)])
    if cfg.exp_name != "None":
        exp_name = cfg.exp_name + "_" + exp_name
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    print("Using WandB:", cfg.use_wandb)
    print("Replan:", cfg.replan)
    wandb.init(
        project=cfg.project,
        entity="bibarelusedfly-cenia",
        name=exp_name,
        config=wandb_config,
        settings=wandb.Settings(_disable_stats=True,),
        mode="online" if cfg.use_wandb else "offline",
        notes=cfg.notes,
    )
    logger = Logger(work_dir, use_tb=cfg.use_tb, use_wandb=cfg.use_wandb)

    # create replay buffer
    data_specs = (
        env.observation_spec(),
        env.action_spec(),
        env.reward_spec(),
        env.discount_spec(),
    )

    # create data storage
    domain = get_domain(cfg.task)

    goal_dir = Path(cfg.goal_buffer_dir) / cfg.task

    print(f"goal buffer dir: {goal_dir}")

    goal_loader = make_replay_loader(
        env,
        goal_dir,
        cfg.goal_buffer_size,
        cfg.num_eval_episodes,
        cfg.goal_buffer_num_workers,
        cfg.discount,
        domain=domain,
        traj_length=1,
        mode="goal",
        cfg=agent.config,
        relabel=False,
    )
    goal_iter = iter(goal_loader)

    timer = utils.Timer()
    csv_path = Path(cfg.csv_path) / "eval_results.csv"

    eval_every_step = utils.Every(cfg.eval_every_steps)

    logger.log("eval_total_time", timer.total_time(), step=0)
    if cfg.agent.name == "mdp_goal":
        for snapshot_path in snapshot_paths:
            print(f"Evaluating snapshot: {snapshot_path}")

            agent = hydra.utils.instantiate(
                cfg.agent,
                obs_shape=env.observation_spec().shape,
                action_shape=env.action_spec().shape,
                path=snapshot_path)

            goal_iter = iter(goal_loader)

            eval_mdp(
                agent,
                env,
                logger,
                goal_iter,
                device,
                cfg.num_eval_episodes,
                cfg.seed,
                replan=cfg.replan,
                snapshot_name=snapshot_path.stem,
                csv_path=csv_path,
                model_name=algorithm_name)
    else:
        raise NotImplementedError

if __name__ == "__main__":
    main()
