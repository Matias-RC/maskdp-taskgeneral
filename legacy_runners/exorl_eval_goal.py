import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

import os

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
# from video import VideoRecorder
import wandb
import omegaconf

torch.backends.cudnn.benchmark = True


def get_domain(task):
    '''Get name of task'''
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


def get_dir(cfg):
    '''Get path to model weights'''
    snapshot_base_dir = Path(cfg.snapshot_base_dir)
    snapshot_dir = snapshot_base_dir / get_domain(cfg.task)
    ## Change this if model used seed != 1
    snapshot = snapshot_dir / str(1) / cfg.algorithm / \
               f"snapshot_{cfg.snapshot_ts}.pt"
    return snapshot


def eval_mdp(
    global_step,
    agent,
    env,
    logger,
    goal_iter,
    device,
    num_eval_episodes,
    replan=False,
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
        # video_recorder.init(env, enabled=True)
        if replan is False:
            with torch.no_grad(), utils.eval_mode(agent):
                actions = agent.act(
                    start_obs[episode].unsqueeze(0),
                    goal_obs[episode].unsqueeze(0),
                    timestep[episode],
                )

            for a in actions:
                time_step = env.step(a)
                # video_recorder.record(env)
                step += 1
                dist = np.linalg.norm(
                    time_step.observation - goal_obs[episode].cpu().numpy()
                )
                dist2goal = min(dist2goal, dist)

            # video_recorder.save(f"{global_step}.mp4")
            # video_recorder.render_goal(env, goal_physics[episode])
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
                # video_recorder.record(env)
                step += 1

            # video_recorder.save(f"{global_step}.mp4")
            # video_recorder.render_goal(env, goal_physics[episode])
            episode += 1
            total_dist2goal.append(dist2goal)

    # Just one log at the end
    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("distance2goal", np.mean(total_dist2goal))
        log("std", np.std(total_dist2goal))
        log("stderr", np.std(total_dist2goal)/np.sqrt(len(total_dist2goal)))
        log("episode_length", step / episode)
        log("step", global_step)

# This links it to eval.yaml
@hydra.main(config_path=".", config_name="eval_exorl")
def main(cfg):

    work_dir = Path.cwd()
    print(f"workspace: {work_dir}")

    utils.set_seed_everywhere(cfg.seed)
    device = torch.device(cfg.device)

    # create envs
    env = dmc.make(cfg.task, seed=cfg.seed)

    # create agent
    path = get_dir(cfg)
    agent = hydra.utils.instantiate(
        cfg.agent,
        obs_shape=env.observation_spec().shape,
        action_shape=env.action_spec().shape,
        path=path,
    )

    # create logger
    cfg.agent.obs_shape = env.observation_spec().shape
    cfg.agent.action_shape = env.action_spec().shape
    cfg.agent.transformer_cfg = agent.config
    
    exp_name = "_".join([cfg.agent.name, cfg.task, str(cfg.replan),
                    str(cfg.seed), str(cfg.algorithm), str(cfg.snapshot_ts)])
    if cfg.exp_name != "None":
        exp_name = cfg.exp_name + "_" + exp_name
    wandb_config = omegaconf.OmegaConf.to_container(
        cfg, resolve=True, throw_on_missing=True
    )
    wandb.init(
        project=cfg.project,
        entity="bibarelusedfly-cenia",
        name=exp_name,
        config=wandb_config,
        settings=wandb.Settings(
            start_method="thread",
            _disable_stats=True,
        ),
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

    # create video recorders
    # video_recorder = VideoRecorder(work_dir if cfg.save_video else None)

    timer = utils.Timer()

    global_step = 0
    eval_every_step = utils.Every(cfg.eval_every_steps)

    # This is aways true, and global_step is always 0
    if eval_every_step(global_step):
        logger.log("eval_total_time", timer.total_time(), global_step)
        if cfg.agent.name == "mdp_goal":
            eval_mdp(
                global_step,
                agent,
                env,
                logger,
                goal_iter,
                device,
                cfg.num_eval_episodes,
                replan=cfg.replan,
            )
        elif cfg.agent.name == "bc_goal":
            eval_bc(
                global_step,
                agent,
                env,
                logger,
                goal_iter,
                device,
                cfg.num_eval_episodes,
                video_recorder,
            )
        elif cfg.agent.name == "seq_goal":
            eval_seq_bc(
                global_step,
                agent,
                env,
                logger,
                goal_iter,
                device,
                cfg.num_eval_episodes,
                video_recorder,
            )
        else:
            raise NotImplementedError


if __name__ == "__main__":
    main()
