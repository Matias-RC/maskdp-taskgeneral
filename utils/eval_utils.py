"""Helpers and evaluation loops for offline snapshot evaluation.

These used to live directly inside the eval entry-point script. They're
pulled out here so that eval_offline.py, like offline.py / online.py, keeps
the root file free of anything but `main`.
"""

from pathlib import Path

import hydra
import numpy as np
import torch

import utils.utils as utils


def get_domain(task):
    """Get name of task"""
    if task.startswith("point_mass_maze"):
        return "point_mass_maze"
    return task.split("_", 1)[0]


def get_data_seed(seed, num_data_seeds):
    return (seed - 1) % num_data_seeds + 1


def get_dir(cfg):
    """Get path to model weights"""

    # Resolves back to the workspace root Hydra was launched from
    original_working_dir = Path(hydra.utils.get_original_cwd())

    # Anchors the snapshot directory to your true workspace root
    snapshot_base_dir = original_working_dir / cfg.snapshot_base_dir
    snapshot_dir = snapshot_base_dir / get_domain(cfg.task)

    snapshot = snapshot_dir / str(1) / f"snapshot_{cfg.snapshot_ts}.pt"
    return snapshot


def eval_seq_bc(
    global_step,
    agent,
    env,
    logger,
    goal_iter,
    device,
    num_eval_episodes,
    video_recorder,
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
        video_recorder.init(env, enabled=True)
        time_budget = timestep[episode] + 5
        obs = start_obs[episode].unsqueeze(0)
        for _ in range(time_budget):
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(obs, goal_obs[episode], global_step)
            time_step = env.step(action)
            obs_t = np.asarray(time_step.observation)
            obs_t = torch.as_tensor(obs_t, device=device)
            obs = torch.cat((obs, obs_t.unsqueeze(0)), dim=0)

            dist = np.linalg.norm(
                time_step.observation - goal_obs[episode].cpu().numpy()
            )
            dist2goal = min(dist2goal, dist)
            video_recorder.record(env)
            step += 1

        video_recorder.save(f"{global_step}.mp4")
        video_recorder.render_goal(env, goal_physics[episode])
        episode += 1
        total_dist2goal.append(dist2goal)

    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("distance2goal", np.mean(total_dist2goal))
        log("std", np.std(total_dist2goal))
        log("episode_length", step / episode)
        log("step", global_step)


def eval_bc(
    global_step,
    agent,
    env,
    logger,
    goal_iter,
    device,
    num_eval_episodes,
    video_recorder,
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
        video_recorder.init(env, enabled=True)
        time_budget = timestep[episode] + 5
        obs = start_obs[episode]
        for _ in range(time_budget):
            with torch.no_grad(), utils.eval_mode(agent):
                action = agent.act(obs, goal_obs[episode], global_step)
            time_step = env.step(action)
            obs = np.asarray(time_step.observation)
            obs = torch.as_tensor(obs, device=device)
            dist = np.linalg.norm(
                time_step.observation - goal_obs[episode].cpu().numpy()
            )
            dist2goal = min(dist2goal, dist)
            video_recorder.record(env)
            step += 1

        video_recorder.render_goal(env, goal_physics[episode])
        video_recorder.save(f"{global_step}.mp4")

        episode += 1
        total_dist2goal.append(dist2goal)

    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("distance2goal", np.mean(total_dist2goal))
        log("std", np.std(total_dist2goal))
        log("episode_length", step / episode)
        log("step", global_step)


def eval_mdp(
    global_step,
    agent,
    env,
    logger,
    goal_iter,
    device,
    num_eval_episodes,
    video_recorder,
    cfg,
    replan=False,
):
    step, episode, total_dist2goal = 0, 0, []
    eval_until_episode = utils.Until(num_eval_episodes)
    batch = next(goal_iter)
    start_obs, start_physics, goal_obs, goal_physics, timestep = utils.to_torch(
        batch, device
    )
    timestep = torch.clamp(timestep, max=250)

    while eval_until_episode(episode):
        time_step = env.reset()
        with env.physics.reset_context():
            env.physics.set_state(start_physics[episode].cpu())
        dist2goal = 1e6

        is_last_episode = (episode == num_eval_episodes - 1)
        video_recorder.init(env, enabled=is_last_episode)

        if replan is False:
            with torch.no_grad(), utils.eval_mode(agent):
                actions = agent.act(
                    start_obs[episode].unsqueeze(0),
                    goal_obs[episode].unsqueeze(0),
                    timestep[episode],
                )

            for a in actions:
                time_step = env.step(a)
                if is_last_episode:
                    video_recorder.record(env)
                step += 1
                dist = np.linalg.norm(
                    time_step.observation - goal_obs[episode].cpu().numpy()
                )
                dist2goal = min(dist2goal, dist)

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

                if is_last_episode:
                    video_recorder.record(env)
                step += 1

        if is_last_episode:
            video_recorder.render_goal(env, goal_physics[episode])
            video_name = f"{cfg.task}_step_{global_step}.mp4"
            video_recorder.save(video_name)

        episode += 1
        total_dist2goal.append(dist2goal)

    with logger.log_and_dump_ctx(global_step, ty="eval") as log:
        log("distance2goal", np.mean(total_dist2goal))
        log("std", np.std(total_dist2goal))
        log("episode_length", step / episode)
        log("step", global_step)


def eval_dataset(
    global_step,
    agent,
    env,
    logger,
    dataset_iter,
    device,
    num_eval_episodes,
    video_recorder,
    cfg,
):
    step, episode = 0, 0
    eval_until_episode = utils.Until(num_eval_episodes)

    batch = next(dataset_iter)
    _, expert_actions, physics_seq, _, _ = utils.to_torch(batch, device)

    while eval_until_episode(episode):
        time_step = env.reset()

        with env.physics.reset_context():
            env.physics.set_state(physics_seq[episode, 0].cpu())

        is_last_episode = (episode == num_eval_episodes - 1)
        video_recorder.init(env, enabled=is_last_episode)

        for action in expert_actions[episode]:
            time_step = env.step(action.cpu().numpy())
            if is_last_episode:
                video_recorder.record(env)
            step += 1

        if is_last_episode:
            video_name = f"dataset_replay_{cfg.task}_step_{global_step}.mp4"
            video_recorder.save(video_name)

        episode += 1

    with logger.log_and_dump_ctx(global_step, ty="eval_dataset") as log:
        log("episode_length", step / episode)
        log("step", global_step)