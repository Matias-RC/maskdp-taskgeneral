import datetime
import io
import random
import traceback
import copy
import tempfile
import atexit
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from utils import get_norm

# Hugging Face programmatic tools
from huggingface_hub import list_repo_files, hf_hub_download


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode)
        bs.seek(0)
        with fn.open("wb") as f:
            f.write(bs.read())


def load_episode(fn, obs):
    with fn.open("rb") as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


def relable_episode(env, episode):
    rewards = []
    reward_spec = env.reward_spec()
    states = episode["physics"]
    for i in range(states.shape[0]):
        with env.physics.reset_context():
            env.physics.set_state(states[i])
        reward = env.task.get_reward(env.physics)
        reward = np.full(reward_spec.shape, reward, reward_spec.dtype)
        rewards.append(reward)
    episode["reward"] = np.array(rewards, dtype=reward_spec.dtype)
    return episode


class OfflineReplayBuffer(IterableDataset):
    def __init__(
        self,
        env,
        replay_dir,
        max_size,
        num_workers,
        discount,
        traj_length,
        mode,
        cfg,
        relabel,
        obs,
    ):
        self._env = env
        self._replay_dir = Path(replay_dir)
        self._mode = mode
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._discount = discount
        self._loaded = False
        self._traj_length = traj_length
        self._cfg = cfg
        self._relabel = relabel
        self._obs = obs

    def _load(self, relable=True):
        if relable:
            print("Labeling data...")
        else:
            print("loading reward free data...")
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
            
        eps_fns = sorted(
            self._replay_dir.rglob("*.npz")
        )  # get all episodes recursively
        
        for eps_fn in eps_fns:
            if self._size > self._max_size:
                print("over size", self._max_size)
                break
                
            try:
                eps_idx = int(eps_fn.stem.split("_")[1])
            except (IndexError, ValueError):
                continue
                
            if eps_idx % self._num_workers != worker_id:
                continue
                
            episode = load_episode(eps_fn, self._obs)
            if relable:
                episode = self._relable_reward(episode)
            self._episode_fns.append(eps_fn)
            self._episodes[eps_fn] = episode
            self._size += episode_len(episode)

    def _sample_episode(self):
        if not self._loaded:
            self._load(self._relabel)
            self._loaded = True
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _relable_reward(self, episode):
        return relable_episode(self._env, episode)

    def _sample(self):
        episode = self._sample_episode()
        idx = np.random.randint(0, episode_len(episode) - self._traj_length + 1) + 1
        obs = episode["observation"][idx - 1 : idx - 1 + self._traj_length]
        action = episode["action"][idx : idx + self._traj_length]
        next_obs = episode["observation"][idx : idx + self._traj_length]
        reward = episode["reward"][idx : idx + self._traj_length]
        discount = episode["discount"][idx : idx + self._traj_length] * self._discount
        timestep = np.arange(idx - 1, idx + self._traj_length - 1)[:, np.newaxis]
        return (obs, action, reward, discount, next_obs, 0)

    def _sample_goal(self):
        episode = self._sample_episode()
        start_idx = np.random.randint(0, 900)
        length = np.random.randint(15, 20)
        start_obs = episode["observation"][start_idx]
        start_physics = episode["physics"][start_idx]
        goal_obs = episode["observation"][start_idx + length - 1]
        goal_physics = episode["physics"][start_idx + length - 1]
        timestep = length - 1
        return (start_obs, start_physics, goal_obs, goal_physics, timestep)

    def _sample_multiple_goal(self):
        episode = self._sample_episode()
        start_idx = np.random.randint(0, 850)
        time_budget = np.array([12, 24, 36, 48, 60])
        start_obs = episode["observation"][start_idx]
        start_physics = episode["physics"][start_idx]
        goal = episode["observation"][start_idx + time_budget]
        goal_physics = episode["physics"][start_idx + time_budget]
        return (start_obs, start_physics, goal, goal_physics, time_budget)

    def _sample_context(self):
        episode = self._sample_episode()
        context_length = self._cfg.context_length
        forecast_length = self._cfg.forecast_length
        start_idx = np.random.randint(100, 850)
        obs = episode["observation"][
            start_idx - 1 : start_idx + context_length
        ]  
        action = episode["action"][start_idx : start_idx + context_length]
        reward = episode["reward"][
            start_idx + context_length : start_idx + context_length + forecast_length
        ]
        physics = episode["physics"][start_idx - 1 : start_idx + context_length]
        remaining = episode["action"][
            start_idx + context_length : start_idx + context_length + forecast_length
        ]
        return (obs, action, physics, reward, remaining)

    def __iter__(self):
        while True:
            if self._mode is None:
                yield self._sample()
            elif self._mode == "goal":
                yield self._sample_goal()
            elif self._mode == "multi_goal":
                yield self._sample_multiple_goal()
            elif self._mode == "prompt":
                yield self._sample_context()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(
    env,
    replay_dir,
    max_size,
    batch_size,
    num_workers,
    discount,
    traj_length=1,
    mode=None,
    cfg=None,
    multi_task=False,
    relabel=True,
    obs="states",
    is_local_data=True,
    hf_path=None,
    download_fraction=0.1,
):
    # hugging face safe local download logic
    if not is_local_data and hf_path is not None:
        # Create a true temporary directory that completely bypasses persistent cache
        temp_dir = tempfile.mkdtemp(prefix="hf_replay_cache_")
        
        # Safe cleanup hook: Erases files on job completion, timeout, or cancellation
        atexit.register(lambda: shutil.rmtree(temp_dir, ignore_errors=True))
        print(f"[Master] Created temporary directory for dataset: {temp_dir}. Will auto-delete on exit.", flush=True)
        
        # Parse hf_path into repo_id and sub_folder
        parts = hf_path.split("/")
        if len(parts) > 2:
            repo_id = f"{parts[0]}/{parts[1]}"
            sub_folder = "/".join(parts[2:])
        else:
            repo_id = hf_path
            sub_folder = None

        print(f"[Master] Target Repo ID: '{repo_id}' | Sub-folder: '{sub_folder}'", flush=True)
        
        # Fetch the registry layout via the official string lookup endpoint
        repo_files = list_repo_files(repo_id, repo_type="dataset")
        
        # Cleanly filter matching records inside the designated subdirectory layout
        all_files = sorted([
            f for f in repo_files 
            if f.endswith('.npz') and (sub_folder is None or f.startswith(sub_folder))
        ])
        
        if not all_files:
            raise ValueError(f"No .npz files found in HF path '{hf_path}' subfolder '{sub_folder}'!")

        # Calculate space constraints using download_fraction
        num_files = max(1, int(len(all_files) * download_fraction))
        files_to_download = all_files[:num_files]
        
        print(f"[Master] Starting download of {num_files} files ({download_fraction*100:.1f}%) from HF...", flush=True)
        
        # Iterate and stream safely while keeping Slurm log updates clean
        for idx, filename in enumerate(files_to_download, 1):
            if idx == 1 or idx == num_files or idx % 5 == 0:
                percent_done = (idx / num_files) * 100
                print(f"[Slurm-Download] Progress: {idx}/{num_files} files grabbed ({percent_done:.1f}%) | Fetching: {filename.split('/')[-1]}", flush=True)
            
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=filename,
                cache_dir=temp_dir, 
                local_dir=temp_dir
            )
        
        print("[Master] Dataset download complete! Proceeding to environment loading...", flush=True)
        
        # Point the local directory target directly to the temporary download tree
        if sub_folder:
            replay_dir = Path(temp_dir) / sub_folder
        else:
            replay_dir = Path(temp_dir)
            
        print(f"[Master] Localized data folder target: {replay_dir}", flush=True)
    else:
        replay_dir = Path(replay_dir)
    # ----------------------------------------------

    max_size_per_worker = max_size // max(1, num_workers)

    iterable = OfflineReplayBuffer(
        env,
        replay_dir,
        max_size_per_worker,
        num_workers,
        discount,
        traj_length,
        mode,
        cfg,
        relabel,
        obs,
    )

    loader = torch.utils.data.DataLoader(
        iterable,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        worker_init_fn=_worker_init_fn,
    )
    return loader