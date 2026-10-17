"""
Gymnasium front-end for the DMC tasks built by envs/dmc.py.

Drop-in replacement for envs/mujoco_gym.py and envs/mujoco_gym_context.py:
create_vector_env has the same signature and the same wrapper stack, so
BaselinePPOTrainer and TransformerPPOTrainer run unchanged with env names such
as "walker_walk", "cheetah_run" or "quadruped_walk" (any name dmc.make accepts).

Differences from the legacy envs/dmc_to_gym.DMCGymWrapper:
  - gymnasium 5-tuple API (terminated / truncated) instead of gym 4-tuple.
  - Flat (obs_dim,) observations; history stacking is left to
    ContextWindowWrapper, as with the gymnasium MuJoCo envs.
  - reset(seed=...) reseeds the task, so vector-env seeding is honored.
  - No physics state in info (it would be copied into every vector-env step).
"""

import gymnasium as gym
import numpy as np
from dm_env import StepType, specs
from gymnasium import spaces

from envs import dmc
from envs.mujoco_gym_context import ContextWindowWrapper


class DMCEnv(gym.Env):
    """Single DMC task (states only) exposed as a gymnasium.Env."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, name, seed=1, action_repeat=1, render_mode=None,
                 render_height=84, render_width=84, camera_id=0):
        self.name = name
        self._env = dmc.make(name, obs_type="states", action_repeat=action_repeat, seed=seed)

        obs_spec = self._env.observation_spec()
        act_spec = self._env.action_spec()
        assert isinstance(act_spec, specs.BoundedArray), act_spec

        self.observation_space = spaces.Box(-np.inf, np.inf, obs_spec.shape, np.float32)
        self.action_space = spaces.Box(
            low=np.broadcast_to(act_spec.minimum, act_spec.shape).astype(np.float32),
            high=np.broadcast_to(act_spec.maximum, act_spec.shape).astype(np.float32),
            dtype=np.float32,
        )

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        self._render_kwargs = dict(height=render_height, width=render_width, camera_id=camera_id)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            # The DMC task draws its initial state from task.random (set once at
            # construction), so reseed it to make reset(seed=...) reproducible.
            self._env.task._random = np.random.RandomState(seed)
        ts = self._env.reset()
        return np.asarray(ts.observation, dtype=np.float32), {}

    def step(self, action):
        # Policies sample unbounded Gaussians; dm_control does not bounds-check.
        action = np.clip(np.asarray(action, dtype=np.float32),
                         self.action_space.low, self.action_space.high)
        ts = self._env.step(action)

        obs = np.asarray(ts.observation, dtype=np.float32)
        reward = float(ts.reward)
        last = ts.step_type == StepType.LAST
        # DMC signals a true terminal state with discount 0; a LAST step with
        # nonzero discount is the time limit, i.e. a truncation to bootstrap from.
        terminated = bool(last and ts.discount == 0.0)
        truncated = bool(last and not terminated)
        return obs, reward, terminated, truncated, {"discount": float(ts.discount)}

    def render(self):
        if self.render_mode == "rgb_array":
            return self._env.physics.render(**self._render_kwargs)
        return None

    def close(self):
        close = getattr(self._env, "close", None)
        if close is not None:
            close()


def make_env(env_name, seed, gamma=0.99, context_len=None, action_repeat=1):
    """Same wrapper stack as envs/mujoco_gym.make_env (+ ContextWindowWrapper if
    context_len is given), so obs_rms sits at the depth the trainers expect."""
    def thunk():
        env = DMCEnv(env_name, seed=seed, action_repeat=action_repeat)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        if context_len is not None:
            env = ContextWindowWrapper(env, context_len)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def create_vector_env(env_name, num_envs, seed, gamma=0.99, context_len=None, action_repeat=1):
    env_fns = [make_env(env_name, seed + i, gamma, context_len, action_repeat) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(env_fns)
