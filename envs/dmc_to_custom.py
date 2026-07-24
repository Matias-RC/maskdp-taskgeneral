from dm_env import StepType, specs
from collections import deque
import torch as th
import numpy as np
import gym
from envs.dmc import Factory
#  DMC_PPO_MDP : DPM

class DPM_EnvWrapper(gym.Env):
    """
    Wraps a dm_env.Environment into a Gym Env
    Assumes that reset/step return an ExtendedTimeStep with:
        .observation (np.ndarray),
        .reward (float),
        .discount (float),
        .step_type (StepType),
        .physics (np.ndarray)
    (This is because the dmc module wraps envs in an ExtendedTimestepWrapper)
    """

    metadata = {"render.modes": []}
    def __init__(self, dmc_env, truncation_limit=1000):
        super().__init__()
        self._env = dmc_env

        # Array(shape=(24,), dtype='float32'
        obs_spec = dmc_env.observation_spec()
        # BoundedArray(shape=(6,), dtype='float32', minimum=-1.0, maximum=1.0)
        act_spec = dmc_env.action_spec()
        self.obs_dim = obs_spec.shape[0]

        low = -np.inf * np.ones((self.obs_dim), dtype=np.float32)
        high = np.inf * np.ones((self.obs_dim), dtype=np.float32)
        self.observation_space = gym.spaces.Box(
            low=low.astype(np.float32),
            high=high.astype(np.float32),
            dtype=np.float32)

        # actions are BoundedArray
        assert isinstance(act_spec, specs.BoundedArray), act_spec
        act_low = np.full(act_spec.shape, act_spec.minimum, dtype=np.float32)
        act_high = np.full(act_spec.shape, act_spec.maximum, dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=act_low,
            high=act_high,
            shape=act_spec.shape,
            dtype=np.float32)
        
        self._max_episode_steps = truncation_limit if truncation_limit < dmc_env._step_limit/dmc_env._num_repeats else dmc_env._step_limit/dmc_env._num_repeats
        self.counter = 0

    def reset(self):
        ts = self._env.reset() # TimeStep

        info = {
            "discount": float(ts.discount),
            "physics": ts.physics,
            "step_type": ts.step_type,
            "obs2": ts.observation
        }
        self.counter = 0
        return ts.observation, info
    
    def step(self, action):
        ts = self._env.step(action)
        self.counter += 1
        done = bool(ts.step_type == StepType.LAST)
        truncated = self.counter == self._max_episode_steps
        info = {
            "discount": float(ts.discount),
            "physics": ts.physics,
            "step_type": ts.step_type,
            "obs2": ts.observation
        }
        return ts.observation, ts.reward, done, truncated, info #Add done twice to account for gym api
    


class AsyncFactory:
    def __init__(self, truncation_limit=1000):
        self.truncation_limit = truncation_limit
    def make(self, idx):
        env = Factory()(
            name=self.name, 
            obs_type=self.obs_type, 
            frame_stack=self.frame_stack, 
            action_repeat=self.action_repeat, 
            seed=self.seed+10*idx,)
        return DPM_EnvWrapper(env, truncation_limit=self.truncation_limit)
    def __call__(self, name, num_workers, obs_type="states", frame_stack=1, action_repeat=1, seed=1):
        self.name = name
        self.obs_type = obs_type
        self.frame_stack = frame_stack
        self.action_repeat = action_repeat
        self.seed = seed
        return gym.vector.AsyncVectorEnv([lambda i=j: self.make(i) for j in range(num_workers)])


