from collections import deque
import gym
import numpy as np
from dm_env import StepType, specs

class DMCGymWrapper(gym.Env):
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

    def __init__(self, dmc_env, obs_seq_len: int = 1):
        '''
        Observation sequence length is to make the environment return up to T
        observations, with the purpose of using a policy that does π(aₜ|sₜ₋ₜ₊₁:ₜ)
        instead of the usual π(aₜ|sₜ)
        '''
        super().__init__()
        self._env = dmc_env
        self.context_len = int(obs_seq_len)

        # Array(shape=(24,), dtype='float32'
        obs_spec = dmc_env.observation_spec()
        # BoundedArray(shape=(6,), dtype='float32', minimum=-1.0, maximum=1.0)
        act_spec = dmc_env.action_spec()
        self.obs_dim = obs_spec.shape[0]

        low = -np.inf * np.ones((self.context_len, self.obs_dim), dtype=np.float32)
        high = np.inf * np.ones((self.context_len, self.obs_dim), dtype=np.float32)
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

        self._obs_hist = deque(maxlen=self.context_len)
        self._valid_len = 0 # This is how many actual observations we have

    def push_obs(self, obs: np.ndarray):
        self._obs_hist.append(obs.astype(np.float32, copy=False))
        self._valid_len = min(self._valid_len + 1, self.context_len)

    def get_observation_stack(self) -> np.ndarray:
        '''
        This gets me an array of (context_len, obs_dim)
        Made up of k actual observations and zero-padding on the left side
        '''
        out = np.zeros((self.context_len, self.obs_dim), dtype=np.float32)
        k = len(self._obs_hist)
        if k > 0:
            out[-k:] = np.stack(self._obs_hist, axis=0)
        return out

    def reset(self):
        ts = self._env.reset() # TimeStep
        obs = np.array(ts.observation, copy=False).astype(np.float32, copy=False)

        self._obs_hist.clear()
        self._valid_len = 0
        self.push_obs(obs)

        obs_stack = self.get_observation_stack()
        info = {"valid_len": self._valid_len}
        return obs_stack

    def step(self, action):
        # SB3 gives numpy float32 actions in [-1, 1]
        ts = self._env.step(action)
        obs = np.array(ts.observation, copy=False).astype(np.float32, copy=False)
        self.push_obs(obs)

        obs_stack = self.get_observation_stack()
        reward = float(ts.reward)
        done = bool(ts.step_type == StepType.LAST)
        info = {
            "discount": float(ts.discount),
            "physics": ts.physics,
            "step_type": ts.step_type,
            "valid_len": self._valid_len
        }
        return obs_stack, reward, done, info
