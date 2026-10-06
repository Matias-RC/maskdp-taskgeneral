import gymnasium as gym
import numpy as np
from gymnasium import spaces


class ContextWindowWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, context_len: int):
        super().__init__(env)
        assert context_len >= 1
        assert isinstance(env.observation_space, spaces.Box), "Only Box observations supported"

        self.k = context_len
        self.obs_shape = env.observation_space.shape

        act_space = env.action_space
        if isinstance(act_space, spaces.Discrete):
            self._discrete = True
            self.act_dim = int(act_space.n)
        elif isinstance(act_space, spaces.Box):
            self._discrete = False
            self.act_dim = int(np.prod(act_space.shape))
        else:
            raise NotImplementedError(f"Unsupported action space: {act_space}")

        # Rolling buffers
        self._obs_buf = np.zeros((self.k, *self.obs_shape), dtype=np.float32)
        self._act_buf = np.zeros((self.k, self.act_dim), dtype=np.float32)
        self._rew_buf = np.zeros((self.k, 1), dtype=np.float32)
        self._mask_buf = np.zeros((self.k,), dtype=np.float32)

        # Bounds are infinite: normalization wrappers change the observation range anyway
        self.observation_space = spaces.Dict({
            "obs": spaces.Box(-np.inf, np.inf, (self.k, *self.obs_shape), np.float32),
            "actions": spaces.Box(-np.inf, np.inf, (self.k, self.act_dim), np.float32),
            "rewards": spaces.Box(-np.inf, np.inf, (self.k, 1), np.float32),
            "mask": spaces.Box(0.0, 1.0, (self.k,), np.float32),
        })

    def _encode_action(self, action) -> np.ndarray:
        if self._discrete:
            one_hot = np.zeros(self.act_dim, dtype=np.float32)
            one_hot[int(action)] = 1.0
            return one_hot
        return np.asarray(action, dtype=np.float32).reshape(-1)

    def _push(self, obs, action_vec, reward):
        # Shift everything one slot to the left, write newest at the end
        self._obs_buf[:-1] = self._obs_buf[1:]
        self._act_buf[:-1] = self._act_buf[1:]
        self._rew_buf[:-1] = self._rew_buf[1:]
        self._mask_buf[:-1] = self._mask_buf[1:]

        self._obs_buf[-1] = obs
        self._act_buf[-1] = action_vec
        self._rew_buf[-1, 0] = reward
        self._mask_buf[-1] = 1.0

    def _get_obs(self):
        return {
            "obs": self._obs_buf.copy(),
            "actions": self._act_buf.copy(),
            "rewards": self._rew_buf.copy(),
            "mask": self._mask_buf.copy(),
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._obs_buf.fill(0.0)
        self._act_buf.fill(0.0)
        self._rew_buf.fill(0.0)
        self._mask_buf.fill(0.0)
        self._push(obs, np.zeros(self.act_dim, dtype=np.float32), 0.0)
        return self._get_obs(), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._push(obs, self._encode_action(action), float(reward))
        return self._get_obs(), reward, terminated, truncated, info
    
def make_env(env_name, seed, gamma=0.99, context_len=16):
    def thunk():
        env = gym.make(env_name)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda r: np.clip(r, -10, 10))
        env = ContextWindowWrapper(env, context_len)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env
    return thunk


def create_vector_env(env_name, num_envs, seed, gamma=0.99, context_len=16):
    env_fns = [make_env(env_name, seed + i, gamma, context_len) for i in range(num_envs)]
    return gym.vector.SyncVectorEnv(env_fns)