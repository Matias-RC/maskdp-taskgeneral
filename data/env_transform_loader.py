"""
Applies the envs/mujoco_gym_context.py (and envs/dmc_gymnasium.py) preprocessing
to offline batches, so a model pretrained on the replay loader sees the same
inputs it will see from the gym env at finetuning time.

Env wrapper stack being reproduced (outermost last):
    NormalizeObservation              obs -> (obs - mean) / sqrt(var + eps)
    TransformObservation              obs -> clip(obs, -clip_obs, clip_obs)
    NormalizeReward(gamma)            r   -> r / sqrt(var(G) + eps),  G_t = gamma * G_{t-1} * (1 - term) + r_t
    TransformReward                   r   -> clip(r, -clip_reward, clip_reward)
    ContextWindowWrapper (optional)   slot t holds (obs_t, a_{t-1}, r_{t-1}); see align_to_context

The statistics are gymnasium RunningMeanStd objects, i.e. exactly the obs_rms /
return_rms that live inside the env wrappers, so after pretraining they can be
written back with load_into_env() (or saved with save() and restored later).

Two ways to obtain the statistics:
  * fit_from_episodes(): one pass over the .npz episodes, replaying them as a
    single env stream would. Exact, ~3 ms per episode. Stats are then frozen.
  * running (default if not fitted): updated from every batch, like the env
    does online. Returns are only accumulated inside each sampled window
    (windows start mid-episode), so var(G) is underestimated; prefer fitting.
"""
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
from gymnasium.wrappers.normalize import NormalizeObservation, NormalizeReward, RunningMeanStd


def _rms_state(rms):
    return {"mean": np.asarray(rms.mean, dtype=np.float64).copy(),
            "var": np.asarray(rms.var, dtype=np.float64).copy(),
            "count": float(rms.count)}


def _load_rms(rms, state):
    rms.mean = np.asarray(state["mean"], dtype=np.float64).copy()
    rms.var = np.asarray(state["var"], dtype=np.float64).copy()
    rms.count = float(state["count"])


def _iter_wrappers(env):
    """Walk the wrapper chain without relying on Wrapper.__getattr__ forwarding."""
    while env is not None:
        yield env
        env = vars(env).get("env")


class EnvTransformLoader:
    """Iterator over a replay loader that yields env-preprocessed batches.

    Batches keep the replay loader format (obs, action, reward, discount, next_obs, extra),
    so it is a drop-in replacement for iter(train_loader) in agent.update().
    """

    def __init__(self, loader, obs_dim, gamma=0.99, epsilon=1e-8, clip_obs=10.0,
                 clip_reward=10.0, action_low=-1.0, action_high=1.0, align_to_context=False):
        self.loader = loader
        self.gamma = gamma
        self.epsilon = epsilon
        self.clip_obs = clip_obs
        self.clip_reward = clip_reward
        self.action_low = action_low
        self.action_high = action_high
        self.align_to_context = align_to_context

        # Same objects and defaults as NormalizeObservation / NormalizeReward
        self.obs_rms = RunningMeanStd(shape=(obs_dim,))
        self.return_rms = RunningMeanStd(shape=())
        self.update_stats = True
        self._it = None

    # ------------------------------------------------------------ fitting
    def fit_from_episodes(self, replay_dir=None):
        """Replay every episode in replay_dir through the normalizers, then freeze them.

        Mirrors the env: obs_rms sees the reset obs and every step obs, return_rms sees
        G after every step, and G is only reset on termination (gymnasium 0.29 does not
        reset it on truncation, so it carries over between DMC episodes).
        """
        replay_dir = Path(replay_dir or self.loader.dataset._replay_dir)
        fns = sorted(replay_dir.rglob("*.npz"))
        if not fns:
            raise ValueError(f"No .npz episodes found in {replay_dir}")

        G = 0.0
        for fn in fns:
            with np.load(fn) as ep:
                obs = ep["observation"].astype(np.float64)
                rew = ep["reward"][1:].reshape(-1).astype(np.float64)  # index 0 is the dummy reset transition
                term = ep["discount"][1:].reshape(-1) == 0.0
            self.obs_rms.update(obs)
            returns = np.empty_like(rew)
            for t in range(rew.shape[0]):
                G = G * self.gamma * (1.0 - term[t]) + rew[t]
                returns[t] = G
            self.return_rms.update(returns)

        self.update_stats = False
        print(f"[EnvTransformLoader] fitted on {len(fns)} episodes | "
              f"obs count {self.obs_rms.count:.0f} | return std {np.sqrt(self.return_rms.var):.4f}")
        return self

    # ----------------------------------------------------------- transforms
    def normalize_obs(self, obs):
        return np.clip((obs - self.obs_rms.mean) / np.sqrt(self.obs_rms.var + self.epsilon),
                       -self.clip_obs, self.clip_obs)

    def normalize_reward(self, rew):
        return np.clip(rew / np.sqrt(self.return_rms.var + self.epsilon),
                       -self.clip_reward, self.clip_reward)

    def _update_from_batch(self, obs, rew, discount):
        self.obs_rms.update(obs.reshape(-1, obs.shape[-1]))
        # Discounted return accumulated from the start of each window
        rew = rew.reshape(rew.shape[0], rew.shape[1])
        not_term = (discount.reshape(rew.shape) != 0.0).astype(np.float64)
        G = np.zeros(rew.shape[0])
        returns = np.empty_like(rew, dtype=np.float64)
        for t in range(rew.shape[1]):
            G = G * self.gamma * not_term[:, t] + rew[:, t]
            returns[:, t] = G
        self.return_rms.update(returns.reshape(-1))

    def transform(self, batch):
        obs, action, reward, discount, next_obs, extra = batch
        obs_np, rew_np = obs.numpy().astype(np.float64), reward.numpy().astype(np.float64)
        if self.update_stats:
            self._update_from_batch(obs_np, rew_np, discount.numpy())

        obs = torch.from_numpy(self.normalize_obs(obs_np)).float()
        next_obs = torch.from_numpy(self.normalize_obs(next_obs.numpy().astype(np.float64))).float()
        reward = torch.from_numpy(self.normalize_reward(rew_np)).float()
        # DMCEnv clips actions before stepping; dataset actions are already in bounds
        action = action.clamp(self.action_low, self.action_high)

        if self.align_to_context:
            # ContextWindowWrapper pairs obs_t with the action/reward that produced it
            action = torch.cat([torch.zeros_like(action[:, :1]), action[:, :-1]], dim=1)
            reward = torch.cat([torch.zeros_like(reward[:, :1]), reward[:, :-1]], dim=1)

        return obs, action, reward, discount, next_obs, extra

    # ------------------------------------------------------------ iteration
    def __iter__(self):
        self._it = iter(self.loader)
        return self

    def __next__(self):
        if self._it is None:
            self._it = iter(self.loader)
        return self.transform(next(self._it))

    # ----------------------------------------------------------- save / load
    def state_dict(self):
        return {
            "obs_rms": _rms_state(self.obs_rms),
            "return_rms": _rms_state(self.return_rms),
            "gamma": self.gamma,
            "epsilon": self.epsilon,
            "clip_obs": self.clip_obs,
            "clip_reward": self.clip_reward,
            "align_to_context": self.align_to_context,
        }

    def load_state_dict(self, state, freeze=True):
        _load_rms(self.obs_rms, state["obs_rms"])
        _load_rms(self.return_rms, state["return_rms"])
        for k in ("gamma", "epsilon", "clip_obs", "clip_reward", "align_to_context"):
            setattr(self, k, state[k])
        self.update_stats = not freeze

    def save(self, path):
        torch.save(self.state_dict(), path)

    @staticmethod
    def load_into_env(state, env, freeze=False, prior_count=None):
        """Write saved stats into the NormalizeObservation / NormalizeReward wrappers of env.

        state      : state_dict() or a path saved with save()
        env        : a single wrapped env or a SyncVectorEnv (AsyncVectorEnv lives in subprocesses
                     and cannot be patched from here)
        freeze     : stop the env from updating the stats any further
        prior_count: cap on the loaded counts (scalar, or {"obs": c, "return": c}). RunningMeanStd
                     is a cumulative average, so after n online steps the online data has weight
                     n / (count + n); with the full pretraining count (~1e7) the stats never move.
                     Mean/var are kept, only their weight as a prior is reduced.
        """
        if not isinstance(state, dict):
            state = torch.load(state, weights_only=False)
        envs = env.envs if hasattr(env, "envs") else [env]
        if prior_count is not None and not isinstance(prior_count, Mapping):
            prior_count = {"obs": prior_count, "return": prior_count}
        prior_count = dict(prior_count or {})

        for e in envs:
            found_obs = found_rew = False
            for w in _iter_wrappers(e):
                if isinstance(w, NormalizeObservation):
                    _load_rms(w.obs_rms, state["obs_rms"])
                    if prior_count.get("obs") is not None:
                        w.obs_rms.count = min(w.obs_rms.count, float(prior_count["obs"]))
                    w.epsilon = state["epsilon"]
                    if freeze:
                        w.obs_rms.update = lambda x: None
                    found_obs = True
                elif isinstance(w, NormalizeReward):
                    _load_rms(w.return_rms, state["return_rms"])
                    if prior_count.get("return") is not None:
                        w.return_rms.count = min(w.return_rms.count, float(prior_count["return"]))
                    w.epsilon = state["epsilon"]
                    if w.gamma != state["gamma"]:
                        print(f"WARNING: env gamma {w.gamma} != pretraining gamma {state['gamma']}")
                    if freeze:
                        w.return_rms.update = lambda x: None
                    found_rew = True
            if not (found_obs and found_rew):
                raise ValueError("NormalizeObservation / NormalizeReward not found in the env wrapper chain")
        print(f"[EnvTransformLoader] loaded into {len(envs)} env(s) | freeze={freeze} | "
              f"prior counts obs {min(state['obs_rms']['count'], prior_count.get('obs') or np.inf):.0f}, "
              f"return {min(state['return_rms']['count'], prior_count.get('return') or np.inf):.0f}")
        return env
