"""
PPO for the causal triplet Transformer actor-critic.

Algorithmically this mirrors algorithms/BaselinePPOTrainer.py step for step:
same rollout length, truncation bootstrapping, GAE, advantage normalisation,
clipped surrogate, unclipped MSE value loss, combined loss, epochs/minibatches,
linear LR annealing and checkpoint cadence.

What changes, and why:

* Observations are the context windows produced by
  envs/mujoco_gym_context.ContextWindowWrapper (dict of obs/actions/rewards/mask).
  Each PPO sample is the whole window that the policy saw when it acted, and the
  model is always queried with last_only=True. Decoding every timestep of a stored
  window would be cheaper, but the earlier positions see a shorter, re-positioned
  history than they did at rollout time, so their log-probs would not match the
  stored ones and the PPO ratio would be wrong.

* Actor and critic share the encoder/decoder trunk, so one forward pass yields
  mean, std and value. The rollout stores V(s_t) from that same pass (identical to
  the baseline recomputing it before the update, since parameters have not moved),
  and the update computes the policy and value losses from a single pass.

* One optimizer with three parameter groups replaces the two optimizers:
  actor head (policy_lr), critic head (value_lr) and the shared trunk (trunk_lr,
  defaults to policy_lr). Adam/RMSprop are per-parameter, so for the two heads
  this is exactly equivalent to two optimizers. Because the trunk receives both
  gradients, vf_coef now genuinely balances the two losses, and the gradient is
  clipped jointly over the whole model (a separately clipped trunk would
  have no clean counterpart in the baseline).
"""
import os
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Mapping

import numpy as np
import torch
import torch.distributions as D
import torch.nn.functional as F
import torch.optim as optim
import wandb
from omegaconf import DictConfig, OmegaConf
from torch import nn

from agents.actor_critic.transformer_actor_critic import CausalTripletTransformerAC
from utils.rl_utils import compute_gae

CTX_KEYS = ("obs", "actions", "rewards", "mask")


def to_dict(cfg):
    if cfg is None:
        return {}
    if isinstance(cfg, DictConfig):
        return OmegaConf.to_container(cfg, resolve=True)
    if isinstance(cfg, Mapping):
        return dict(cfg)
    return dict(vars(cfg))


def find_wrapper_attr(env, name):
    """Walk the wrapper chain without relying on Wrapper.__getattr__ forwarding."""
    while env is not None:
        if name in vars(env):
            return vars(env)[name]
        env = vars(env).get("env")
    return None


class RolloutBuffer:
    """Fixed-size on-device storage for one rollout of T steps x N envs."""

    def __init__(self, T, N, L, obs_dim, act_dim, device):
        def z(*shape):
            return torch.zeros(*shape, dtype=torch.float32, device=device)

        self.ctx = {
            "obs": z(T, N, L, obs_dim),
            "actions": z(T, N, L, act_dim),
            "rewards": z(T, N, L, 1),
            "mask": z(T, N, L),
        }
        self.actions = z(T, N, act_dim)
        self.log_probs = z(T, N)
        self.values = z(T, N)
        self.rewards = z(T, N)
        self.dones = z(T, N)

    def store(self, t, ctx, action, log_prob, value, reward, done):
        for k in CTX_KEYS:
            self.ctx[k][t] = ctx[k]
        self.actions[t] = action
        self.log_probs[t] = log_prob
        self.values[t] = value
        self.rewards[t] = reward
        self.dones[t] = done


class TransformerTrainer(ABC):
    def __init__(self, envs, config):
        self.envs = envs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = config

        obs_space = envs.single_observation_space
        self.context_len, self.obs_dim = obs_space["obs"].shape
        self.act_dim = envs.single_action_space.shape[0]

        # Model config: the window length is dictated by the env wrapper.
        self.model_cfg = to_dict(getattr(config, "model", None))
        cfg_len = self.model_cfg.get("context_len", self.context_len)
        if cfg_len != self.context_len:
            print(f"WARNING: model.context_len={cfg_len} != env window {self.context_len}; using the env's.")
        self.model_cfg["context_len"] = self.context_len
        for key in ("attn_pdrop", "resid_pdrop"):
            if self.model_cfg.get(key, 0.0) != 0.0:
                print(f"WARNING: {key}>0 makes rollout and update log-probs disagree under PPO.")

        self.model = CausalTripletTransformerAC(self.obs_dim, self.act_dim, self.model_cfg).to(self.device)

        # Parameter groups: actor head / critic head / shared trunk.
        head_ids = {id(p) for p in self.model.actor.parameters()} | {id(p) for p in self.model.critic.parameters()}
        trunk_params = [p for p in self.model.parameters() if id(p) not in head_ids]
        policy_lr = getattr(self.config, "policy_lr", 1e-3)
        value_lr = getattr(self.config, "value_lr", 1e-3)
        trunk_lr = getattr(self.config, "trunk_lr", None) or policy_lr
        groups = [
            {"name": "actor", "params": list(self.model.actor.parameters()), "lr": policy_lr},
            {"name": "critic", "params": list(self.model.critic.parameters()), "lr": value_lr},
            {"name": "trunk", "params": trunk_params, "lr": trunk_lr},
        ]
        for g in groups:
            g["base_lr"] = g["lr"]

        optimizer_name = getattr(self.config, "optim", "adam").lower()
        eps = getattr(self.config, "optim_eps", 1e-5)
        if optimizer_name == "rmsprop":
            self.optimizer = optim.RMSprop(groups, eps=eps, alpha=getattr(self.config, "optim_alpha", 0.99))
        elif optimizer_name == "adam":
            self.optimizer = optim.Adam(groups, eps=eps)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

        self.buffer = RolloutBuffer(
            self.config.rollout_steps, self.config.num_envs,
            self.context_len, self.obs_dim, self.act_dim, self.device,
        )

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"Transformer AC: {n_params / 1e6:.3f}M params | window={self.context_len} | {self.model_cfg}")

    # ----------------------------------------------------------- helpers
    def to_tensors(self, obs):
        """Batched dict obs from the vector env -> dict of float tensors on device."""
        return {k: torch.as_tensor(np.asarray(obs[k]), dtype=torch.float32, device=self.device) for k in CTX_KEYS}

    @staticmethod
    def stack_obs(obs_list):
        return {k: np.stack([o[k] for o in obs_list]) for k in CTX_KEYS}

    def policy_value(self, ctx):
        """One shared forward pass on the newest timestep -> mean [B,A], std [B,A], value [B]."""
        mean, std, value = self.model(ctx["obs"], ctx["actions"], ctx["rewards"], ctx["mask"], last_only=True)
        return mean.squeeze(1), std.squeeze(1), value.squeeze(1)

    def get_lr(self, name):
        return next(g["lr"] for g in self.optimizer.param_groups if g["name"] == name)

    def save_checkpoint(self, step=None):
        save_path = self.config.save_path
        if step is not None:
            base, ext = os.path.splitext(save_path)
            save_path = f"{base}_{step}{ext}"

        save_dir = os.path.dirname(save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        obs_rms = find_wrapper_attr(self.envs.envs[0], "obs_rms")
        if obs_rms is None:
            print("WARNING: Could not find obs_rms in the wrapper chain.")

        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "model_config": self.model_cfg,
            "obs_dim": self.obs_dim,
            "act_dim": self.act_dim,
            "config": to_dict(self.config),
            "obs_rms_mean": None if obs_rms is None else obs_rms.mean,
            "obs_rms_var": None if obs_rms is None else obs_rms.var,
        }
        torch.save(checkpoint, save_path)
        print(f"\n*** Checkpoint saved: {save_path} ***\n")

    def linear_lr_schedule(self, step, total_steps):
        """Linearly anneals every param group down to 0 (same formula as the baseline)."""
        frac = 1.0 - (step - 1.0) / total_steps
        for g in self.optimizer.param_groups:
            g["lr"] = frac * g["base_lr"]

    # ----------------------------------------------------------- rollout
    @torch.no_grad()
    def rollout(self, obs, recent_returns, global_step):
        buf = self.buffer
        gamma = self.config.gamma

        for t in range(self.config.rollout_steps):
            ctx = self.to_tensors(obs)
            mean, std, value = self.policy_value(ctx)
            dist = D.Normal(mean, std)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=-1)

            next_obs, reward, terminated, truncated, infos = self.envs.step(action.cpu().numpy())
            reward = np.asarray(reward, dtype=np.float32).copy()

            # 1. Truncations count as 'done' so GAE does not flow across the reset.
            done = np.logical_or(terminated, truncated)

            # 2. Pure truncations (time limit, not a real termination): bootstrap V of the
            #    final window into the reward. All such envs are evaluated in one batch.
            if "_final_observation" in infos:
                idxs = [
                    i for i, has_final in enumerate(infos["_final_observation"])
                    if has_final and truncated[i] and not terminated[i]
                ]
                if idxs:
                    final_ctx = self.to_tensors(self.stack_obs([infos["final_observation"][i] for i in idxs]))
                    _, _, terminal_values = self.policy_value(final_ctx)
                    reward[idxs] += gamma * terminal_values.cpu().numpy()

            if "final_info" in infos:
                for idx, has_final_info in enumerate(infos["_final_info"]):
                    if has_final_info:
                        ep_info = infos["final_info"][idx]
                        if ep_info is not None and "episode" in ep_info:
                            recent_returns.append(float(ep_info["episode"]["r"]))

            buf.store(
                t, ctx, action, log_prob, value,
                torch.as_tensor(reward, device=self.device),
                torch.as_tensor(done, dtype=torch.float32, device=self.device),
            )

            obs = next_obs
            global_step += self.config.num_envs

        return obs, global_step, recent_returns

    # ------------------------------------------------------------- train
    def train(self):
        obs, _ = self.envs.reset(seed=self.config.seed)
        global_step = 0
        update_step = 0
        recent_returns = deque(maxlen=100)

        save_interval = getattr(self.config, "save_interval", 1000000)
        next_save_step = save_interval

        while global_step < self.config.total_steps:
            if getattr(self.config, "anneal_lr", True):
                self.linear_lr_schedule(global_step, self.config.total_steps)

            obs, global_step, recent_returns = self.rollout(obs, recent_returns, global_step)

            metrics = self.update(self.buffer, obs)
            update_step += 1
            metrics["train/policy_lr"] = self.get_lr("actor")
            metrics["train/value_lr"] = self.get_lr("critic")
            metrics["train/trunk_lr"] = self.get_lr("trunk")

            if len(recent_returns) > 0:
                smoothed_return = np.mean(recent_returns)
                metrics["train/smoothed_return"] = smoothed_return
                metrics["global_step"] = global_step

                if getattr(self.config, "use_wandb", False):
                    wandb.log(metrics, step=global_step)

                print(f"Update: {update_step} | Step: {global_step} | Return: {smoothed_return:.2f}")

            if global_step >= next_save_step:
                self.save_checkpoint(step=global_step)
                next_save_step += save_interval

        self.save_checkpoint(step="final")

    @abstractmethod
    def update(self, buffer, next_obs):
        pass


class TransformerPPOTrainer(TransformerTrainer):
    def update(self, buf, next_obs):
        with torch.no_grad():
            _, _, next_value = self.policy_value(self.to_tensors(next_obs))

        gae_lambda = getattr(self.config, "gae_lambda", 0.95)
        advantages, returns = compute_gae(
            buf.rewards, buf.values, buf.dones, next_value, self.config.gamma, gae_lambda
        )

        y_pred, y_true = buf.values.cpu().numpy(), returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # Flatten [T, N, ...] -> [T*N, ...]; each sample keeps its full context window.
        b_ctx = {k: v.reshape(-1, *v.shape[2:]) for k, v in buf.ctx.items()}
        b_actions = buf.actions.reshape(-1, self.act_dim)
        b_old_log_probs = buf.log_probs.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)

        b_advantages = (b_advantages - b_advantages.mean()) / (b_advantages.std() + 1e-8)

        batch_size = b_actions.shape[0]
        minibatch_size = getattr(self.config, "minibatch_size", 64)
        update_epochs = getattr(self.config, "update_epochs", 10)
        clip_coef = getattr(self.config, "clip_eps", 0.2)
        ent_coef = getattr(self.config, "ent_coef", 0.0)
        vf_coef = getattr(self.config, "vf_coef", 0.5)
        max_grad_norm = getattr(self.config, "max_grad_norm", 0.5)

        totals = dict(policy_loss=0.0, value_loss=0.0, entropy=0.0, kl=0.0, clip_frac=0.0, grad_norm=0.0)
        num_updates = 0

        self.model.train()
        for _ in range(update_epochs):
            b_inds = torch.randperm(batch_size, device=self.device)
            for start in range(0, batch_size, minibatch_size):
                mb_inds = b_inds[start:start + minibatch_size]

                # Single shared pass: policy and value from the same encoded window.
                mean, std, mb_values = self.policy_value({k: v[mb_inds] for k, v in b_ctx.items()})
                dist = D.Normal(mean, std)
                new_log_probs = dist.log_prob(b_actions[mb_inds]).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                log_ratio = new_log_probs - b_old_log_probs[mb_inds]
                ratio = torch.exp(log_ratio)

                with torch.no_grad():
                    approx_kl = ((ratio - 1) - log_ratio).mean().item()
                    clip_frac = ((ratio < 1 - clip_coef) | (ratio > 1 + clip_coef)).float().mean().item()

                mb_adv = b_advantages[mb_inds]
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                value_loss = F.mse_loss(mb_values, b_returns[mb_inds])

                loss = policy_loss - (ent_coef * entropy) + (value_loss * vf_coef)

                self.optimizer.zero_grad()
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()

                totals["policy_loss"] += policy_loss.item()
                totals["value_loss"] += value_loss.item()
                totals["entropy"] += entropy.item()
                totals["kl"] += approx_kl
                totals["clip_frac"] += clip_frac
                totals["grad_norm"] += grad_norm.item()
                num_updates += 1

        return {
            "train/policy_loss": totals["policy_loss"] / num_updates,
            "train/value_loss": totals["value_loss"] / num_updates,
            "train/entropy": totals["entropy"] / num_updates,
            "train/explained_variance": explained_var,
            "train/approx_kl": totals["kl"] / num_updates,
            "train/clip_fraction": totals["clip_frac"] / num_updates,
            "train/grad_norm": totals["grad_norm"] / num_updates,
        }