from utils.utils import to_torch
import torch.nn as nn
import torch as th
import math

class MLPValueHead(nn.Module):
    """
    A standalone Critic (Value) network using an MLP. 
    Maintains the .optimizer attribute to match the original architecture's API.
    """
    def __init__(self, obs_dim, lr, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1)
        )
        # Keep the optimizer embedded to match previous ExtraDecoderForValueHead interface
        self.optimizer = th.optim.Adam(self.parameters(), lr=lr)
        
    def forward(self, x):
        return self.net(x)


class MLPActor(nn.Module):
    """
    A standalone Actor network using an MLP.
    """
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim)
        )
        
    def forward(self, x):
        return self.net(x)


class MlpWrapperPPO(nn.Module):
    """
    Drop-in replacement for the Transformer-based MDPWrapperPPO.
    Uses MLPs and only attends to the most recent state in the sequence.
    """
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        use_tb,
        mask_ratio,
        transformer_cfg,
        name,
        pretrained=False,
        vf_head_type="transformer_decoder", # Ignored in this baseline
        std_lr=1e-2,
    ):
        super().__init__()
        
        # Save standard configurations needed for PPO updates
        self.device = device
        self.config = transformer_cfg
        
        obs_dim = obs_shape[1]
        action_dim = action_shape[1]
        self.n_embd = self.config.n_embd
        # Initialize Completely Separate Backbones
        self.actor = MLPActor(obs_dim, action_dim, self.n_embd).to(device)
        self.value_head = MLPValueHead(obs_dim, lr, self.n_embd).to(device)
        
        # Trainable parameter for policy standard deviation
        self.log_std = nn.Parameter(th.zeros(action_dim, device=device) + math.log(transformer_cfg.initial_std))
        
        # Setup Policy Optimizer (incorporating std_lr exactly as before)
        self.opt = th.optim.Adam([
            {"params": self.actor.parameters(), "lr": lr},
            {"params": [self.log_std], "lr": std_lr}
        ])

    @property
    def std(self):
        return self.log_std.clamp(-20, 2).exp()

    def _prepare_inputs(self, states: th.Tensor, actions: th.Tensor, rewards: th.Tensor, training=True):
        """
        Kept for structural parity, but we no longer need to dynamically pad or mask
        since the MLP only looks at the current timestep. Returns data unmodified.
        """
        return states, actions, rewards

    def act(self, states, actions, rewards, values=None, pad_lengths=None): 
        with th.no_grad():
            # Only care about the most recent state for the MLP baseline
            current_state = states[:, -1, :] 
            
            # Policy output: Forward actor to predict action mean
            pi_mean = self.actor(current_state)
            
            # Construct policy action distribution
            dist = th.distributions.Normal(pi_mean, self.std)
            
            # Value output: Forward custom value network
            vf = self.value_head(current_state)
            
        return dist, vf

    def update_value_head(self, batch, global_step):
        metrics = dict()
        
        states, actions, latents, returns, advantages, rewards = to_torch(batch, self.device)

        # Isolate the latest step in the sequence
        current_states = states[:, -1, :]
        target_returns = returns[:, -1]

        # Predict values V(s) from current states
        predicted_values = self.value_head(current_states).squeeze(-1)
        
        # Mean Squared Error Loss against returns
        value_loss = nn.functional.mse_loss(predicted_values, target_returns)

        # Step value function optimizer
        self.value_head.optimizer.zero_grad()
        value_loss.backward()
        nn.utils.clip_grad_norm_(self.value_head.parameters(), max_norm=0.5)
        self.value_head.optimizer.step()

        metrics["value_loss"] = value_loss.item()
        metrics["v_mean"] = predicted_values.mean().item()
        
        return metrics

    def update_policy(self, batch, global_step):
        metrics = dict()

        states, actions, old_log_probs, returns, advantages, rewards = to_torch(batch, self.device)
        
        # Isolate the latest step in the sequence
        current_states = states[:, -1, :]
        target_actions = actions[:, -1]
        target_advantages = advantages[:, -1]
        target_old_log_probs = old_log_probs[:, -1]

        target_advantages = (target_advantages - target_advantages.mean()) / (target_advantages.std() + 1e-8)

        # Forward current state through Actor
        pi_mean = self.actor(current_states)

        # Use the trainable standard deviation
        current_std = self.std
        dist = th.distributions.Normal(pi_mean, current_std)
        log_probs = dist.log_prob(target_actions).sum(dim=-1)

        log_ratio = log_probs - target_old_log_probs
        ratio = th.exp(log_ratio)
        
        clip_coef = getattr(self.config, "clip_coef", 0.2)
        
        surr1 = ratio * target_advantages
        surr2 = th.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * target_advantages
        policy_loss = -th.min(surr1, surr2).mean()

        entropy = dist.entropy().mean()
        ent_coef = getattr(self.config, "ent_coef", 0.0)
        
        loss = policy_loss - ent_coef * entropy

        self.opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=0.5)
        nn.utils.clip_grad_norm_([self.log_std], max_norm=0.5)
        self.opt.step()

        with th.no_grad():
            approx_kl = ((ratio - 1) - log_ratio).mean().item()
            clipfrac = ((ratio - 1.0).abs() > clip_coef).float().mean().item()

        metrics["policy_loss"] = policy_loss.item()
        metrics["entropy"] = entropy.item()
        metrics["approx_kl"] = approx_kl
        metrics["clipfrac"] = clipfrac
        metrics["pi_mean"] = pi_mean.mean().item()
        metrics["std"] = current_std.mean().item()

        return metrics