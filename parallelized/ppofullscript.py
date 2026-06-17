import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import TanhTransform
import numpy as np
from torch.distributions import Normal

class Agent(nn.Module):
    def __init__(self, obs_dim, action_dim, num_layers, h_dim, device):
        super().__init__()
        self.device = device
        self.action_dim = action_dim
        self.pi_layers = nn.ModuleList([nn.Linear(obs_dim, h_dim, device=device)] + [nn.Linear(h_dim, h_dim, device=device) for _ in range(num_layers)]+[nn.Linear(h_dim, action_dim*2, device=device)])
        self.vf_layers = nn.ModuleList([nn.Linear(obs_dim, h_dim, device=device)] + [nn.Linear(h_dim, h_dim, device=device) for _ in range(num_layers)]+[nn.Linear(h_dim, 1, device=device)])

    def forward(self, obs):
        pi_x = obs
        vf_x = obs
        for i in range(len(self.pi_layers) - 1):
            pi_x = nn.functional.relu(self.pi_layers[i](pi_x))
            vf_x = nn.functional.relu(self.vf_layers[i](vf_x))
        pi_x = self.pi_layers[-1](pi_x)
        pi = pi_x[:, :self.action_dim]
        log_var = pi_x[:, self.action_dim:]
        vf = self.vf_layers[-1](vf_x)
        return pi, log_var, vf
    
    def predict(self, states, actions, rewards, advantages, old_log_probs, old_values):
        self.train()

        action_mean, log_variance, values = self.forward(states)
        action_std = (0.5*log_variance).exp()
        base_dist = Normal(action_mean, action_std)
        dist = TransformedDistribution(base_dist, [TanhTransform(cache_size=1)])
        log_probs = dist.log_prob(actions).sum(dim=-1) 
        target_values = advantages + old_values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        # 5. Clipped Actor Loss
        ratio = torch.exp(log_probs - old_log_probs)
        clip_eps = getattr(self.config, 'clip_eps', 0.2)
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # 6. Clipped Critic Loss (Prevents catastrophic value surges within deep backbones)
        v_loss_unclipped = (values - target_values) ** 2
        v_pred_clipped = old_values + torch.clamp(values - old_values, -clip_eps, clip_eps)
        v_loss_clipped = (v_pred_clipped - target_values) ** 2
        critic_loss = 0.5 * torch.max(v_loss_unclipped, v_loss_clipped).mean()

        # 7. Entropy Loss
        entropy_loss = dist.entropy().sum(dim=-1).mean() #Unknown method .entropy()
        
        
        value_coef = getattr(self.config, 'value_coef', 0.5)
        entropy_coef = getattr(self.config, 'entropy_coef', 0.01)
        
        total_loss = actor_loss + value_coef * critic_loss - entropy_coef * entropy_loss
        
        # 8. Parameter Optimization
        self.optimizer.zero_grad()
        total_loss.backward()
        
        max_grad_norm = getattr(self.config, 'max_grad_norm', 0.5)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()
        
        # Telemetry: Compute approximate KL for tracking drift
        with torch.no_grad():
            approx_kl = ((ratio - 1) - (log_probs - old_log_probs)).mean().item()
            
        return {
            "actor_loss": actor_loss.item(),
            "critic_loss": critic_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": total_loss.item(),
            "approx_kl": approx_kl
        }



