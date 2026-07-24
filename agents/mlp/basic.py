import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import utils

class MlpBC(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        # Simple MLP network for Behavioral Cloning
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, action_dim),
            nn.Tanh()  # Matches the original MaskedDP action_head output
        )

    def forward(self, obs):
        # obs shape can be [Batch, Time, ObsDim]
        # nn.Linear automatically applies across the last dimension
        return self.net(obs)


class MlpAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        use_tb,
        mask_ratio,      # Kept for drop-in compatibility
        transformer_cfg, # Kept for drop-in compatibility
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb

        # Initialize the simpler MLP model
        self.model = MlpBC(obs_shape[0], action_shape[0]).to(device)
        
        # Optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(
            "number of parameters: %e" % sum(p.numel() for p in self.model.parameters())
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def eval_validation(self, val_iter, step=None):
        '''Evaluates validation BC loss'''
        metrics = dict()
        batch = next(val_iter)
        
        # Unpack identically to MaskedDP
        obs, action, _, _, _, _ = utils.to_torch(batch, self.device)
        
        with torch.no_grad():
            pred_a = self.model(obs)
            # MSE loss mimicking the actions
            action_loss = F.mse_loss(pred_a, action)

        metrics["val_action_loss"] = action_loss.item()
        # Mock mask/state metrics so loggers expecting these keys don't break
        metrics["val_mask_loss"] = 0.0
        metrics["val_state_loss"] = 0.0

        return metrics

    def update(self, replay_iter, step=None):
        metrics = dict()

        batch = next(replay_iter)
        obs, action, _, _, _, _ = utils.to_torch(batch, self.device)
        
        # 1. Forward pass (states to actions)
        pred_a = self.model(obs)
        
        # 2. Compute BC Loss (MSE)
        action_loss = F.mse_loss(pred_a, action)

        # 3. Optimize
        self.opt.zero_grad(set_to_none=True)
        action_loss.backward()
        self.opt.step()

        metrics["action_loss"] = action_loss.item()
        # Mock mask/state metrics so loggers expecting these keys don't break
        metrics["mask_loss"] = action_loss.item()
        metrics["state_loss"] = action_loss.item()

        return metrics