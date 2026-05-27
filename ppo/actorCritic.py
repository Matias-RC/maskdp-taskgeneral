import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import utils
from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block, CausalSelfAttention, mySequential
from agent.mdp import MaskedDP
import math

class MaskDPA2C(nn.Module):
    def __init__(
        self,
        cfg,
        obs_shape,
        action_shape,
        device,
        lr,
        use_tb,
        path=None,
    ):
        super().__init__()
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.PE_type = cfg.PE_type

        # init from snapshot
        payload = None
        if path is not None:
            print("loading existing model...")
            payload = torch.load(path)
            self.config = payload["cfg"]
        else:
            self.config = cfg
        # Incorportate the backbone
        if cfg.PE_type == "triplet":
            from agent.mdpAA_triplet import MaskDPTriplet, TripletDPAgent
            self.backbone = MaskDPTriplet(obs_shape, action_shape, self.config)
        elif cfg.PE_type == "joint":
            from agent.mdpAA_jointPE import MaskDPJointPE, JointDPAgent
            self.backbone = MaskDPJointPE(obs_shape, action_shape, self.config)
        else:
            raise NotImplementedError
        print("number of parameters: %e", sum(p.numel() for p in self.backbone.parameters()))
        if path is not None:
            self.backbone.load_state_dict(payload["model"])
            print("Payload succesfully loaded...")

        # Introduce stochasticity
        self.actor_logstd = nn.Parameter(torch.zeros(1, action_shape, device=self.device))
        # 4. Determine layer-freezing behavior
        if getattr(cfg, "freeze_backbone", False):
            self.freeze_transformer_core()
            # Optimize only the pre-trained decoder heads and the policy variance parameter
            trainable_params = (
                list(self.backbone.action_head.parameters()) + 
                list(self.backbone.reward_head.parameters()) + 
                [self.actor_logstd]
            )
        else:
            # Complete model joint fine-tuning
            trainable_params = list(self.parameters())

        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

    def freeze_transformer_core(self):
        """Freezes core Transformer blocks and token embedding projections, leaving heads open."""
        for param in self.backbone.encoder_blocks.parameters():
            param.requires_grad = False
        for param in self.backbone.decoder_blocks.parameters():
            param.requires_grad = False
        for param in self.backbone.state_embed.parameters():
            param.requires_grad = False
        for param in self.backbone.action_embed.parameters():
            param.requires_grad = False
        for param in self.backbone.reward_embed.parameters():
            param.requires_grad = False
        print("Transformer core frozen. Optimizing pre-trained heads for policy/value conversion.")

    def _get_predictions(self, seq_s, seq_a, seq_r):
        """
        Internal helper that runs the forward sequence loop to extract policy 
        and value predictions from the final sequence position in the decoder.
        """
        # Run encoder using last_two_masking
        if self.PE_type == "joint":
            latent, _, ids_restore, num_mods = self.backbone.forward_encoder(
                seq_s, seq_a, seq_r, mask_ratio=None, finetune_input=True
            )
        else:
            latent, _, ids_restore = self.backbone.forward_encoder(
                seq_s, seq_a, seq_r, mask_ratio=None, finetune_input=True
            )
        
        # Reconstruct sequence representation space for decoder input
        mask_tokens = self.backbone.mask_token.repeat(
            latent.shape[0], ids_restore.shape[1] - latent.shape[1], 1
        )
        x_ = torch.cat([latent, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, latent.shape[2])
        )
        
        # Interleave embeddings & add positional information
        s = self.backbone.decoder_state_embed(x[:, ::3])
        a = self.backbone.decoder_action_embed(x[:, 1::3])
        r = self.backbone.decoder_reward_embed(x[:, 2::3])
        x = torch.stack([s, a, r], dim=1).permute(0, 2, 1, 3).reshape_as(x)
        x = self.backbone.apply_masking(x)

        # Transform using decoder self-attention layers
        for blk in self.backbone.decoder_blocks:
            x = blk(x, self.backbone.attn_mask)

        # Map decoded outputs to head spaces
        pred_a = self.backbone.action_head(x[:, 1::3])
        pred_r = self.backbone.reward_head(x[:, 2::3])

        # Slice out the final timestep prediction indices for the current step context
        action_mean = pred_a[:, -1, :]
        value_pred = pred_r[:, -1, :] # The pre-trained reward head now predicts expected return V(s)

        return action_mean, value_pred
    
    def act(self, seq_s, seq_a, seq_r, deterministic=False):
        self.backbone.eval()
        with torch.no_grad():
            action_mean, value = self._get_predictions(seq_s, seq_a, seq_r)
            
        if deterministic:
            action = action_mean
        else:
            action_std = self.actor_logstd.exp().expand_as(action_mean)
            dist = torch.distributions.Normal(action_mean, action_std)
            action = dist.sample()
            
        return action, value

    def _forward_transformer(self, states, actions, rewards):
        """Unified causal forward pass across sequence contexts without random sequence masking."""
        batch_size, T, _ = states.shape
        num_mods = 3 if rewards is not None else 2
        seq_len = num_mods * T

        original_masking = self.backbone.random_masking
        
        def sequential_no_masking(x, mask_ratio):
            N, L, D = x.shape
            mask = torch.zeros([N, L], device=x.device)
            ids_restore = torch.arange(L, device=x.device).expand(N, -1)
            return x, mask, ids_restore
            
        self.backbone.random_masking = sequential_no_masking
        
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=self.device))
        causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
        
        original_attn_mask = self.backbone.attn_mask
        self.backbone.attn_mask = causal_mask
        
        if self.PE_type == "joint":
            x, _, ids_restore, mods = self.backbone.forward_encoder(
                states, actions, rewards, mask_ratio=0.0, finetune_input=False
            )
            _, pred_a, pred_v = self.backbone.forward_decoder(x, ids_restore, mods)
        else:
            x, _, ids_restore = self.backbone.forward_encoder(
                states, actions, rewards, mask_ratio=0.0, finetune_input=False
            )
            _, pred_a, pred_v = self.backbone.forward_decoder(x, ids_restore)
            
        self.backbone.random_masking = original_masking
        self.backbone.attn_mask = original_attn_mask
        
        return pred_a, pred_v.squeeze(-1)  

    def predict(self, states, actions, rewards, advantages, old_log_probs, old_values):
        """
        Takes in: states, actions, rewards, advantages
        shapes of the form [B, T, d] or [B, T + 1, d'] depending on the data.
        internals: prediction, loss calculation, parameter optimization.
        """
        self.backbone.train()
        
        action_mean, values = self._forward_transformer(states, actions, rewards)
        
        action_std = self.actor_logstd.exp().expand_as(action_mean)
        dist = torch.distributions.Normal(action_mean, action_std)
        
        # Sum log probabilities over the action dimensions
        log_probs = dist.log_prob(actions).sum(dim=-1) 
        
        if advantages.dim() == 3:
            advantages = advantages.squeeze(-1)
        
        # Critic Loss: Value function target
        # Since standard returns weren't explicitly passed in the signature, 
        # we construct the target return using the provided advantages: R = A + V_old
        target_values = advantages + old_values

        # Normalize advantages strictly at the mini-batch window step
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
        entropy_loss = dist.entropy().sum(dim=-1).mean()
        
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