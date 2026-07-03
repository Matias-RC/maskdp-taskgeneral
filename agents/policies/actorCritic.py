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

        # 1. Init from snapshot
        payload = None
        if path is not None:
            print("loading existing model...")
            payload = torch.load(path)
            self.config = payload["cfg"]
        else:
            self.config = cfg
            
        # 2. Incorporate the backbone
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
            print("Payload successfully loaded...")

        T = self.config.traj_length
        self.traj_length = T
        dev = self.device

        # (3T x 3T)
        # Diag template: s sees s (1,0,0); actions/rewards see the whole step block (1,1,1)
        diag_template_tl = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0]
        ], device=dev)
        tl_lower = torch.kron(torch.tril(torch.ones(T, T, device=dev), diagonal=-1), torch.ones(3, 3, device=dev))
        tl_diag = torch.kron(torch.eye(T, device=dev), diag_template_tl)
        tl = tl_lower + tl_diag

        # (3T x 2T)
        # Template: Only state query (row 0) looks at the 2 mask target slots
        template_tr = torch.tensor([
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0]
        ], device=dev)
        tr = torch.kron(torch.eye(T, device=dev), template_tr)

        # (2T x 3T)
        # Template: Mask queries look at s_t (col 0), but ignore a_t and r_t
        template_bl = torch.tensor([
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0]
        ], device=dev)
        bl_lower = torch.kron(torch.tril(torch.ones(T, T, device=dev), diagonal=-1), torch.ones(2, 3, device=dev))
        bl_diag = torch.kron(torch.eye(T, device=dev), template_bl)
        bl = bl_lower + bl_diag

        # (2T x 2T) 
        # Template: Timestep-isolated block-diagonals for mask token pairs
        br = torch.kron(torch.eye(T, device=dev), torch.ones(2, 2, device=dev))

        # assembly 
        top_half = torch.cat([tl, tr], dim=1)      # (3T, 5T)
        bottom_half = torch.cat([bl, br], dim=1)   # (2T, 5T)
        full_pattern_mask = torch.cat([top_half, bottom_half], dim=0) # (5T, 5T)

        self.backbone.attn_mask = full_pattern_mask.unsqueeze(0).unsqueeze(0)

        a_dim = action_shape[0] if isinstance(action_shape, (tuple, list)) else action_shape
        self.actor_logstd = nn.Parameter(torch.zeros(1, a_dim, device=self.device))

        if getattr(cfg, "freeze_backbone", False):
            self.freeze_transformer_core()
            trainable_params = (
                list(self.backbone.action_head.parameters()) + 
                list(self.backbone.reward_head.parameters()) + 
                [self.actor_logstd]
            )
        else:
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
        actions, values = self._forward_transformer(seq_s, seq_a, seq_r)
        
        action_mean = actions[:, -1]
        value_pred = values[:, -1]
        
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
        B, T, _ = states.shape
        assert actions.shape[1] == rewards.shape[1] # Ensure that atleast ground truth is appropiately represented.
        _, Tp, _ = rewards.shape
        assert T == Tp+1, f"Expected T:{T} and Tp:{Tp} to be off by 1. They were off by: {T - Tp}" #Ensrue also correct association I had set or T == Tp but maybe not necesary

        # Now process through the encoder blocks.
        # TODO: Make this part of the class definition for each PE_type, @matias-RC's mistake for training with incorrect assumptions
        assert self.PE_type == "joint"

        # Temporary patch:
        if Tp == 0:
            s_emb = self.backbone.state_embed(states) + self.backbone.mod_embed_s + self.backbone.pos_embed[:, :T, :]
            
            # Encoder pass for just the single state s0
            x = s_emb  # Shape: (B, 1, C)
            attn_mask = self.backbone.attn_mask[:, :, :T, :T]
            for blk in self.backbone.encoder_blocks:
                x = blk(x, attn_mask)
            x = self.backbone.encoder_norm(x)

            # Decoder pass with dummy/masked targets
            mask_tokens_a = self.backbone.mask_token.view(1, 1, -1).expand(B, T, -1)
            mask_tokens_r = self.backbone.mask_token.view(1, 1, -1).expand(B, T, -1)

            projected_s = self.backbone.decoder_state_embed(x)
            projected_a = self.backbone.decoder_action_embed(mask_tokens_a)
            projected_r = self.backbone.decoder_reward_embed(mask_tokens_r)

            dec_a_m = projected_a + self.backbone.mod_embed_a + self.backbone.decoder_pos_embed[:, :T, :]
            dec_r_m = projected_r + self.backbone.mod_embed_r + self.backbone.decoder_pos_embed[:, :T, :]

            masked_tokens = torch.stack([dec_a_m, dec_r_m], dim=2).reshape(B, 2 * T, -1)

            # Mask construction simplifies heavily when Tp = 0
            base_split = 3 * self.backbone.traj_length
            g_len = 1  # 3 * 0 + 1
            m_len = 2 * T

            q1 = self.backbone.attn_mask[:, :, :g_len, :g_len]
            q2 = self.backbone.attn_mask[:, :, :g_len, base_split : base_split + m_len]
            q3 = self.backbone.attn_mask[:, :, base_split : base_split + m_len, :g_len]
            q4 = self.backbone.attn_mask[:, :, base_split : base_split + m_len, base_split : base_split + m_len]

            top_half = torch.cat([q1, q2], dim=-1)
            bottom_half = torch.cat([q3, q4], dim=-1)  # Fixed dim mismatch hazard if shapes differ
            unified_attn_mask = torch.cat([top_half, bottom_half], dim=-2)

            x_unified = torch.cat([projected_s, masked_tokens], dim=1)
            for blk in self.backbone.decoder_blocks:
                x_unified = blk(x_unified, unified_attn_mask)
            
            pred_tokens = x_unified[:, g_len:]

            actions = self.backbone.action_head(pred_tokens[:, 0::2])
            vf = self.backbone.reward_head(pred_tokens[:, 1::2])
            return actions, vf

        s_emb = self.backbone.state_embed(states) + self.backbone.mod_embed_s + self.backbone.pos_embed[:, :T, :]
        a_emb = self.backbone.action_embed(actions) + self.backbone.mod_embed_a + self.backbone.pos_embed[:, :Tp, :]
        r_emb = self.backbone.reward_embed(rewards) + self.backbone.mod_embed_r + self.backbone.pos_embed[:, :Tp, :]
        _, _, C = r_emb.shape
        x = torch.stack(
            (
                s_emb[:, :Tp],  # s0 ... s_{Tp-1}
                a_emb,          # a0 ... a_{Tp-1}
                r_emb,          # r0 ... r_{Tp-1}
            ),
            dim=2,
        )  # (B, Tp, 3, C)

        x = x.reshape(B, 3 * Tp, C)

        # Append final state s_Tp
        x = torch.cat([x, s_emb[:, Tp:Tp+1]], dim=1)
        attn_mask = self.backbone.attn_mask[:, :, :T+2*Tp, :T+2*Tp]

        for blk in self.backbone.encoder_blocks:
            x = blk(x, attn_mask)
        
        x = self.backbone.encoder_norm(x)

        x_s = x[:, 0::3]
        x_a = x[:, 1::3]
        x_r = x[:, 2::3]

        mask_tokens_a = self.backbone.mask_token.view(1, 1, -1).expand(B, T, -1)
        mask_tokens_r = self.backbone.mask_token.view(1, 1, -1).expand(B, T, -1)

        concat_a = torch.cat([x_a, mask_tokens_a], dim=1)  # (B, 2 * T, C)
        concat_r = torch.cat([x_r, mask_tokens_r], dim=1)  # (B, 2 * T, C)

        projected_s = self.backbone.decoder_state_embed(x_s)
        projected_a = self.backbone.decoder_action_embed(concat_a)  # Shape: (B, 2 * Tp, C_out)
        projected_r = self.backbone.decoder_reward_embed(concat_r)

        dec_x_a = projected_a[:, :Tp]
        dec_a_m = projected_a[:, Tp:] + self.backbone.mod_embed_a + self.backbone.decoder_pos_embed[:, :T, :]

        dec_x_r = projected_r[:, :Tp]
        dec_r_m = projected_r[:, Tp:] + self.backbone.mod_embed_r + self.backbone.decoder_pos_embed[:, :T, :]

        masked_tokens = torch.stack([dec_a_m, dec_r_m], dim=2).reshape(B, 2 * T, -1)

        # Construct mask: self.traj_length is base
        base_split = 3 * self.backbone.traj_length
        g_len = 3 * Tp + 1
        m_len = 2 * T

        q1 = self.backbone.attn_mask[:, :, :g_len, :g_len]
        q2 = self.backbone.attn_mask[:, :, :g_len, base_split : base_split + m_len]
        q3 = self.backbone.attn_mask[:, :, base_split : base_split + m_len, :g_len]
        q4 = self.backbone.attn_mask[:, :, base_split : base_split + m_len, base_split : base_split + m_len]

        top_half = torch.cat([q1, q2], dim=-1)
        bottom_half = torch.cat([q3, q4], dim=-1)
        unified_attn_mask = torch.cat([top_half, bottom_half], dim=-2)
        x_dec_gt = torch.stack(
            (
                projected_s[:, :Tp],
                dec_x_a,            
                dec_x_r             
            ),
            dim=2,
        ).reshape(B, 3 * Tp, -1)

        x_dec_gt_full = torch.cat([x_dec_gt, projected_s[:, Tp:Tp+1]], dim=1)
        x_unified = torch.cat([x_dec_gt_full, masked_tokens], dim=1)
        for blk in self.backbone.decoder_blocks:
            x_unified = blk(x_unified, unified_attn_mask)
        
        pred_tokens = x_unified[:, g_len:]

        actions = self.backbone.action_head(pred_tokens[:, 0::2])
        vf = self.backbone.reward_head(pred_tokens[:, 1::2])
        return actions, vf

    def predict(self, states, actions, rewards, advantages, old_log_probs, old_values):
        """
        states: T
        actions: Tp
        rewards: Tp
        advantages: Tp
        old_log_probs: Tp
        old_values: Tp
        where T and Tp are lengths
        """
        self.backbone.train()
        
        action_mean, values = self._forward_transformer(states, actions, rewards)
        action_mean = action_mean[:, :-1]
        values = values[:, :-1].squeeze(-1)
        
        assert actions.shape[:2] == values.shape[:2]
        assert advantages.shape == old_values.shape
        assert old_log_probs.shape == old_values.shape

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