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

class MaskDP_BC(nn.Module):
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
        self.PE_type = getattr(cfg, "PE_type", "joint")

        # 1. Init from snapshot
        payload = None
        if path is not None:
            print("Loading existing pre-trained backbone for BC...")
            payload = torch.load(path)
            self.config = payload["cfg"]
        else:
            self.config = cfg
            
        # 2. Incorporate the backbone (2 Modalities: State, Action)
        if self.PE_type == "joint":
            from agent.mdpAA_jointPE import MaskDPJointPE
            self.backbone = MaskDPJointPE(obs_shape, action_shape, self.config).to(self.device)
        else:
            raise NotImplementedError("BC wrapper currently only configured for joint PE.")
        
        print("Number of backbone parameters: %e" % sum(p.numel() for p in self.backbone.parameters()))
        if path is not None:
            self.backbone.load_state_dict(payload["model"])
            print("Payload successfully loaded.")

        T = self.config.traj_length
        self.traj_length = T
        dev = self.device

        diag_template_tl = torch.tensor([
            [1.0, 0.0],
            [1.0, 1.0],
        ], device=dev)
        tl_lower = torch.kron(torch.tril(torch.ones(T, T, device=dev), diagonal=-1), torch.ones(2, 2, device=dev))
        tl_diag = torch.kron(torch.eye(T, device=dev), diag_template_tl)
        tl = tl_lower + tl_diag

        template_tr = torch.tensor([
            [1.0],
            [0.0]
        ], device=dev)
        tr = torch.kron(torch.eye(T, device=dev), template_tr)

        template_bl = torch.tensor([
            [1.0, 0.0]
        ], device=dev)
        bl_lower = torch.kron(torch.tril(torch.ones(T, T, device=dev), diagonal=-1), torch.ones(1, 2, device=dev))
        bl_diag = torch.kron(torch.eye(T, device=dev), template_bl)
        bl = bl_lower + bl_diag
        br = torch.kron(torch.eye(T, device=dev), torch.ones(1, 1, device=dev))

        top_half = torch.cat([tl, tr], dim=1)      # (2T, 3T)
        bottom_half = torch.cat([bl, br], dim=1)   # (T, 3T)
        self.full_pattern_mask = torch.cat([top_half, bottom_half], dim=0).unsqueeze(0).unsqueeze(0) # (1, 1, 3T, 3T)
        self.backbone.attn_mask = self.full_pattern_mask

        # 4. Optimization Setup
        if getattr(cfg, "freeze_backbone", False):
            self.freeze_transformer_core()
            trainable_params = list(self.backbone.action_head.parameters())
        else:
            trainable_params = list(self.parameters())

        self.optimizer = torch.optim.AdamW(trainable_params, lr=self.lr)

    def freeze_transformer_core(self):
        """Freezes core Transformer blocks, leaving the action head open."""
        for param in self.backbone.encoder_blocks.parameters():
            param.requires_grad = False
        for param in self.backbone.decoder_blocks.parameters():
            param.requires_grad = False
        for param in self.backbone.state_embed.parameters():
            param.requires_grad = False
        for param in self.backbone.action_embed.parameters():
            param.requires_grad = False
        print("Transformer core frozen. Optimizing pre-trained action head for BC.")

    def forward_train(self, states, actions):
        self.backbone.train()
        B, T, _ = states.shape

        s_emb = self.backbone.state_embed(states) + self.backbone.mod_embed_s + self.backbone.pos_embed[:, :T, :]
        a_emb = self.backbone.action_embed(actions) + self.backbone.mod_embed_a + self.backbone.pos_embed[:, :T, :]
        
        # 2. Encoder
        x = torch.stack([s_emb, a_emb], dim=2).reshape(B, 2 * T, -1)
        attn_mask_enc = self.backbone.attn_mask[:, :, :2*T, :2*T]
        for blk in self.backbone.encoder_blocks:
            x = blk(x, attn_mask_enc)
        x = self.backbone.encoder_norm(x)
        
        x_s = x[:, 0::2]
        x_a = x[:, 1::2]
        
        projected_s = self.backbone.decoder_state_embed(x_s)
        projected_a = self.backbone.decoder_action_embed(x_a)
        x_dec_gt = torch.stack([projected_s, projected_a], dim=2).reshape(B, 2 * T, -1)
        
        mask_tokens_a = self.backbone.mask_token.view(1, 1, -1).expand(B, T, -1)
        projected_a_mask = self.backbone.decoder_action_embed(mask_tokens_a)
        dec_a_m = projected_a_mask + self.backbone.mod_embed_a + self.backbone.decoder_pos_embed[:, :T, :]
        
        x_unified = torch.cat([x_dec_gt, dec_a_m], dim=1) # (B, 3T, C)
        
        unified_attn_mask = self.backbone.attn_mask[:, :, :3*T, :3*T]
        for blk in self.backbone.decoder_blocks:
            x_unified = blk(x_unified, unified_attn_mask)

        pred_tokens = x_unified[:, 2*T:]
        action_preds = self.backbone.action_head(pred_tokens)
        
        return action_preds

    def act(self, seq_s, seq_a):
        self.backbone.eval()
        with torch.no_grad():
            B, T, _ = seq_s.shape
            Tp = seq_a.shape[1]
            assert T == Tp + 1, "State sequence must be exactly 1 step longer than action sequence."

            # If it's the very first step
            if Tp == 0:
                s_emb = self.backbone.state_embed(seq_s) + self.backbone.mod_embed_s + self.backbone.pos_embed[:, :T, :]
                x = s_emb
                attn_mask = self.backbone.attn_mask[:, :, :T, :T]
                for blk in self.backbone.encoder_blocks:
                    x = blk(x, attn_mask)
                x = self.backbone.encoder_norm(x)

                mask_tokens_a = self.backbone.mask_token.view(1, 1, -1).expand(B, T, -1)
                projected_s = self.backbone.decoder_state_embed(x)
                projected_a = self.backbone.decoder_action_embed(mask_tokens_a)
                dec_a_m = projected_a + self.backbone.mod_embed_a + self.backbone.decoder_pos_embed[:, :T, :]

                base_split = 2 * self.traj_length
                q1 = self.backbone.attn_mask[:, :, :1, :1]
                q2 = self.backbone.attn_mask[:, :, :1, base_split : base_split + T]
                q3 = self.backbone.attn_mask[:, :, base_split : base_split + T, :1]
                q4 = self.backbone.attn_mask[:, :, base_split : base_split + T, base_split : base_split + T]

                unified_attn_mask = torch.cat([
                    torch.cat([q1, q2], dim=-1),
                    torch.cat([q3, q4], dim=-1)
                ], dim=-2)

                x_unified = torch.cat([projected_s, dec_a_m], dim=1)
                for blk in self.backbone.decoder_blocks:
                    x_unified = blk(x_unified, unified_attn_mask)
                
                action_mean = self.backbone.action_head(x_unified[:, 1:])
                return action_mean[:, -1]

            # If Tp > 0
            s_emb = self.backbone.state_embed(seq_s) + self.backbone.mod_embed_s + self.backbone.pos_embed[:, :T, :]
            a_emb = self.backbone.action_embed(seq_a) + self.backbone.mod_embed_a + self.backbone.pos_embed[:, :Tp, :]
            
            x = torch.stack([s_emb[:, :Tp], a_emb], dim=2).reshape(B, 2 * Tp, -1)
            x = torch.cat([x, s_emb[:, Tp:Tp+1]], dim=1)
            
            attn_mask = self.backbone.attn_mask[:, :, :2*Tp+1, :2*Tp+1]
            for blk in self.backbone.encoder_blocks:
                x = blk(x, attn_mask)
            x = self.backbone.encoder_norm(x)

            x_s = x[:, 0::2]
            x_a = x[:, 1::2]

            mask_tokens_a = self.backbone.mask_token.view(1, 1, -1).expand(B, T, -1)
            concat_a = torch.cat([x_a, mask_tokens_a], dim=1) 

            projected_s = self.backbone.decoder_state_embed(x_s)
            projected_a = self.backbone.decoder_action_embed(concat_a)
            
            dec_x_a = projected_a[:, :Tp]
            dec_a_m = projected_a[:, Tp:] + self.backbone.mod_embed_a + self.backbone.decoder_pos_embed[:, :T, :]

            base_split = 2 * self.traj_length
            g_len = 2 * Tp + 1
            m_len = T

            q1 = self.backbone.attn_mask[:, :, :g_len, :g_len]
            q2 = self.backbone.attn_mask[:, :, :g_len, base_split : base_split + m_len]
            q3 = self.backbone.attn_mask[:, :, base_split : base_split + m_len, :g_len]
            q4 = self.backbone.attn_mask[:, :, base_split : base_split + m_len, base_split : base_split + m_len]

            unified_attn_mask = torch.cat([
                torch.cat([q1, q2], dim=-1),
                torch.cat([q3, q4], dim=-1)
            ], dim=-2)

            x_dec_gt = torch.stack([projected_s[:, :Tp], dec_x_a], dim=2).reshape(B, 2 * Tp, -1)
            x_dec_gt_full = torch.cat([x_dec_gt, projected_s[:, Tp:Tp+1]], dim=1)
            
            x_unified = torch.cat([x_dec_gt_full, dec_a_m], dim=1)
            for blk in self.backbone.decoder_blocks:
                x_unified = blk(x_unified, unified_attn_mask)
            
            pred_tokens = x_unified[:, g_len:]
            action_mean = self.backbone.action_head(pred_tokens)
            
            # Return action for the current step (Tp)
            return action_mean[:, -1]

    def update(self, states, actions):
        """
        Executes a single gradient step over a batch of full sequences.
        """
        action_preds = self.forward_train(states, actions)
        
        # Simple MSE loss for continuous actions (Behavioral Cloning)
        loss = F.mse_loss(action_preds, actions)

        self.optimizer.zero_grad()
        loss.backward()
        
        max_grad_norm = getattr(self.config, 'max_grad_norm', 0.5)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()

        return {"bc_loss": loss.item()}