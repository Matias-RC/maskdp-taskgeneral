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

import random

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

    @torch.no_grad()
    def act(self, seq_s, seq_a):
        self.backbone.eval()

        x, _, ids_restore, num_mods = self.backbone.forward_encoder(
            states=seq_s,
            actions=seq_a,
            rewards=None,
            mask_ratio=0.0,
            finetune_input=True,
            do_last_n=1,
        )

        _, actions, _ = self.backbone.forward_decoder(
            x,
            ids_restore,
            num_mods,
        )

        return actions[:, -1, :]

    def update(self, seq_s, seq_a):
        self.backbone.train()
        #seq_s has shape : [256, 64, 24]
        T = seq_s.shape[1]

        # Phase 1
        # Git gud (At natural play pred -> a_T=64) 
        x, mask, ids_restore, num_mods = self.backbone.forward_encoder(states=seq_s, actions=seq_a, rewards=None, mask_ratio=0.0, finetune_input=True, do_last_n=1)
        states, actions, _ = self.backbone.forward_decoder(x, ids_restore, num_mods)
        a_last = actions[:,-1,:]
        a_targ = seq_a[:, -1, :]
                
        # Phase 2

        l = random.randint(1, T-2)
        x, mask, ids_restore, num_mods = self.backbone.forward_encoder(states=seq_s[:, :l, :], actions=seq_a[:, :l, :], rewards=None, mask_ratio=0.0, finetune_input=True, do_last_n=1)
        states, actions, _ = self.backbone.forward_decoder(x, ids_restore, num_mods)
        a_last_alt = actions[:,-1,:]
        a_targ_alt = seq_a[:, l-1, :]

        loss = F.mse_loss(a_last, a_targ) + F.mse_loss(a_last_alt, a_targ_alt)
        loss.backward()
        
        max_grad_norm = getattr(self.config, 'max_grad_norm', 0.5)
        torch.nn.utils.clip_grad_norm_(self.parameters(), max_grad_norm)
        self.optimizer.step()

        return {"bc_loss": loss.item()}           
