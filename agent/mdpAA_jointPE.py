"""
You define positional encoding for time-steps only. You define a separate modality encoding to distinguish between states, actions, 
and rewards. During pre-training, you only use the encodings for states and actions. During fine-tuning, you use everything. The m-
ain advantage of this is that it’s easier to add on extra modalities in the fine-tuning stage if we wanted more modalities without 
needing to run pretraining all over again. I personally prefer this factorised encoding more, because we cleanly separate modality 
and time encoding this way, whereas it’s all mixed together in (1)
"""

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Optional

import utils
from dm_control.utils import rewards
from einops import rearrange, reduce, repeat
from agent.modules.attention import Block, CausalSelfAttention


class MaskDPJointPE(nn.Module):
    def __init__(self, obs_dim, action_dim, config):
        super().__init__()
        # MAE encoder specifics
        self.n_embd = config.n_embd
        
        # Positional encoding is now strictly bound to time steps, not total sequence length
        self.traj_length = config.traj_length
        self.max_time_steps = config.traj_length 

        self.pe = config.pe
        self.norm = config.norm
        print("norm", self.norm)
        print("Action dim:", action_dim)
        # Modality Encoders
        self.state_embed = nn.Linear(obs_dim, self.n_embd)
        self.action_embed = nn.Linear(action_dim, self.n_embd)
        self.reward_embed = nn.Linear(1, self.n_embd)
        
        # Learnable Modality Embeddings
        self.mod_embed_s = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.mod_embed_a = nn.Parameter(torch.zeros(1, 1, self.n_embd))
        self.mod_embed_r = nn.Parameter(torch.zeros(1, 1, self.n_embd))

        self.encoder_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_enc_layer)]
        )
        self.encoder_norm = nn.LayerNorm(self.n_embd)
        
        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_state_embed = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_action_embed = nn.Linear(self.n_embd, self.n_embd)
        self.decoder_reward_embed = nn.Linear(self.n_embd, self.n_embd)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, self.n_embd))

        self.decoder_blocks = nn.ModuleList(
            [Block(config) for _ in range(config.n_dec_layer)]
        )

        self.action_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, action_dim),
            nn.Tanh(),
        )  # decoder to patch
        
        self.state_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, obs_dim),
        )

        self.reward_head = nn.Sequential(
            nn.LayerNorm(self.n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(self.n_embd, 1),
        )

        # --------------------------------------------------------------------------
        self.initialize_weights()

    def initialize_weights(self):
        # Time-based positional encoding (1D over the number of actual timesteps)
        pos_embed = utils.get_1d_sincos_pos_embed_from_grid(self.n_embd, self.max_time_steps)
        pe = torch.from_numpy(pos_embed).float().unsqueeze(0) / 2.0
        self.register_buffer("pos_embed", pe)
        self.register_buffer("decoder_pos_embed", pe)
        
        # Attention mask must support the max theoretical flattened length (3 modalities)
        max_seq_len = self.max_time_steps * 3
        self.register_buffer(
            "attn_mask", torch.ones(max_seq_len, max_seq_len)[None, None, ...]
        )
        
        # Initialize tokens and embeddings
        torch.nn.init.normal_(self.mask_token, std=0.02)
        torch.nn.init.normal_(self.mod_embed_s, std=0.02)
        torch.nn.init.normal_(self.mod_embed_a, std=0.02)
        torch.nn.init.normal_(self.mod_embed_r, std=0.02)
        
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))
        
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def special_masking(self, x, last_n=2):
        N, L, D = x.shape

        assert L > last_n

        x_masked = x[:, :-last_n]

        mask = torch.ones((N, L), device=x.device)
        mask[:, :-2] = 0

        ids_restore = torch.arange(L, device=x.device).expand(N, -1)

        return x_masked, mask, ids_restore

    
    def forward_encoder(
            self, 
            states, 
            actions, 
            rewards: Optional[torch.Tensor], 
            mask_ratio, 
            finetune_input=False,
            do_last_n=2
        ):
        batch_size, T, obs_dim = states.size()
        
        # Add modality and timestep embeddings directly to features
        s_emb = self.state_embed(states) + self.mod_embed_s + self.pos_embed[:, :T, :]
        a_emb = self.action_embed(actions) + self.mod_embed_a + self.pos_embed[:, :T, :]
        
        # Dynamically compose sequence based on modalities provided (pre-training vs finetuning)
        if rewards is not None:
            r_emb = self.reward_embed(rewards) + self.mod_embed_r + self.pos_embed[:, :T, :]
            # Stack dim=2 creates sequence: s0, a0, r0, s1, a1, r1...
            x = torch.stack([s_emb, a_emb, r_emb], dim=2).reshape(batch_size, 3 * T, self.n_embd)
            num_mods = 3
        else:
            x = torch.stack([s_emb, a_emb], dim=2).reshape(batch_size, 2 * T, self.n_embd)
            num_mods = 2

        if not finetune_input:
            x, mask, ids_restore = self.random_masking(x, mask_ratio)
        else:
            x, mask, ids_restore = self.special_masking(x, last_n=do_last_n)
        # Slice attention mask to match current flattened length
        curr_len = x.shape[1]
        attn_mask = self.attn_mask[:, :, :curr_len, :curr_len]

        for blk in self.encoder_blocks:
            x = blk(x, attn_mask)
        x = self.encoder_norm(x)
        
        # Return num_mods to dynamically inform the decoder
        return x, mask, ids_restore, num_mods
    
    def forward_decoder(self, x, ids_restore, num_mods):
        # Append mask tokens
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] - x.shape[1], 1
        )
        x_ = torch.cat([x, mask_tokens], dim=1)
        x = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2])
        )
        
        T = x.shape[1] // num_mods

        # Re-apply modality and timestep embeddings for the un-masked sequence (since mask tokens lack them)
        if num_mods == 3:
            s = self.decoder_state_embed(x[:, 0::3]) + self.mod_embed_s + self.decoder_pos_embed[:, :T, :]
            a = self.decoder_action_embed(x[:, 1::3]) + self.mod_embed_a + self.decoder_pos_embed[:, :T, :]
            r = self.decoder_reward_embed(x[:, 2::3]) + self.mod_embed_r + self.decoder_pos_embed[:, :T, :]
            x = torch.stack([s, a, r], dim=2).reshape_as(x)
        else:
            s = self.decoder_state_embed(x[:, 0::2]) + self.mod_embed_s + self.decoder_pos_embed[:, :T, :]
            a = self.decoder_action_embed(x[:, 1::2]) + self.mod_embed_a + self.decoder_pos_embed[:, :T, :]
            x = torch.stack([s, a], dim=2).reshape_as(x)

        curr_len = x.shape[1]
        attn_mask = self.attn_mask[:, :, :curr_len, :curr_len]

        for blk in self.decoder_blocks:
            x = blk(x, attn_mask)

        # Projection heads
        s = self.state_head(x[:, 0::num_mods])
        a = self.action_head(x[:, 1::num_mods])
        
        if num_mods == 3:
            r = self.reward_head(x[:, 2::num_mods])
            return s, a, r
        else:
            return s, a, None
    
    def apply_masking(self, x):
        T = x.shape[1] // 3

        # Slice modalities out using strides of 3
        s = self.decoder_state_embed(x[:, 0::3]) + self.mod_embed_s + self.decoder_pos_embed[:, :T, :]
        a = self.decoder_action_embed(x[:, 1::3]) + self.mod_embed_a + self.decoder_pos_embed[:, :T, :]
        r = self.decoder_reward_embed(x[:, 2::3]) + self.mod_embed_r + self.decoder_pos_embed[:, :T, :]
        
        # Interleave them back together cleanly into [B, 3 * T, n_embd]
        x = torch.stack([s, a, r], dim=2).reshape_as(x)
        return x

    def forward_loss(self, target_s, target_a, target_r, pred_s, pred_a, pred_r, mask):
        batch_size, T, _ = target_s.size()
        
        if self.norm == "l2":
            target_s = target_s / (torch.norm(target_s, dim=-1, keepdim=True) + 1e-8)
            if target_r is not None:
                target_r = target_r / (torch.norm(target_r, dim=-1, keepdim=True) + 1e-8)
        elif self.norm == "mae":
            mean_s = target_s.mean(dim=-1, keepdim=True)
            var_s = target_s.var(dim=-1, keepdim=True)
            target_s = (target_s - mean_s) / (var_s + 1.0e-6) ** 0.5
            if target_r is not None:
                mean_r = target_r.mean(dim=-1, keepdim=True)
                var_r = target_r.var(dim=-1, keepdim=True)
                target_r = (target_r - mean_r) / (var_r + 1.0e-6) ** 0.5

        loss_s = (pred_s - target_s) ** 2
        loss_a = (pred_a - target_a) ** 2
        
        if target_r is not None and pred_r is not None:
            loss_r = (pred_r - target_r) ** 2
            loss = (
                torch.stack([loss_s.mean(dim=-1), loss_a.mean(dim=-1), loss_r.mean(dim=-1)], dim=2)
                .reshape(batch_size, 3 * T)
            )
            loss_r_mean = loss_r.mean()
        else:
            loss = (
                torch.stack([loss_s.mean(dim=-1), loss_a.mean(dim=-1)], dim=2)
                .reshape(batch_size, 2 * T)
            )
            loss_r_mean = torch.tensor(0.0, device=target_s.device)

        masked_loss = (loss * mask).sum() / (mask.sum() + 1e-8)
        loss_s_mean = loss_s.mean()
        loss_a_mean = loss_a.mean()
        
        return masked_loss, loss_s_mean, loss_a_mean, loss_r_mean


class JointDPAgent:
    def __init__(
        self,
        name,
        obs_shape,
        action_shape,
        device,
        lr,
        batch_size,
        use_tb,
        mask_ratio,
        transformer_cfg,
    ):
        self.action_dim = action_shape[0]
        self.lr = lr
        self.device = device
        self.use_tb = use_tb
        self.config = transformer_cfg

        # models
        self.model = MaskDPJointPE(obs_shape[0], action_shape[0], transformer_cfg).to(device)
        self.mask_ratio = mask_ratio
        
        # optimizers
        self.opt = torch.optim.Adam(self.model.parameters(), lr=lr)
        print(
            "number of parameters: %e", sum(p.numel() for p in self.model.parameters())
        )

        self.train()

    def train(self, training=True):
        self.training = training
        self.model.train(training)

    def update_mdp(self, states, actions, rewards: Optional[torch.Tensor] = None):
        # Now handles optional rewards without enforcing zeros natively.
        metrics = dict()
        mask_ratio = np.random.choice(self.mask_ratio)
        
        latent, mask, ids_restore, num_mods = self.model.forward_encoder(
            states, actions, rewards, mask_ratio
        )
        pred_s, pred_a, pred_r = self.model.forward_decoder(
            latent, ids_restore, num_mods
        ) 
        mask_loss, state_loss, action_loss, reward_loss = self.model.forward_loss(
            states, actions, rewards, pred_s, pred_a, pred_r, mask
        )
        
        if self.config.loss == "masked":
            loss = mask_loss
        elif self.config.loss == "total":
            loss = state_loss + action_loss
            if rewards is not None:
                loss += reward_loss
        else:
            raise NotImplementedError

        self.opt.zero_grad(set_to_none=True)
        loss.backward()
        self.opt.step()

        metrics["mask_loss"] = mask_loss.item()
        metrics["state_loss"] = state_loss.item()
        metrics["action_loss"] = action_loss.item()
        
        if rewards is not None:
            metrics["reward_loss"] = reward_loss.item()

        return metrics

    def eval_validation(self, val_iter, step=None, use_rewards=False):
        metrics = dict()
        batch = next(val_iter)
        obs, action, reward, _, _, _ = utils.to_torch(batch, self.device)

        rewards_to_pass = reward if use_rewards else None

        mask_ratio = np.random.choice(self.mask_ratio)
        latent, mask, ids_restore, num_mods = self.model.forward_encoder(
            obs, action, rewards_to_pass, mask_ratio
        )
        pred_s, pred_a, pred_r = self.model.forward_decoder(
            latent, ids_restore, num_mods
        )
        mask_loss, state_loss, action_loss, reward_loss = self.model.forward_loss(
            obs, action, rewards_to_pass, pred_s, pred_a, pred_r, mask
        )

        metrics["val_mask_loss"] = mask_loss.item()
        metrics["val_state_loss"] = state_loss.item()
        metrics["val_action_loss"] = action_loss.item()
        
        if rewards_to_pass is not None:
            metrics["val_reward_loss"] = reward_loss.item()

        return metrics

    def update(self, replay_iter, step=None, use_rewards=False):
        metrics = dict()
        batch = next(replay_iter)
        obs, action, reward, _, _, _ = utils.to_torch(batch, self.device)
        
        rewards_to_pass = reward if use_rewards else None
        
        metrics.update(self.update_mdp(obs, action, rewards_to_pass))
        return metrics