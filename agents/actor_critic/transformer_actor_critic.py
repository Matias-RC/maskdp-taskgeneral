"""
Causal triplet Transformer actor-critic.

Token layout per timestep t follows envs/mujoco_gym_context.ContextWindowWrapper:

    triplet t = (s_t, a_{t-1}, r_{t-1})

i.e. the observation at t plus the action/reward that led to it (zeros after a
reset). No triplet contains a_t, so the "next action" query of timestep t never
sees its own target, and all three tokens of a triplet are available at decision
time. That is what makes it safe to attend *within* a patch of three.

Padding follows the same wrapper: left-padded windows, newest step at index -1,
mask = 1 for real steps and 0 for padding.
"""
from collections.abc import Mapping
from types import SimpleNamespace

import numpy as np
import torch as th
import torch.nn as nn

from agents.modules.attention import Block, CrossBlock
from agents.actor_critic.actor_critic import Actor, Critic


def cfg_get(cfg, key, default=None):
    """Works for dict, OmegaConf DictConfig and SimpleNamespace/argparse configs."""
    if isinstance(cfg, Mapping) or hasattr(cfg, "get"):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def sincos_pos_embed(dim, length):
    omega = 1.0 / 10000 ** (np.arange(dim // 2, dtype=np.float64) / (dim / 2.0))
    out = np.einsum("m,d->md", np.arange(length, dtype=np.float64), omega)
    return np.concatenate([np.sin(out), np.cos(out)], axis=1)


class CausalTripletTransformerAC(nn.Module):
    """
    config keys
        n_embd, n_head,
        context_len (16)   max number of triplets L in a window
        n_layer (2)        encoder Blocks
        n_dec_layer (1)    decoder CrossBlocks
        attn_pdrop (0.0), resid_pdrop (0.0)
            Keep dropout at 0 for PPO: with dropout on, the log-probs recomputed
            during the update no longer match the ones stored at rollout time.

    Shapes (L = context length, D = n_embd)
        states  : [B, L, obs_dim]
        actions : [B, L, act_dim]   continuous (Box), no one-hot / NO_OP
        rewards : [B, L] or [B, L, 1]
        mask    : [B, L]            1/True = real step, 0/False = padding
    """

    N_TOK = 3                        # tokens per timestep (patch size)
    TOK_S, TOK_A, TOK_R = 0, 1, 2    # order inside a patch
    Q_ACT, Q_VAL = 0, 1              # decoder questions

    def __init__(self, obs_dim, act_dim, config):
        super().__init__()
        D = self.n_embd = cfg_get(config, "n_embd")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = cfg_get(config, "context_len", 16)
        self.n_tokens = self.N_TOK * self.context_len
        n_layer = cfg_get(config, "n_layer", 2)
        n_dec_layer = cfg_get(config, "n_dec_layer", 1)
        blk_cfg = SimpleNamespace(
            n_embd=D,
            n_head=cfg_get(config, "n_head"),
            attn_pdrop=cfg_get(config, "attn_pdrop", 0.0),
            resid_pdrop=cfg_get(config, "resid_pdrop", 0.0),
        )

        # ---- context embedding (x) ----
        self.state_embed = nn.Linear(obs_dim, D)
        self.action_embed = nn.Linear(act_dim, D)      # Box actions: linear, not Embedding
        self.reward_embed = nn.Linear(1, D)
        self.modality_embed = nn.Parameter(th.zeros(self.N_TOK, D))

        # ---- encoder: one pass of block-causal self-attention ----
        self.blocks = nn.ModuleList([Block(blk_cfg) for _ in range(n_layer)])
        self.norm = nn.LayerNorm(D)

        # ---- decoder: per-timestep questions answered by cross-attention ----
        self.query_embed = nn.Linear(obs_dim, D)       # "board_embed" in the TRM reference
        self.query_mlp = nn.Sequential(
            nn.LayerNorm(D), nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D)
        )
        self.question_embed = nn.Parameter(th.zeros(2, D))  # [Q_ACT, Q_VAL]
        self.decoder_blocks = nn.ModuleList([CrossBlock(blk_cfg) for _ in range(n_dec_layer)])
        self.decoder_norm = nn.LayerNorm(D)

        # ---- heads (same modules as the MLP baseline, input is now D) ----
        self.actor = Actor(D, act_dim)     # -> mean, std ; log_std is a free parameter
        self.critic = Critic(D)            # -> [.., 1]

        # ---- fixed buffers (non-persistent: rebuilt from config on load) ----
        pe = th.from_numpy(sincos_pos_embed(D, self.n_tokens)).float().unsqueeze(0) / 2.0
        self.register_buffer("pos_embed", pe, persistent=False)            # [1, 3L, D]
        patch = th.arange(self.n_tokens) // self.N_TOK
        # patch_causal[i, j] = token i may attend token j  <=>  patch(j) <= patch(i)
        self.register_buffer(
            "patch_causal", patch.unsqueeze(1) >= patch.unsqueeze(0), persistent=False
        )                                                                    # [3L, 3L]

        self._init_weights()

    def _init_weights(self):
        # Actor/Critic keep their orthogonal init (mean head std=0.01, value head std=1.0)
        skip = {id(m) for head in (self.actor, self.critic) for m in head.modules()}
        for m in self.modules():
            if id(m) in skip:
                continue
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        for p in (self.modality_embed, self.question_embed):
            nn.init.normal_(p, std=0.02)

    # ------------------------------------------------------------ inputs
    def _check_inputs(self, states, actions, rewards):
        B, L, _ = states.shape
        assert L <= self.context_len, f"window {L} > context_len {self.context_len}"
        assert actions.shape[:2] == (B, L) and actions.shape[-1] == self.act_dim
        assert rewards.shape[:2] == (B, L)
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)
        return rewards

    @staticmethod
    def _bool_mask(mask, states):
        B, L = states.shape[:2]
        if mask is None:
            return th.ones(B, L, dtype=th.bool, device=states.device)
        return mask.to(states.device) > 0.5 if mask.dtype != th.bool else mask.to(states.device)

    # ------------------------------------------------------------- masks
    def build_attn_mask(self, mask):
        """
        mask [B, L] bool -> attn_mask [B, 1, 3L, 3L].

        Block-causal over patches of 3 (a token sees its own triplet and all earlier
        ones) and padded keys removed. The diagonal is always kept so that rows of
        padded tokens never become all -inf (softmax NaN would leak into the valid
        rows through att @ v even with zero weight).
        """
        B, L = mask.shape
        n = self.N_TOK * L
        key_valid = mask.repeat_interleave(self.N_TOK, dim=1)                  # [B, 3L]
        allowed = self.patch_causal[:n, :n].unsqueeze(0) & key_valid[:, None, :]
        allowed = allowed | th.eye(n, dtype=th.bool, device=mask.device)
        return allowed.unsqueeze(1).to(self.pos_embed.dtype)

    # ----------------------------------------------------------- encoder
    def embed_context(self, states, actions, rewards):
        """-> x [B, 3L, D], interleaved s_0, a_0, r_0, s_1, a_1, r_1, ..."""
        B, L, _ = states.shape
        s = self.state_embed(states) + self.modality_embed[self.TOK_S]
        a = self.action_embed(actions) + self.modality_embed[self.TOK_A]
        r = self.reward_embed(rewards) + self.modality_embed[self.TOK_R]
        x = th.stack([s, a, r], dim=2).reshape(B, self.N_TOK * L, self.n_embd)
        return x + self.pos_embed[:, : self.N_TOK * L]

    def encode(self, states, actions, rewards, mask=None):
        """-> y [B, 3L, D]"""
        rewards = self._check_inputs(states, actions, rewards)
        mask = self._bool_mask(mask, states)
        x = self.embed_context(states, actions, rewards)
        attn_mask = self.build_attn_mask(mask)
        for blk in self.blocks:
            x = blk(x, attn_mask)
        return self.norm(x)

    # ----------------------------------------------------------- decoder
    def decode(self, y, states, mask=None):
        """
        y      : [B, 3L, D]  encoder output
        states : [B, L, obs_dim]  raw observations, one query per timestep
        mask   : [B, L]

        Each timestep's queries cross-attend only to that timestep's triplet:
            y       -> [B*L, 3, D]
            queries -> [B*L, D]  per question, stacked as [B*L, 2, D]
        Stacking the two questions is equivalent to two separate passes because
        CrossBlock has no attention between queries.

        -> mean [B, L, act_dim], std [B, L, act_dim], value [B, L]
        """
        B, L = states.shape[:2]
        D = self.n_embd
        assert y.shape == (B, self.N_TOK * L, D), f"y {tuple(y.shape)} vs states L={L}"

        ctx = y.reshape(B * L, self.N_TOK, D)                                   # [B*L, 3, D]
        q = self.query_mlp(self.query_embed(states)).reshape(B * L, 1, D)       # [B*L, 1, D]
        q = q + self.question_embed.unsqueeze(0)                                # [B*L, 2, D]
        for blk in self.decoder_blocks:
            q = blk(q, ctx)
        h = self.decoder_norm(q).view(B, L, 2, D)

        if mask is not None:   # padded timesteps -> deterministic, finite outputs
            h = h.masked_fill(~self._bool_mask(mask, states)[..., None, None], 0.0)

        mean, std = self.actor(h[:, :, self.Q_ACT])
        value = self.critic(h[:, :, self.Q_VAL]).squeeze(-1)
        return mean, std, value

    # ----------------------------------------------------------- forward
    def forward(self, states, actions, rewards, mask=None, last_only=False):
        """
        last_only=True decodes only the newest timestep (index -1), which is what a
        rollout needs; outputs then have L=1. The encoder always sees the full window.
        """
        mask = self._bool_mask(mask, states)
        y = self.encode(states, actions, rewards, mask)
        if last_only:
            y, states, mask = y[:, -self.N_TOK:], states[:, -1:], mask[:, -1:]
        return self.decode(y, states, mask)