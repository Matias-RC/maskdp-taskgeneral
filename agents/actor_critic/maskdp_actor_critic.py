"""
MaskDP (agents/maskdp/rewards_joint_PE.MaskDPJointPE) as a PPO actor-critic.

Same I/O contract as agents/actor_critic/transformer_actor_critic.CausalTripletTransformerAC,
so TransformerPPOTrainer can use either:

    forward(states [B,L,obs], actions [B,L,act], rewards [B,L(,1)], mask [B,L], last_only=True)
        -> mean [B,1,act], std [B,1,act], value [B,1]

Token layout follows envs/mujoco_gym_context.ContextWindowWrapper (and the pretraining data,
which was aligned with data/env_transform_loader.EnvTransformLoader(align_to_context=True)):

    triplet t = (s_t, a_{t-1}, r_{t-1})

Architecture (pretrained weights load 1:1 into every module except the new ones):

    encoder  bidirectional Blocks over the full window, no random masking; padded keys removed.
    decoder  bidirectional Blocks over [decoder-embedded window, next-action query].
             The query is the MAE mask token re-embedded as an action token at timestep L,
             i.e. the action slot of triplet L, which holds a_{L-1} = the next action.
             The window positions 0..L-1 use the pretrained sincos table; position L is the
             same sincos formula evaluated one step further (the table is length-independent).
    heads    actor: the pretrained action_head gives the mean, plus a free log_std (new).
             critic: LayerNorm + Critic MLP (new), on the same query latent as the actor,
             mirroring decoder_norm -> Critic in CausalTripletTransformerAC.
    dropped  state_head, reward_head, mod_embed_p (unused when nothing is masked).

fresh_heads=True (optional) replaces both heads with freshly initialised baseline MLPs
(agents/actor_critic/actor_critic.Actor / Critic) whose input is

    [LayerNorm(query latent), s_t]

i.e. the decoder output concatenated with the current normalized state, which is exactly
what the baseline actor-critic sees (s_t = states[:, -1], the newest triplet of the
left-padded window). The pretrained action_head is not loaded in this mode.

freeze_backbone=True (optional) freezes the encoder/decoder trunk (every parameter outside
actor/critic): requires_grad=False and the trunk stays in eval mode, so PPO only trains the
heads on top of the pretrained representation. Combines with either head mode.
"""
from types import SimpleNamespace

import torch as th
import torch.nn as nn

from agents.actor_critic.actor_critic import Actor, Critic
from agents.actor_critic.transformer_actor_critic import cfg_get
from agents.modules.attention import Block
from utils import utils


class MaskDPActor(nn.Module):
    """Pretrained MaskDP action head as the Gaussian mean; log_std is a free parameter."""

    def __init__(self, n_embd, act_dim):
        super().__init__()
        self.mean_head = nn.Sequential(
            nn.LayerNorm(n_embd),
            nn.ReLU(inplace=True),
            nn.Linear(n_embd, act_dim),
            nn.Tanh(),
        )
        self.log_std = nn.Parameter(th.zeros(act_dim))

    def forward(self, x, state=None):
        mean = self.mean_head(x)
        return mean, th.exp(self.log_std).expand_as(mean)


class MaskDPFreshActor(nn.Module):
    """New baseline Actor on [LayerNorm(latent), s_t]."""

    def __init__(self, n_embd, obs_dim, act_dim):
        super().__init__()
        self.norm = nn.LayerNorm(n_embd)
        self.policy = Actor(n_embd + obs_dim, act_dim)

    def forward(self, x, state):
        return self.policy(th.cat([self.norm(x), state], dim=-1))


class MaskDPCritic(nn.Module):
    """Critic MLP on LayerNorm(latent), with s_t concatenated when state_dim > 0."""

    def __init__(self, n_embd, state_dim=0):
        super().__init__()
        self.norm = nn.LayerNorm(n_embd)
        self.value = Critic(n_embd + state_dim)

    def forward(self, x, state=None):
        h = self.norm(x)
        if state is not None:
            h = th.cat([h, state], dim=-1)
        return self.value(h)


class MaskDPActorCritic(nn.Module):
    """
    config keys
        n_embd, n_head, n_enc_layer (1), n_dec_layer (1),
        context_len (16)   max number of triplets L in a window
        attn_pdrop (0.0), resid_pdrop (0.0)   keep 0 for PPO
        fresh_heads (False)  new actor/critic MLPs on [decoder latent, current state]
        freeze_backbone (False)  train only actor/critic; the encoder/decoder trunk is frozen
    """

    N_TOK = 3

    # pretrained key prefix -> adapter key prefix; None = not used by the adapter
    PRETRAINED_REMAP = {
        "action_head.": "actor.mean_head.",
        "state_head.": None,
        "reward_head.": None,
        "mod_embed_p": None,
        # fixed buffers, rebuilt here (pos_embed is checked against the snapshot in load_pretrained)
        "pos_embed": None,
        "decoder_pos_embed": None,
        "attn_mask": None,
    }

    def __init__(self, obs_dim, act_dim, config):
        super().__init__()
        D = self.n_embd = cfg_get(config, "n_embd")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.context_len = cfg_get(config, "context_len", 16)
        blk_cfg = SimpleNamespace(
            n_embd=D,
            n_head=cfg_get(config, "n_head"),
            attn_pdrop=cfg_get(config, "attn_pdrop", 0.0),
            resid_pdrop=cfg_get(config, "resid_pdrop", 0.0),
        )

        # ---- encoder (names match MaskDPJointPE) ----
        self.state_embed = nn.Linear(obs_dim, D)
        self.action_embed = nn.Linear(act_dim, D)
        self.reward_embed = nn.Linear(1, D)
        self.mod_embed_s = nn.Parameter(th.zeros(1, 1, D))
        self.mod_embed_a = nn.Parameter(th.zeros(1, 1, D))
        self.mod_embed_r = nn.Parameter(th.zeros(1, 1, D))
        self.encoder_blocks = nn.ModuleList([Block(blk_cfg) for _ in range(cfg_get(config, "n_enc_layer", 1))])
        self.encoder_norm = nn.LayerNorm(D)

        # ---- decoder (names match MaskDPJointPE) ----
        self.decoder_state_embed = nn.Linear(D, D)
        self.decoder_action_embed = nn.Linear(D, D)
        self.decoder_reward_embed = nn.Linear(D, D)
        self.mask_token = nn.Parameter(th.zeros(1, 1, D))
        self.decoder_blocks = nn.ModuleList([Block(blk_cfg) for _ in range(cfg_get(config, "n_dec_layer", 1))])

        # ---- heads ----
        self.fresh_heads = bool(cfg_get(config, "fresh_heads", False))
        if self.fresh_heads:
            self.actor = MaskDPFreshActor(D, obs_dim, act_dim)
            self.critic = MaskDPCritic(D, state_dim=obs_dim)
        else:
            self.actor = MaskDPActor(D, act_dim)
            self.critic = MaskDPCritic(D)

        # Timesteps 0..L-1 for the window, L for the next-action query. Same scaling as MaskDPJointPE.
        pe = th.from_numpy(utils.get_1d_sincos_pos_embed_from_grid(D, self.context_len + 1)).float()
        self.register_buffer("pos_embed", pe.unsqueeze(0) / 2.0, persistent=False)  # [1, L+1, D]

        self._init_weights()

        self.freeze_backbone = bool(cfg_get(config, "freeze_backbone", False))
        if self.freeze_backbone:
            for p in self.backbone_parameters():
                p.requires_grad_(False)

    def backbone_parameters(self):
        """Encoder/decoder trunk = every parameter outside the actor and critic heads."""
        head_ids = {id(p) for p in self.actor.parameters()} | {id(p) for p in self.critic.parameters()}
        return [p for p in self.parameters() if id(p) not in head_ids]

    def train(self, mode=True):
        """A frozen trunk stays in eval mode (no dropout) whatever the trainer asks."""
        super().train(mode)
        if self.freeze_backbone:
            for name, child in self.named_children():
                if name not in ("actor", "critic"):
                    child.eval()
        return self

    def _init_weights(self):
        # Same init as MaskDPJointPE; the baseline Actor/Critic MLPs keep their orthogonal init.
        skip = {id(m) for mod in self.modules() if isinstance(mod, (Actor, Critic)) for m in mod.modules()}
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
        for p in (self.mask_token, self.mod_embed_s, self.mod_embed_a, self.mod_embed_r):
            nn.init.normal_(p, std=0.02)

    # ------------------------------------------------------------ pretrained weights
    @staticmethod
    def arch_from_snapshot(path):
        """Model config keys stored in an offline_training.py snapshot."""
        cfg = th.load(path, map_location="cpu", weights_only=False)["cfg"]
        return {k: cfg_get(cfg, k) for k in ("n_embd", "n_head", "n_enc_layer", "n_dec_layer", "traj_length")}

    def load_pretrained(self, path, verbose=True):
        """Load a MaskDPJointPE snapshot. Returns (loaded keys, newly initialised keys)."""
        payload = th.load(path, map_location="cpu", weights_only=False)
        src = payload["model"]
        traj_length = cfg_get(payload["cfg"], "traj_length")
        assert self.context_len <= traj_length, \
            f"context_len {self.context_len} > pretraining traj_length {traj_length}"
        # The snapshot's table must equal ours on the window positions.
        assert th.allclose(src["pos_embed"][:, : self.context_len], self.pos_embed[:, : self.context_len].cpu(), atol=1e-6), \
            "pretrained pos_embed differs from the adapter's sincos table"

        remap = dict(self.PRETRAINED_REMAP)
        if self.fresh_heads:
            remap["action_head."] = None  # the actor is a new MLP on [latent, s_t]

        state = {}
        for k, v in src.items():
            prefix = next((p for p in remap if k.startswith(p)), None)
            if prefix is None:
                state[k] = v
            elif remap[prefix] is not None:
                state[remap[prefix] + k[len(prefix):]] = v

        missing, unexpected = self.load_state_dict(state, strict=False)
        assert not unexpected, f"unexpected pretrained keys: {unexpected}"
        new_ok = ("critic.", "actor.") if self.fresh_heads else ("critic.", "actor.log_std")
        bad = [k for k in missing if not k.startswith(new_ok)]
        assert not bad, f"pretrained snapshot is missing trunk/actor keys: {bad}"
        if verbose:
            print(f"[MaskDPActorCritic] loaded {len(state)} tensors from {path}; new: {missing}")
        return list(state), missing

    # ------------------------------------------------------------ masks
    @staticmethod
    def _bool_mask(mask, states):
        B, L = states.shape[:2]
        if mask is None:
            return th.ones(B, L, dtype=th.bool, device=states.device)
        return mask.to(states.device) > 0.5 if mask.dtype != th.bool else mask.to(states.device)

    def _attn_mask(self, key_valid):
        """key_valid [B, n] -> bidirectional [B, 1, n, n] with padded keys removed.

        The diagonal is kept so padded query rows never become all -inf (NaN through att @ v).
        """
        n = key_valid.shape[1]
        allowed = key_valid[:, None, :].expand(-1, n, -1) | th.eye(n, dtype=th.bool, device=key_valid.device)
        return allowed.unsqueeze(1).to(self.pos_embed.dtype)

    # ------------------------------------------------------------ forward
    def encode(self, states, actions, rewards, key_valid):
        B, L, _ = states.shape
        pe = self.pos_embed[:, :L]
        s = self.state_embed(states) + self.mod_embed_s + pe
        a = self.action_embed(actions) + self.mod_embed_a + pe
        r = self.reward_embed(rewards) + self.mod_embed_r + pe
        x = th.stack([s, a, r], dim=2).reshape(B, self.N_TOK * L, self.n_embd)
        attn_mask = self._attn_mask(key_valid)
        for blk in self.encoder_blocks:
            x = blk(x, attn_mask)
        return self.encoder_norm(x)

    def decode(self, y, key_valid):
        """y [B, 3L, D] -> next-action latent [B, 1, D]"""
        B, n, D = y.shape
        L = n // self.N_TOK
        pe = self.pos_embed[:, :L]
        s = self.decoder_state_embed(y[:, 0::3]) + self.mod_embed_s + pe
        a = self.decoder_action_embed(y[:, 1::3]) + self.mod_embed_a + pe
        r = self.decoder_reward_embed(y[:, 2::3]) + self.mod_embed_r + pe
        ctx = th.stack([s, a, r], dim=2).reshape(B, n, D)

        q = self.decoder_action_embed(self.mask_token) + self.mod_embed_a + self.pos_embed[:, L:L + 1]
        x = th.cat([ctx, q.expand(B, -1, -1)], dim=1)                         # [B, 3L+1, D]
        key_valid = th.cat([key_valid, key_valid.new_ones(B, 1)], dim=1)
        attn_mask = self._attn_mask(key_valid)
        for blk in self.decoder_blocks:
            x = blk(x, attn_mask)
        return x[:, -1:]

    def forward(self, states, actions, rewards, mask=None, last_only=True):
        """Only the next action is decoded, so outputs always have L=1 (= last_only=True)."""
        assert last_only, "MaskDPActorCritic only decodes the next action (last_only=True)"
        B, L, _ = states.shape
        assert L <= self.context_len, f"window {L} > context_len {self.context_len}"
        assert actions.shape[:2] == (B, L) and actions.shape[-1] == self.act_dim
        if rewards.dim() == 2:
            rewards = rewards.unsqueeze(-1)
        key_valid = self._bool_mask(mask, states).repeat_interleave(self.N_TOK, dim=1)  # [B, 3L]

        z = self.decode(self.encode(states, actions, rewards, key_valid), key_valid)
        # Newest triplet of the left-padded window = s_t, the baseline's input.
        s_t = states[:, -1:] if self.fresh_heads else None
        mean, std = self.actor(z, s_t)
        value = self.critic(z, s_t).squeeze(-1)
        return mean, std, value
