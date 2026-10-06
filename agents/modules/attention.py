import math
import logging

import torch
import torch.nn as nn
from torch.nn import functional as F



logger = logging.getLogger(__name__)


class mySequential(nn.Sequential):
    def forward(self, *inputs):
        for module in self._modules.values():
            if type(inputs) == tuple:
                inputs = module(*inputs)
            else:
                raise NotImplementedError
                # inputs = module(inputs)
        return inputs


class CausalSelfAttention(nn.Module):
    """
    A vanilla multi-head masked self-attention layer with a projection at the end.
    It is possible to use torch.nn.MultiheadAttention here but I am including an
    explicit implementation here to show that there is nothing too scary here.
    """

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        # regularization
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        # output projection
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.n_head = config.n_head

    def forward(self, x, mask):
        (
            B,
            T,
            C,
        ) = x.size()  # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        k = (
            self.key(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        q = (
            self.query(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)
        v = (
            self.value(x).view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        )  # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(mask[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)
        y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side

        # output projection
        y = self.resid_drop(self.proj(y))
        return y


class Block(nn.Module):
    """an unassuming Transformer block"""
    def __init__(self, config):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, mask):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.mlp(self.ln2(x))
        return x


class CrossAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.query = nn.Linear(config.n_embd, config.n_embd)
        self.key = nn.Linear(config.n_embd, config.n_embd)
        self.value = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.attn_pdrop)
        self.resid_drop = nn.Dropout(config.resid_pdrop)
        self.proj = nn.Linear(config.n_embd, config.n_embd)

    def forward(self, x, context, mask=None):
        B, Tq, C = x.size()
        Tk = context.size(1)
        hs = C // self.n_head

        q = self.query(x).view(B, Tq, self.n_head, hs).transpose(1, 2)
        k = self.key(context).view(B, Tk, self.n_head, hs).transpose(1, 2)
        v = self.value(context).view(B, Tk, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))
        if mask is not None:
            att = att.masked_fill(mask[:, :, :Tq, :Tk] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        att = self.attn_drop(att)

        y = (att @ v).transpose(1, 2).contiguous().view(B, Tq, C)
        return self.resid_drop(self.proj(y))


class CrossBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_q = nn.LayerNorm(config.n_embd)
        self.ln_kv = nn.LayerNorm(config.n_embd)
        self.ln_mlp = nn.LayerNorm(config.n_embd)
        self.attn = CrossAttention(config)
        self.mlp = nn.Sequential(
            nn.Linear(config.n_embd, 4 * config.n_embd),
            nn.GELU(),
            nn.Linear(4 * config.n_embd, config.n_embd),
            nn.Dropout(config.resid_pdrop),
        )

    def forward(self, x, context, mask=None):
        x = x + self.attn(self.ln_q(x), self.ln_kv(context), mask)
        x = x + self.mlp(self.ln_mlp(x))
        return x

class RecursiveBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        D = cfg.n_embd
        self.ln_sa = nn.LayerNorm(D)
        self.self_attn = CausalSelfAttention(cfg)
        self.ln_q = nn.LayerNorm(D)
        self.ln_kv = nn.LayerNorm(D)
        self.cross_attn = CrossAttention(cfg)
        self.ln_mlp = nn.LayerNorm(D)
        self.mlp = nn.Sequential(
            nn.Linear(D, 4 * D),
            nn.GELU(),
            nn.Linear(4 * D, D),
            nn.Dropout(cfg.resid_pdrop),
        )

    def forward(self, h, ctx, n_cross, mask):
        h = h + self.self_attn(self.ln_sa(h), mask)
        q = h[:, :n_cross]
        q = q + self.cross_attn(self.ln_q(q), self.ln_kv(ctx))
        h = torch.cat([q, h[:, n_cross:]], dim=1)
        return h + self.mlp(self.ln_mlp(h))