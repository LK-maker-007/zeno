from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

VOCAB = 256


@dataclass
class Config:
    ctx: int = 512
    d: int = 256
    layers: int = 6
    heads: int = 8
    dropout: float = 0.0


class Block(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.heads = c.heads
        self.ln1, self.ln2 = nn.LayerNorm(c.d), nn.LayerNorm(c.d)
        self.qkv, self.proj = nn.Linear(c.d, 3 * c.d), nn.Linear(c.d, c.d)
        self.mlp = nn.Sequential(nn.Linear(c.d, 4 * c.d), nn.GELU(), nn.Linear(4 * c.d, c.d))
        self.drop = nn.Dropout(c.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(d, dim=2)
        q, k, v = (z.view(b, t, self.heads, d // self.heads).transpose(1, 2) for z in (q, k, v))
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.drop(self.proj(a.transpose(1, 2).reshape(b, t, d)))
        return x + self.drop(self.mlp(self.ln2(x)))


class ChildMind(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.tok, self.pos = nn.Embedding(VOCAB, c.d), nn.Embedding(c.ctx, c.d)
        self.blocks = nn.ModuleList(Block(c) for _ in range(c.layers))
        self.ln = nn.LayerNorm(c.d)
        self.head = nn.Linear(c.d, VOCAB, bias=False)
        self.head.weight = self.tok.weight
        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module) -> None:
        if isinstance(m, nn.Linear | nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)
        if isinstance(m, nn.Linear) and m.bias is not None:
            nn.init.zeros_(m.bias)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        x = self.tok(idx) + self.pos(torch.arange(idx.shape[1], device=idx.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln(x))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
