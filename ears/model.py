from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

# QuartzNet (Kriman et al. 2020, arXiv 1910.10261) Table 1: (kernel, channels) for C1, blocks B1..B5, C2, C3.
C1 = (33, 256)
BLOCKS = [(33, 256), (39, 256), (51, 512), (63, 512), (75, 512)]
C2 = (87, 512)
C3 = 1024


@dataclass
class Config:
    n_mels: int = 64
    vocab: int = 29
    repeat: int = 1
    modules: int = 5
    dropout: float = 0.0


class SepConv(nn.Module):
    # Time-channel separable convolution: depthwise over time, pointwise over channels, then batch norm.
    def __init__(self, cin: int, cout: int, k: int, stride: int = 1, dilation: int = 1):
        super().__init__()
        self.dw = nn.Conv1d(cin, cin, k, stride, dilation * (k // 2), dilation, groups=cin, bias=False)
        self.pw = nn.Conv1d(cin, cout, 1, bias=False)
        self.bn = nn.BatchNorm1d(cout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.pw(self.dw(x)))


class Block(nn.Module):
    def __init__(self, cin: int, cout: int, k: int, modules: int, dropout: float):
        super().__init__()
        self.mods = nn.ModuleList(SepConv(cin if i == 0 else cout, cout, k) for i in range(modules))
        self.res = nn.Sequential(nn.Conv1d(cin, cout, 1, bias=False), nn.BatchNorm1d(cout))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x
        for i, m in enumerate(self.mods):
            y = m(y)
            if i + 1 < len(self.mods):
                y = self.drop(F.relu(y))
        return self.drop(F.relu(y + self.res(x)))


class QuartzNet(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        self.c1 = SepConv(c.n_mels, C1[1], C1[0], stride=2)
        blocks, cin = [], C1[1]
        for k, ch in BLOCKS:
            for _ in range(c.repeat):
                blocks.append(Block(cin, ch, k, c.modules, c.dropout))
                cin = ch
        self.blocks = nn.Sequential(*blocks)
        # The paper puts "dilation 2" on C4, a 1x1 layer where it does nothing; C2 is the wide kernel.
        self.c2 = SepConv(cin, C2[1], C2[0], dilation=2)
        self.c3 = nn.Sequential(nn.Conv1d(C2[1], C3, 1, bias=False), nn.BatchNorm1d(C3))
        self.c4 = nn.Conv1d(C3, c.vocab, 1)

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # x: (B, n_mels, T). Returns logits (B, vocab, ceil(T / 2)) and the output lengths.
        x = F.relu(self.c1(x))
        x = self.blocks(x)
        x = F.relu(self.c2(x))
        x = F.relu(self.c3(x))
        return self.c4(x), (lengths + 1) // 2
