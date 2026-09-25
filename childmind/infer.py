from __future__ import annotations

import math

import torch

from childmind.data import UNKNOWN, decode, encode
from childmind.model import ChildMind

NEWLINE = 10


@torch.no_grad()
def answer(model: ChildMind, text: str, max_new: int = 24) -> tuple[str, float]:
    device = next(model.parameters()).device
    x = torch.tensor([encode(text)], device=device)
    out, logp = [], 0.0
    for _ in range(max_new):
        lp = torch.log_softmax(model(x[:, -model.c.ctx :])[0, -1].float(), dim=-1)
        nxt = int(lp.argmax())
        logp += float(lp[nxt])
        if nxt == NEWLINE:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    ans = decode(out).strip()
    p = math.exp(logp)
    # Confidence means "I hold a real answer". A confident "unknown" is therefore a low score, not a high one.
    return ans, (1.0 - p) if ans == UNKNOWN else p
