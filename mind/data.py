from __future__ import annotations

import torch

from scoreboard.reading import UNKNOWN, Example


def pad_bytes(texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
    enc = [list(t.encode()) for t in texts]
    width = max([1, *(len(e) for e in enc)])
    x = torch.zeros(len(enc), width, dtype=torch.long)
    for i, e in enumerate(enc):
        x[i, : len(e)] = torch.tensor(e, dtype=torch.long)
    return x, x != 0


def _occurrences(fact: str, answer: str) -> list[tuple[int, int]]:
    # Byte offsets of every occurrence of the answer inside a fact. Supervision is the answer string only.
    f, a = fact.encode(), answer.encode()
    out, i = [], f.find(a)
    while i >= 0:
        out.append((i, i + len(a) - 1))
        i = f.find(a, i + 1)
    return out


def collate(batch: list[Example], device: str) -> dict[str, torch.Tensor]:
    # At least one (masked) fact slot: a batch whose every context is empty would otherwise give the encoder zero rows.
    b, n_facts = len(batch), max(1, *(len(e.facts) for e in batch))
    flat = [f for e in batch for f in e.facts]
    fx, _ = pad_bytes(flat)
    facts = torch.zeros(b, n_facts, fx.shape[1], dtype=torch.long)
    fact_mask = torch.zeros(b, n_facts, dtype=torch.bool)
    start = torch.zeros(b, n_facts, fx.shape[1], dtype=torch.bool)
    end = torch.zeros_like(start)
    k = 0
    for i, e in enumerate(batch):
        for j, f in enumerate(e.facts):
            facts[i, j] = fx[k]
            fact_mask[i, j] = True
            if e.answer != UNKNOWN:
                for s, t in _occurrences(f, e.answer):
                    start[i, j, s] = end[i, j, t] = True
            k += 1
    qx, _ = pad_bytes([e.question for e in batch])
    unknown = torch.tensor([e.answer == UNKNOWN for e in batch])
    # An answer absent from every fact would be unlearnable by copying; count it so it is visible, not hidden.
    uncopyable = int(sum(1 for i, e in enumerate(batch) if e.answer != UNKNOWN and not start[i].any()))
    out = {"facts": facts, "fact_mask": fact_mask, "question": qx, "start": start, "end": end, "unknown": unknown}
    out = {n: t.to(device) for n, t in out.items()}
    out["uncopyable"] = uncopyable
    return out
