from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

MAX_SPAN = 24


@dataclass
class Config:
    emb: int = 128
    hidden: int = 400
    layers: int = 2
    key: int = 128
    hops: int = 2


class Encoder(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.emb = nn.Embedding(256, c.emb, padding_idx=0)
        self.rnn = nn.GRU(c.emb, c.hidden, c.layers, batch_first=True, bidirectional=True)
        self.pool = nn.Linear(2 * c.hidden, 1)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        lengths = mask.sum(1).clamp(min=1).cpu()
        # Packing keeps the backward direction from reading padding.
        packed = pack_padded_sequence(self.emb(x), lengths, batch_first=True, enforce_sorted=False)
        h, _ = pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=x.shape[1])
        # An empty row (no bytes) would give an all-masked softmax and NaN; let it pool its first slot instead.
        safe = mask.clone()
        safe[:, 0] |= ~mask.any(1)
        w = self.pool(h).squeeze(-1).float().masked_fill(~safe, -1e9)
        return h, (torch.softmax(w, -1).unsqueeze(-1) * h.float()).sum(1)


class FastWeightReader(nn.Module):
    def __init__(self, c: Config):
        super().__init__()
        self.c = c
        d = 2 * c.hidden
        self.enc = Encoder(c)
        self.key, self.value, self.query = nn.Linear(d, c.key), nn.Linear(d, c.key), nn.Linear(d, c.key)
        self.beta = nn.Linear(d, 1)
        self.hop = nn.Linear(2 * c.key, c.key)
        self.use_hop = nn.Linear(c.key, 1)
        self.start, self.end, self.ctx = nn.Linear(d, c.key), nn.Linear(d, c.key), nn.Linear(c.key, c.key)
        self.unknown = nn.Sequential(nn.Linear(2 + 2 * c.key, c.key), nn.GELU(), nn.Linear(c.key, 1))

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def write(self, facts: torch.Tensor, fact_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        b, n, width = facts.shape
        flat = facts.view(b * n, width)
        h, p = self.enc(flat, flat != 0)
        h, p = h.view(b, n, width, -1), p.view(b, n, -1)
        k = F.normalize(self.key(p).float(), dim=-1)
        v = self.value(p).float()
        beta = torch.sigmoid(self.beta(p).float()).squeeze(-1) * fact_mask
        m = torch.zeros(b, self.c.key, self.c.key, device=facts.device)
        pk = torch.zeros_like(m)
        # The fast-weight state accumulates across writes, so it stays in fp32 even under fp16 autocast.
        with torch.autocast(facts.device.type, enabled=False):
            # One delta-rule write per fact, in order: a later fact with the same key overwrites an earlier one.
            for j in range(n):
                kj, bj = k[:, j], beta[:, j, None, None]
                m = m + bj * torch.einsum("bi,bj->bij", v[:, j] - torch.einsum("bij,bj->bi", m, kj), kj)
                pk = pk + bj * torch.einsum("bi,bj->bij", kj - torch.einsum("bij,bj->bi", pk, kj), kj)
        return {"M": m, "P": pk, "v": v, "h": h, "fact_mask": fact_mask, "byte_mask": facts != 0}

    def read(self, mem: dict[str, torch.Tensor], question: torch.Tensor) -> dict[str, torch.Tensor]:
        _, pq = self.enc(question, question != 0)
        q0 = F.normalize(self.query(pq).float(), dim=-1)
        q, reads, residuals = q0, [], []
        for hop in range(self.c.hops):
            with torch.autocast(question.device.type, enabled=False):
                r = torch.einsum("bij,bj->bi", mem["M"], q)
                residuals.append((q - torch.einsum("bij,bj->bi", mem["P"], q)).norm(dim=-1))
            reads.append(r)
            if hop + 1 < self.c.hops:
                q = F.normalize(self.hop(torch.cat([q, r], -1)).float(), dim=-1)
        g = torch.sigmoid(self.use_hop(q0).float())
        r = (1 - g) * reads[0] + g * reads[-1]
        e = torch.stack([residuals[0], residuals[-1]], -1)
        unk = self.unknown(torch.cat([e, r, q], -1).to(pq.dtype)).squeeze(-1)
        attn = torch.einsum("bnd,bd->bn", mem["v"], r).float().masked_fill(~mem["fact_mask"], -1e9)
        log_attn = torch.log_softmax(attn.float(), -1)
        c = self.ctx(r)
        sl = torch.einsum("bntd,bd->bnt", self.start(mem["h"]), c).float()
        el = torch.einsum("bntd,bd->bnt", self.end(mem["h"]), c).float()
        bm = mem["byte_mask"]
        sl = (torch.log_softmax(sl.masked_fill(~bm, -1e9), -1) + log_attn[..., None]).masked_fill(~bm, -1e9)
        el = (torch.log_softmax(el.masked_fill(~bm, -1e9), -1) + log_attn[..., None]).masked_fill(~bm, -1e9)
        return {"unknown_logit": unk.float(), "start": sl, "end": el}

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return self.read(self.write(batch["facts"], batch["fact_mask"]), batch["question"])


def loss_fn(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> torch.Tensor:
    unk = batch["unknown"].float()
    loss = F.binary_cross_entropy_with_logits(out["unknown_logit"], unk)
    ans = ~batch["unknown"]
    if ans.any():
        b = out["start"].shape[0]
        s = out["start"].view(b, -1).masked_fill(~batch["start"].view(b, -1), -1e9).logsumexp(-1)
        e = out["end"].view(b, -1).masked_fill(~batch["end"].view(b, -1), -1e9).logsumexp(-1)
        # Answers not copyable from any fact get no span loss; collate counts them.
        ok = ans & batch["start"].view(b, -1).any(-1)
        if ok.any():
            loss = loss - (s[ok] + e[ok]).mean()
    return loss


@torch.no_grad()
def decode(out: dict[str, torch.Tensor], facts: torch.Tensor) -> list[tuple[str, float]]:
    results = []
    p_unk = torch.sigmoid(out["unknown_logit"])
    for i in range(facts.shape[0]):
        width = out["start"].shape[-1]
        s = out["start"][i].reshape(-1)
        best, conf = None, -1.0
        for flat in s.topk(min(5, s.numel())).indices.tolist():
            j, t0 = divmod(flat, width)
            ends = out["end"][i, j, t0 : t0 + MAX_SPAN]
            t1 = t0 + int(ends.argmax())
            c = float(s[flat].exp() * ends.max().exp())
            if c > conf:
                best, conf = (j, t0, t1), c
        j, t0, t1 = best
        text = bytes(facts[i, j, t0 : t1 + 1].tolist()).decode(errors="replace").strip(" .")
        pu = float(p_unk[i])
        # Confidence means "I hold a real answer", as for every entrant.
        results.append(("unknown", 1 - pu) if pu > 0.5 else (text, (1 - pu) * conf))
    return results
