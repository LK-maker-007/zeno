from __future__ import annotations

import numpy as np


def edits(ref: list, hyp: list) -> int:
    # Levenshtein distance: substitutions, deletions and insertions.
    prev = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        cur = [i]
        for j, h in enumerate(hyp, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (r != h)))
        prev = cur
    return prev[-1]


def error_rate(refs: list[str], hyps: list[str], unit: str) -> tuple[int, int]:
    split = str.split if unit == "word" else list
    errors = sum(edits(split(r), split(h)) for r, h in zip(refs, hyps, strict=True))
    return errors, sum(len(split(r)) for r in refs)


def speaker_bootstrap(
    errors: list[int], lengths: list[int], speakers: list[str], n_boot: int = 2000, seed: int = 0
) -> tuple[float, float, float]:
    # Corpus error rate with a 95% CI. Utterances from one speaker are correlated, so speakers are resampled.
    ids = sorted(set(speakers))
    e = np.array([sum(x for x, s in zip(errors, speakers, strict=True) if s == k) for k in ids], float)
    n = np.array([sum(x for x, s in zip(lengths, speakers, strict=True) if s == k) for k in ids], float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(ids), (n_boot, len(ids)))
    rates = e[draws].sum(1) / n[draws].sum(1)
    lo, hi = np.percentile(rates, [2.5, 97.5])
    return float(e.sum() / n.sum()), float(lo), float(hi)
