from __future__ import annotations

import numpy as np


def auroc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    pos, neg = s[y], s[~y]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = _midranks(np.concatenate([pos, neg]))
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _midranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x))
    i = 0
    while i < len(x):
        j = i
        while j < len(x) and xs[j] == xs[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1) + 1
        i = j
    return ranks


def delong_paired(scores_a, scores_b, labels) -> tuple[float, float, float]:
    # Fast DeLong (Sun & Xu 2014): returns (auc_a - auc_b, its standard error, two-sided z statistic).
    y = np.asarray(labels, bool)
    sa, sb = np.asarray(scores_a, float), np.asarray(scores_b, float)
    m, n = int(y.sum()), int((~y).sum())
    v10, v01, aucs = [], [], []
    for s in (sa, sb):
        pos, neg = s[y], s[~y]
        tx, ty, tz = _midranks(pos), _midranks(neg), _midranks(np.concatenate([pos, neg]))
        aucs.append((tz[:m].sum() - m * (m + 1) / 2) / (m * n))
        v10.append((tz[:m] - tx) / n)
        v01.append(1.0 - (tz[m:] - ty) / m)
    s10, s01 = np.cov(np.vstack(v10)), np.cov(np.vstack(v01))
    cov = s10 / m + s01 / n
    diff = aucs[0] - aucs[1]
    se = float(np.sqrt(max(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1], 0.0)))
    return float(diff), se, float(diff / se) if se > 0 else float("nan")


def cluster_bootstrap_diff(stat, a, b, clusters, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    # Items about one subject are correlated, so resample subjects, not items.
    rng = np.random.default_rng(seed)
    sa, ya = np.asarray(a[0], float), np.asarray(a[1], bool)
    sb, yb = np.asarray(b[0], float), np.asarray(b[1], bool)
    c = np.asarray(clusters)
    ids = np.unique(c)
    members = {k: np.flatnonzero(c == k) for k in ids}
    point = stat(sa, ya) - stat(sb, yb)
    diffs = []
    for _ in range(n_boot):
        idx = np.concatenate([members[k] for k in rng.choice(ids, len(ids), replace=True)])
        d = stat(sa[idx], ya[idx]) - stat(sb[idx], yb[idx])
        if not np.isnan(d):
            diffs.append(d)
    if not diffs:
        return float(point), float("nan"), float("nan")
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _expected_errors(scores, labels) -> np.ndarray:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    order = np.argsort(-s, kind="mergesort")
    ss, err = s[order], (~y[order]).astype(float)
    # Order inside a tie group is arbitrary, so each position carries the group's mean error:
    # the expected value over all orderings, instead of whatever order the probes arrived in.
    i = 0
    while i < len(ss):
        j = i
        while j < len(ss) and ss[j] == ss[i]:
            j += 1
        err[i:j] = err[i:j].mean()
        i = j
    return err


def selective_accuracy(scores, labels, coverage: float) -> float:
    err = _expected_errors(scores, labels)
    k = max(1, round(coverage * len(err)))
    return float(1.0 - err[:k].mean())


def aurc(scores, labels) -> float:
    err = _expected_errors(scores, labels)
    return float((np.cumsum(err) / np.arange(1, len(err) + 1)).mean())
