from __future__ import annotations

import argparse
import json
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np

from scoreboard.entrants import CoinFlip, Entrant, LexicalCosine, Oracle
from scoreboard.metrics import aurc, auroc, cluster_bootstrap_diff, selective_accuracy
from scoreboard.world import World, make_world, normalize

ROOT = Path(__file__).resolve().parent.parent
# Answerable known facts against the two unanswerable strata the memory must catch (plan v3 §7 M1).
CONFIRMATORY = ("i_known", "ii_near_miss", "iii_unseen_subject")
ENTRANTS = {
    "oracle": lambda w, a: Oracle(w.facts, w.probes),
    "coinflip": lambda w, a: CoinFlip(w.facts, w.probes, seed=w.seed),
    "lexical": lambda w, a: LexicalCosine(),
    "childmind": lambda w, a: _childmind(a.childmind_ckpt),
}


def _childmind(ckpt: Path) -> Entrant:
    # Imported here so the scoreboard runs without torch unless a model entrant is raced.
    from childmind.entrant import ChildMindEntrant

    return ChildMindEntrant(ckpt)


def correct(answer: str, gold: str | None) -> bool:
    return gold is not None and normalize(answer) == normalize(gold)


def race(entrant: Entrant, world: World, verbose: bool = True) -> dict:
    batch0 = {(f.subject, f.relation) for f in world.facts if f.batch == 0}
    early = [p for p in world.probes if p.stratum == "i_known" and (p.subject, p.relation) in batch0]
    n_batches = max(f.batch for f in world.facts) + 1
    tell_s, curve = 0.0, []
    for b in range(n_batches):
        batch = [f for f in world.facts if f.batch == b]
        for f in batch:
            t0 = time.perf_counter()
            entrant.tell(f.text, f.source, f.time)
            tell_s += time.perf_counter() - t0
        acc = float(np.mean([correct(entrant.ask(p.question)[0], p.gold) for p in early]))
        curve.append(acc)
        if verbose:
            print(
                f"  {entrant.name:<26} batch {b + 1}/{n_batches}  told {len(batch):>4}  "
                f"recall on batch-1 facts {acc:.3f} (n={len(early)})",
                flush=True,
            )

    answers, confs, ask_s = [], [], 0.0
    for p in world.probes:
        t0 = time.perf_counter()
        a, c = entrant.ask(p.question)
        ask_s += time.perf_counter() - t0
        answers.append(a)
        confs.append(float(c))
    ok = np.array([correct(a, p.gold) for a, p in zip(answers, world.probes, strict=True)])
    conf = np.array(confs)
    strata = np.array([p.stratum for p in world.probes])
    cmask = np.isin(strata, CONFIRMATORY)

    per_stratum = {
        s: {
            "n": int((strata == s).sum()),
            "accuracy": float(ok[strata == s].mean()),
            "mean_conf": float(conf[strata == s].mean()),
        }
        for s in dict.fromkeys(strata)
    }
    return {
        "name": entrant.name,
        "T1_recall_known": per_stratum["i_known"]["accuracy"],
        "T1_recall_paraphrase": per_stratum["iv_paraphrase"]["accuracy"],
        "T2_recall_curve": curve,
        "T2_forgetting": curve[0] - curve[-1],
        "T3_auroc": auroc(conf[cmask], ok[cmask]),
        "T3_selective_acc_50": selective_accuracy(conf[cmask], ok[cmask], 0.5),
        "T3_selective_acc_80": selective_accuracy(conf[cmask], ok[cmask], 0.8),
        "T3_aurc": aurc(conf[cmask], ok[cmask]),
        "T5_ms_per_tell": 1000 * tell_s / len(world.facts),
        "T5_ms_per_ask": 1000 * ask_s / len(world.probes),
        "per_stratum": per_stratum,
        "_conf": conf[cmask],
        "_ok": ok[cmask],
        "_subjects": np.array([p.subject for p in world.probes])[cmask],
    }


def self_check(world: World) -> None:
    o = race(Oracle(world.facts, world.probes), world, verbose=False)
    c = race(CoinFlip(world.facts, world.probes, seed=world.seed), world, verbose=False)
    problems = []
    if o["T1_recall_known"] != 1.0 or o["T1_recall_paraphrase"] != 1.0 or o["T3_auroc"] != 1.0:
        problems.append(
            f"oracle not perfect: T1 {o['T1_recall_known']}, {o['T1_recall_paraphrase']}, AUROC {o['T3_auroc']}"
        )
    if not 0.44 <= c["T3_auroc"] <= 0.56:
        problems.append(f"coin flip AUROC {c['T3_auroc']:.3f} outside [0.44, 0.56]")
    if problems:
        sys.exit("self-check FAILED: " + "; ".join(problems))
    print(f"self-check ok: oracle AUROC {o['T3_auroc']:.3f}, coin flip AUROC {c['T3_auroc']:.3f}", flush=True)


def report(results: list[dict], reference: str | None) -> None:
    print(
        f"\n{'entrant':<26}{'recall':>8}{'para':>7}{'forget':>8}{'AUROC':>8}{'sel@50':>8}{'sel@80':>8}"
        f"{'AURC':>8}{'ms/tell':>9}{'ms/ask':>9}"
    )
    print("-" * 101)
    for r in results:
        print(
            f"{r['name']:<26}{r['T1_recall_known']:>8.3f}{r['T1_recall_paraphrase']:>7.3f}"
            f"{r['T2_forgetting']:>+8.3f}{r['T3_auroc']:>8.3f}{r['T3_selective_acc_50']:>8.3f}"
            f"{r['T3_selective_acc_80']:>8.3f}{r['T3_aurc']:>8.3f}{r['T5_ms_per_tell']:>9.3f}"
            f"{r['T5_ms_per_ask']:>9.3f}"
        )
    print("\nper stratum: accuracy / mean confidence")
    strata = list(results[0]["per_stratum"])
    print(f"{'entrant':<26}" + "".join(f"{s:>22}" for s in strata))
    for r in results:
        cells = "".join(
            f"{r['per_stratum'][s]['accuracy']:>13.3f} / {r['per_stratum'][s]['mean_conf']:.3f}" for s in strata
        )
        print(f"{r['name']:<26}{cells}")
    ref = next((r for r in results if r["key"] == reference), None)
    if ref is None:
        return
    print(f"\npaired vs {reference}, AURC (lower is better), cluster bootstrap by subject, 95% CI")
    for r in results:
        if r is ref:
            continue
        d, lo, hi = cluster_bootstrap_diff(aurc, (r["_conf"], r["_ok"]), (ref["_conf"], ref["_ok"]), r["_subjects"])
        print(f"  {r['name']:<26} dAURC {d:+.3f}  [{lo:+.3f}, {hi:+.3f}]", flush=True)


def run(seed: int, entrants: list[str], reference: str | None, out_dir: Path, args: argparse.Namespace) -> Path:
    world = make_world(seed)
    counts = {}
    for p in world.probes:
        counts[p.stratum] = counts.get(p.stratum, 0) + 1
    print(
        f"world seed {seed}  digest {world.digest()}  facts {len(world.facts)}  "
        f"people {len({f.subject for f in world.facts})}  probes {counts}",
        flush=True,
    )
    self_check(world)
    results = []
    for name in entrants:
        print(f"racing {name}", flush=True)
        results.append({**race(ENTRANTS[name](world, args), world), "key": name})
    report(results, reference)
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"\npeak RSS {peak_mb:.0f} MB", flush=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"seed{seed}_{world.digest()}_{int(time.time())}.json"
    path.write_text(
        json.dumps(
            {
                "command": " ".join(sys.argv),
                "seed": seed,
                "world_digest": world.digest(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "peak_rss_mb": peak_mb,
                "results": [{k: v for k, v in r.items() if not k.startswith("_")} for r in results],
            },
            indent=2,
        )
    )
    print(f"wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--entrants", nargs="+", default=["oracle", "coinflip", "lexical"], choices=sorted(ENTRANTS))
    ap.add_argument("--reference", default="lexical")
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    ap.add_argument("--childmind-ckpt", type=Path, default=ROOT / "runs" / "childmind" / "last.pt")
    a = ap.parse_args()
    run(a.seed, a.entrants, a.reference, a.out, a)
