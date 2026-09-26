from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch

from ears.data import cache, index


def run(a: argparse.Namespace) -> None:
    # Decodes the audio once and saves the features, so every training job reuses them instead of each reading
    # 28k files from network storage.
    print(f"torch {torch.__version__}  workers {a.workers}  root {a.root}", flush=True)
    out = {}
    for spec in a.splits:
        name, subset = spec.split("=")
        t0 = time.time()
        utts = index(Path(a.root), [subset])
        speakers = len({u.speaker for u in utts})
        print(f"{name} = {subset}: {len(utts)} utterances, {speakers} speakers, indexed in {time.time() - t0:.1f} s")
        if not utts:
            raise SystemExit(f"{subset} not found under {a.root}")
        t0 = time.time()
        data = cache(utts, a.workers, f"features {name}")
        print(f"{name}: {len(data)} utterances decoded in {time.time() - t0:.0f} s", flush=True)
        out[name] = {"utts": [(str(u.audio), u.text, u.speaker, u.subset) for u in utts], "data": data}
    t0 = time.time()
    torch.save(out, a.out)
    print(f"wrote {a.out}: {Path(a.out).stat().st_size / 1e9:.2f} GB in {time.time() - t0:.0f} s", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--splits", nargs="+", default=["train=train-clean-100", "dev=dev-clean", "test=test-clean"])
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", required=True)
    run(ap.parse_args())
