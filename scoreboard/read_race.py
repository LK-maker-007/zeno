from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path
from typing import Protocol

import numpy as np
from tqdm import tqdm

from scoreboard.metrics import auroc
from scoreboard.reading import CATEGORIES, UNKNOWN, Example, build, digest
from scoreboard.regex_reader import RegexReader
from scoreboard.world import normalize

ROOT = Path(__file__).resolve().parent.parent
# Smoke runs and development score on the validation worlds; the test worlds are kept for the pre-registered race.
WORLDS = {"test": range(800, 805), "val": range(900, 905)}


class Reader(Protocol):
    name: str

    def read(self, facts: tuple[str, ...], question: str) -> tuple[str, float]: ...


class ChildMindReader:
    # Adapter: the C1 byte model reads the same facts and question through its own prompt format.
    def __init__(self, ckpt: Path, threads: int = 2, device: str = "cpu"):
        import torch

        from childmind.data import prompt
        from childmind.infer import answer
        from childmind.model import ChildMind, Config

        torch.set_num_threads(threads)
        state = torch.load(ckpt, map_location=device)
        self.model = ChildMind(Config(**state["config"])).to(device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self._prompt, self._answer = prompt, answer
        self.name = f"C1-transformer ({ckpt.parent.parent.name})"

    def read(self, facts: tuple[str, ...], question: str) -> tuple[str, float]:
        return self._answer(self.model, self._prompt(list(facts), question))


class MindReader:
    # Adapter: the A1 fast-weight reader writes the facts into memory, then reads the question, one example at a time.
    def __init__(self, ckpt: Path, threads: int = 2, device: str = "cpu"):
        import torch

        from mind.data import collate
        from mind.model import Config, FastWeightReader, decode

        torch.set_num_threads(threads)
        state = torch.load(ckpt, map_location=device)
        self.model = FastWeightReader(Config(**state["config"])).to(device)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self._no_grad, self._collate, self._decode, self._device = torch.no_grad, collate, decode, device
        self.name = f"A1-fast-weight ({ckpt.parent.parent.name})"

    def read(self, facts: tuple[str, ...], question: str) -> tuple[str, float]:
        # The answer field is a placeholder: collate only uses it to build training targets.
        b = self._collate([Example(facts, question, UNKNOWN, "", "")], self._device)
        with self._no_grad():
            return self._decode(self.model(b), b["facts"])[0]


def score(reader: Reader, data: list[Example]) -> dict:
    rows, t0 = [], time.perf_counter()
    for e in tqdm(data, desc=f"  {reader.name}", leave=False):
        got, conf = reader.read(e.facts, e.question)
        rows.append((e.category, e.answer == UNKNOWN, normalize(got) == normalize(e.answer), conf))
    ms = 1000 * (time.perf_counter() - t0) / len(data)
    out = {"name": reader.name, "ms_per_read": ms, "categories": {}}
    for c in [*CATEGORIES, "ALL"]:
        sel = [r for r in rows if c in ("ALL", r[0])]
        ans = [r for r in sel if not r[1]]
        unk = [r for r in sel if r[1]]
        out["categories"][c] = {
            "n": len(sel),
            "acc": float(np.mean([r[2] for r in sel])),
            "em_answerable": float(np.mean([r[2] for r in ans])) if ans else float("nan"),
            "acc_unknown": float(np.mean([r[2] for r in unk])) if unk else float("nan"),
            "auroc": auroc([r[3] for r in sel], [r[2] for r in sel]),
        }
    return out


def report(results: list[dict]) -> None:
    print(f"\naccuracy by category (answerable EM / unknown accuracy in brackets)\n{'category':<18}{'n':>6}", end="")
    for r in results:
        print(f"{r['name'][:30]:>34}", end="")
    print()
    for c in [*CATEGORIES, "ALL"]:
        # Categories that fall back to `simple` come out smaller than requested, so show their size.
        print(f"{c:<18}{results[0]['categories'][c]['n']:>6}", end="")
        for r in results:
            m = r["categories"][c]
            print(f"{m['acc']:>10.3f} ({m['em_answerable']:.3f} / {m['acc_unknown']:.3f})".rjust(34), end="")
        print()
    print(f"{'AUROC (ALL)':<24}" + "".join(f"{r['categories']['ALL']['auroc']:>34.3f}" for r in results))
    print(f"{'ms per read':<24}" + "".join(f"{r['ms_per_read']:>34.2f}" for r in results), flush=True)


def run(a: argparse.Namespace) -> Path:
    worlds = WORLDS[a.worlds]
    data = build(worlds, a.n, CATEGORIES, seed=a.seed)
    print(
        f"{a.worlds} worlds {worlds.start}-{worlds.stop - 1}  {a.n} per category  n={len(data)}  "
        f"digest {digest(data)}  seed {a.seed}  device {a.device}",
        flush=True,
    )
    for c in CATEGORIES:
        e = next(x for x in data if x.category == c)
        print(f"--- {c}\n" + "\n".join(f"  <f> {f}" for f in e.facts) + f"\n  <q> {e.question}\n  gold: {e.answer}")
    readers: list[Reader] = [RegexReader(all_templates=False), RegexReader(all_templates=True)]
    readers += [ChildMindReader(p, a.threads, a.device) for p in a.childmind]
    readers += [MindReader(p, a.threads, a.device) for p in a.mind]
    results = []
    for r in readers:
        print(f"racing {r.name}", flush=True)
        results.append(score(r, data))
    report(results)
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / f"read_seed{a.seed}_{digest(data)}_{int(time.time())}.json"
    path.write_text(
        json.dumps(
            {
                "command": " ".join(["python -m scoreboard.read_race", *sys.argv[1:]]),
                "seed": a.seed,
                "worlds": a.worlds,
                "device": a.device,
                "digest": digest(data),
                "python": platform.python_version(),
                "results": results,
            },
            indent=2,
        )
    )
    print(f"wrote {path}", flush=True)
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100, help="examples per category per world (5 worlds)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--worlds", choices=sorted(WORLDS), default="test")
    ap.add_argument("--childmind", type=Path, nargs="*", default=[])
    ap.add_argument("--mind", type=Path, nargs="*", default=[], help="A1 fast-weight reader checkpoints")
    ap.add_argument("--device", default="cpu", help="ms per read is only comparable across runs on the same device")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--out", type=Path, default=ROOT / "results")
    run(ap.parse_args())
