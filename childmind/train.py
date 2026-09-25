from __future__ import annotations

import argparse
import json
import math
import platform
import random
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from childmind.data import UNKNOWN, build, encode, prompt
from childmind.infer import answer
from childmind.model import ChildMind, Config
from scoreboard import reading

GATE = 0.95


def default_out() -> Path:
    kaggle = Path("/kaggle/working")
    return kaggle / "childmind" if kaggle.exists() else Path(__file__).resolve().parent.parent / "runs" / "childmind"


def collate(batch: list[tuple[str, str]], ctx: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    seqs = [encode(p + a + "\n") for p, a in batch]
    longest = max(len(s) for s in seqs)
    if longest > ctx + 1:
        sys.exit(f"example of {longest} bytes exceeds ctx {ctx}")
    x = torch.zeros(len(seqs), longest - 1, dtype=torch.long)
    y = torch.full((len(seqs), longest - 1), -100, dtype=torch.long)
    for i, s in enumerate(seqs):
        x[i, : len(s) - 1] = torch.tensor(s[:-1])
        y[i, : len(s) - 1] = torch.tensor(s[1:])
    return x.to(device), y.to(device)


def describe(name: str, data: list[tuple[str, str]]) -> None:
    lens = [len(encode(p + a)) for p, a in data]
    unknown = sum(a == UNKNOWN for _, a in data)
    print(
        f"{name}: {len(data)} examples, answerable {len(data) - unknown} / unknown {unknown}, "
        f"bytes min {min(lens)} median {statistics.median(lens):.0f} max {max(lens)}",
        flush=True,
    )


@torch.no_grad()
def evaluate(
    model: ChildMind, val: list[tuple[str, str]], heldout: list[tuple[str, str]], ctx: int, device: str, bs: int
) -> dict:
    model.eval()
    losses = []
    for i in range(0, len(val), bs):
        x, y = collate(val[i : i + bs], ctx, device)
        losses.append(float(F.cross_entropy(model(x).float().flatten(0, 1), y.flatten(), ignore_index=-100)))

    def scores(data: list[tuple[str, str]]) -> tuple[float, float]:
        hit_known, n_known, hit_unknown, n_unknown = 0, 0, 0, 0
        for p, a in tqdm(data, desc="  eval", leave=False):
            got = answer(model, p)[0]
            if a == UNKNOWN:
                n_unknown += 1
                hit_unknown += got == UNKNOWN
            else:
                n_known += 1
                hit_known += got == a
        return hit_known / max(1, n_known), hit_unknown / max(1, n_unknown)

    em, unk = scores(val)
    em_h, unk_h = scores(heldout)
    model.train()
    return {
        "val_loss": statistics.mean(losses),
        "em_answerable": em,
        "acc_unknown": unk,
        "em_answerable_heldout_q": em_h,
        "acc_unknown_heldout_q": unk_h,
    }


def run(a: argparse.Namespace) -> None:
    random.seed(a.seed)
    torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(a.threads)
    print(
        f"python {platform.python_version()}  torch {torch.__version__}  device {device}"
        f"{' ' + torch.cuda.get_device_name(0) if device == 'cuda' else f' threads {a.threads}'}  seed {a.seed}",
        flush=True,
    )

    if a.task == "reading":
        # Same examples as the mind/ entrant (scoreboard v2), rendered into C1's prompt format.
        def as_pairs(ex: list[reading.Example]) -> list[tuple[str, str]]:
            return [(prompt(list(e.facts), e.question), e.answer) for e in ex]

        held_cats = [c for c in reading.CATEGORIES if c not in reading.TRAIN_CATEGORIES]
        train = as_pairs(
            reading.build(range(1000, 1000 + a.train_worlds), a.per_category, reading.TRAIN_CATEGORIES, a.seed)
        )
        val = as_pairs(reading.build(range(900, 905), a.val_per_category, reading.TRAIN_CATEGORIES, a.seed + 1))
        heldout = as_pairs(reading.build(range(900, 905), a.val_per_category, held_cats, a.seed + 2))
    elif a.overfit:
        train = build(range(1000, 1001), a.overfit, seed=a.seed)
        val, heldout = train, build(range(1000, 1001), a.overfit, seed=a.seed, heldout_question=True)
    else:
        train = build(range(1000, 1000 + a.train_worlds), a.per_world, seed=a.seed)
        val = build(range(900, 900 + a.val_worlds), a.val_per_world, seed=a.seed + 1)
        heldout = build(range(900, 900 + a.val_worlds), a.val_per_world, seed=a.seed + 2, heldout_question=True)
    describe("train", train)
    describe("val", val)
    describe("val, held-out question phrasing", heldout)
    for p, ans in train[:3]:
        print("--- sample\n" + p + ans, flush=True)

    c = Config(ctx=a.ctx, d=a.d, layers=a.layers, heads=a.heads)
    model = ChildMind(c).to(device)
    print(f"model {asdict(c)}  params {model.n_params():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.1, betas=(0.9, 0.95))
    warm = max(1, a.steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, a.steps) / a.steps))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": asdict(c), "args": vars(a) | {"out": str(out)}}, indent=2))
    log = (out / "metrics.jsonl").open("a")

    rng = random.Random(a.seed)
    t0, tokens = time.time(), 0
    bar = tqdm(range(1, a.steps + 1), desc="train")
    for step in bar:
        x, y = collate(rng.sample(train, a.batch), a.ctx, device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            loss = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten(), ignore_index=-100)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        tokens += int((y != -100).sum())
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, val, heldout, a.ctx, device, a.batch)
            m |= {"step": step, "train_loss": loss.item(), "tokens": tokens, "tok_per_s": tokens / (time.time() - t0)}
            print(
                f"step {step}  " + "  ".join(f"{k} {v:.4f}" for k, v in m.items() if isinstance(v, float)), flush=True
            )
            log.write(json.dumps(m) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "config": asdict(c), "step": step, "metrics": m}, out / "last.pt")
    log.close()
    passed = m["em_answerable"] >= GATE and m["acc_unknown"] >= GATE
    print(
        f"\nC1 gate (in-context answer EM and unknown accuracy >= {GATE} on validation worlds): "
        f"{'PASS' if passed else 'FAIL'}  em {m['em_answerable']:.3f}  unknown {m['acc_unknown']:.3f}  "
        f"held-out phrasing em {m['em_answerable_heldout_q']:.3f} (reported, not gated)",
        flush=True,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--train-worlds", type=int, default=200)
    ap.add_argument("--per-world", type=int, default=1000)
    ap.add_argument("--val-worlds", type=int, default=5)
    ap.add_argument("--val-per-world", type=int, default=100)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--overfit", type=int, default=0, help="train and validate on this many examples (sanity check)")
    ap.add_argument("--task", choices=["v1", "reading"], default="v1", help="v1 world, or scoreboard v2 reading task")
    ap.add_argument("--per-category", type=int, default=143, help="reading task: per training category per world")
    ap.add_argument("--val-per-category", type=int, default=40, help="reading task: per category per validation world")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(default_out()))
    run(ap.parse_args())
