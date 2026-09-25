from __future__ import annotations

import argparse
import collections
import json
import math
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
from tqdm import tqdm

from mind.data import collate
from mind.model import Config, FastWeightReader, decode, loss_fn
from scoreboard.reading import CATEGORIES, TRAIN_CATEGORIES, UNKNOWN, Example, build
from scoreboard.world import normalize


def default_out() -> Path:
    kaggle = Path("/kaggle/working")
    return kaggle / "mind" if kaggle.exists() else Path(__file__).resolve().parent.parent / "runs" / "mind"


@torch.no_grad()
def evaluate(model: FastWeightReader, data: list[Example], device: str, bs: int) -> dict:
    model.eval()
    hits: dict[str, list[bool]] = collections.defaultdict(list)
    for i in range(0, len(data), bs):
        chunk = data[i : i + bs]
        b = collate(chunk, device)
        for e, (got, _) in zip(chunk, decode(model(b), b["facts"]), strict=True):
            ok = normalize(got) == normalize(e.answer)
            hits[e.category].append(ok)
            hits["ans" if e.answer != UNKNOWN else "unk"].append(ok)
    model.train()
    return {k: sum(v) / len(v) for k, v in hits.items()}


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
    train = build(range(1000, 1000 + a.train_worlds), a.per_category, TRAIN_CATEGORIES, seed=a.seed)
    val = build(range(900, 905), a.val_per_category, CATEGORIES, seed=a.seed + 1)
    for name, d in (("train", train), ("val", val)):
        cats = collections.Counter(e.category for e in d)
        unk = sum(e.answer == UNKNOWN for e in d)
        print(
            f"{name}: {len(d)} examples, unknown {unk} / answerable {len(d) - unk}, by category {dict(cats)}",
            flush=True,
        )
    for e in train[:2]:
        print("--- sample\n" + "\n".join(f"<f> {f}" for f in e.facts) + f"\n<q> {e.question}\n<a> {e.answer}")

    c = Config(hidden=a.hidden)
    model = FastWeightReader(c).to(device)
    print(f"model {asdict(c)}  params {model.n_params():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.01)
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
    t0, uncopyable = time.time(), 0
    bar = tqdm(range(1, a.steps + 1), desc="train")
    for step in bar:
        b = collate(rng.sample(train, a.batch), device)
        uncopyable += b["uncopyable"]
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            loss = loss_fn(model(b), b)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        sched.step()
        bar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{sched.get_last_lr()[0]:.2e}")
        if step % a.eval_every == 0 or step == a.steps:
            m = evaluate(model, val, device, a.batch)
            m |= {"step": step, "train_loss": loss.item(), "uncopyable_seen": uncopyable, "seconds": time.time() - t0}
            print(
                f"step {step}  " + "  ".join(f"{k} {v:.3f}" for k, v in m.items() if isinstance(v, float)), flush=True
            )
            log.write(json.dumps(m) + "\n")
            log.flush()
            torch.save({"model": model.state_dict(), "config": asdict(c), "step": step, "metrics": m}, out / "last.pt")
    log.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--hidden", type=int, default=400)
    ap.add_argument("--train-worlds", type=int, default=200)
    ap.add_argument("--per-category", type=int, default=143, help="examples per training category per world")
    ap.add_argument("--val-per-category", type=int, default=40, help="per category per validation world")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(default_out()))
    run(ap.parse_args())
