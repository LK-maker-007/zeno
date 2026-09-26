from __future__ import annotations

import argparse
import json
import math
import platform
import random
import time
from dataclasses import asdict
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

from ears.data import Utt, batches, cache, index, pad
from ears.features import HOP, SAMPLE_RATE, mask
from ears.model import Config, QuartzNet
from ears.score import error_rate, speaker_bootstrap
from ears.text import greedy


def default_out() -> Path:
    kaggle = Path("/kaggle/working")
    return kaggle / "ears" if kaggle.exists() else Path(__file__).resolve().parent.parent / "runs" / "ears"


def hours(data: list[tuple[torch.Tensor, torch.Tensor]]) -> float:
    return sum(f.shape[1] for f, _ in data) * HOP / SAMPLE_RATE / 3600


@torch.no_grad()
def transcribe(model: QuartzNet, data: list, device: str, bs: int) -> list[str]:
    model.eval()
    hyps: list[str] = [""] * len(data)
    order = sorted(range(len(data)), key=lambda i: data[i][0].shape[1])
    for k in range(0, len(order), bs):
        idx = order[k : k + bs]
        x, xl, _, _ = pad([data[i] for i in idx], device)
        with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
            logits, ol = model(x, xl)
        for j, i in enumerate(idx):
            hyps[i] = greedy(logits[j].float(), int(ol[j]))
    model.train()
    return hyps


def evaluate(model: QuartzNet, data: list, utts: list[Utt], device: str, bs: int, ci: bool = False) -> dict:
    hyps = transcribe(model, data, device, bs)
    refs = [u.text for u in utts]
    we, wn = error_rate(refs, hyps, "word")
    ce, cn = error_rate(refs, hyps, "char")
    out = {"wer": we / wn, "cer": ce / cn, "words": wn}
    if ci:
        per = [error_rate([r], [h], "word") for r, h in zip(refs, hyps, strict=True)]
        wer, lo, hi = speaker_bootstrap([e for e, _ in per], [n for _, n in per], [u.speaker for u in utts])
        out |= {"wer": wer, "wer_ci95": [lo, hi], "speakers": len({u.speaker for u in utts})}
        out["samples"] = [{"ref": r, "hyp": h} for r, h in list(zip(refs, hyps, strict=True))[:8]]
    return out


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
    root = Path(a.root)
    if a.overfit:
        train_utts = index(root, [a.train[0]])[: a.overfit]
        sets = {"dev": train_utts}
    else:
        train_utts = index(root, a.train)
        sets = {"dev": index(root, [a.dev]), "test": index(root, [a.test])}
    print(f"index: train {len(train_utts)} utterances, {len({u.speaker for u in train_utts})} speakers", flush=True)
    for name, utts in sets.items():
        shared = {u.speaker for u in utts} & {u.speaker for u in train_utts}
        print(
            f"index: {name} {len(utts)} utterances, {len({u.speaker for u in utts})} speakers, "
            f"{len(shared)} of them also in train",
            flush=True,
        )
    if not a.overfit and any({u.speaker for u in utts} & {u.speaker for u in train_utts} for utts in sets.values()):
        raise SystemExit("an evaluation speaker also appears in training data; scores would not measure new voices")
    if not train_utts or not all(sets.values()):
        raise SystemExit(f"an empty split under {root}: the mirror lacks a requested subset")
    t0 = time.time()
    train = cache(train_utts, a.workers, "features train")
    data = {name: cache(utts, a.workers, f"features {name}") for name, utts in sets.items()}
    print(f"features in {time.time() - t0:.0f} s: train {hours(train):.2f} h", flush=True)
    for name, d in data.items():
        print(f"features: {name} {hours(d):.2f} h", flush=True)
    f0, y0 = train[0]
    print(f"first utterance: features {tuple(f0.shape)} {f0.dtype}, {len(y0)} labels, text {train_utts[0].text!r}")
    too_short = sum(math.ceil(f.shape[1] / 2) < len(y) for f, y in train)
    print(f"utterances with fewer output frames than labels (CTC cannot fit them): {too_short}", flush=True)

    c = Config(dropout=a.dropout)
    model = QuartzNet(c).to(device)
    print(f"model {asdict(c)}  params {model.n_params():,}", flush=True)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.weight_decay)
    per_epoch = math.ceil(len(train) / a.batch)
    total = per_epoch * a.epochs
    warm = min(a.warmup, total // 10 + 1)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total))
    )
    scaler = torch.amp.GradScaler("cuda", enabled=device == "cuda")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"model": asdict(c), "args": vars(a) | {"out": str(out)}}, indent=2))
    log = (out / "metrics.jsonl").open("a")
    rng, gen = random.Random(a.seed), torch.Generator().manual_seed(a.seed)
    lengths = [f.shape[1] for f, _ in train]
    step, t_start = 0, time.time()
    for epoch in range(1, a.epochs + 1):
        t_epoch, losses = time.time(), []
        for b in tqdm(batches(lengths, a.batch, rng), desc=f"epoch {epoch}", mininterval=60):
            x, xl, y, yl = pad([train[i] for i in b], device)
            if not a.overfit:
                mask(x, xl, gen)
            with torch.autocast(device, dtype=torch.float16, enabled=device == "cuda"):
                logits, ol = model(x, xl)
            logp = logits.float().log_softmax(1).permute(2, 0, 1)
            loss = F.ctc_loss(logp, y, ol, yl, blank=0, zero_infinity=True)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            step += 1
            losses.append(loss.item())
        elapsed = time.time() - t_start
        last = epoch == a.epochs or elapsed / 3600 + (time.time() - t_epoch) / 3600 > a.max_hours
        m = {
            "epoch": epoch,
            "step": step,
            "train_loss": sum(losses) / len(losses),
            "epoch_seconds": time.time() - t_epoch,
            "hours_elapsed": elapsed / 3600,
        }
        if epoch == 1:
            print(f"projection: {a.epochs} epochs take {m['epoch_seconds'] * a.epochs / 3600:.1f} h", flush=True)
        if epoch % a.eval_every == 0 or last:
            m |= {f"dev_{k}": v for k, v in evaluate(model, data["dev"], sets["dev"], device, a.batch).items()}
            torch.save(
                {"model": model.state_dict(), "config": asdict(c), "epoch": epoch, "metrics": m}, out / "last.pt"
            )
        print("  ".join(f"{k} {v:.4f}" if isinstance(v, float) else f"{k} {v}" for k, v in m.items()), flush=True)
        log.write(json.dumps(m) + "\n")
        log.flush()
        if last:
            if epoch < a.epochs:
                print(f"stopped after epoch {epoch}: the next would pass --max-hours {a.max_hours}", flush=True)
            break
    log.close()
    final = {name: evaluate(model, data[name], sets[name], device, a.batch, ci=True) for name in sets}
    for name, r in final.items():
        print(f"\n{name}: WER {r['wer']:.4f} 95% CI {r['wer_ci95']} (speaker bootstrap)  CER {r['cer']:.4f}")
        for s in r["samples"]:
            print(f"  ref: {s['ref']}\n  hyp: {s['hyp']}")
    (out / "final.json").write_text(json.dumps(final, indent=2))
    print(f"wrote {out / 'final.json'}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="a LibriSpeech mirror; every *.trans.txt under it is indexed")
    ap.add_argument("--train", nargs="+", default=["train-clean-100"])
    ap.add_argument("--dev", default="dev-clean")
    ap.add_argument("--test", default="test-clean")
    ap.add_argument("--overfit", type=int, default=0, help="train and evaluate on this many utterances (sanity check)")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--max-hours", type=float, default=10.0, help="stop early rather than lose the run to the limit")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-3)
    ap.add_argument("--warmup", type=int, default=1000)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--eval-every", type=int, default=5)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(default_out()))
    run(ap.parse_args())
