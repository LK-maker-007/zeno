from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import torch

# Smoke run: one seed, validation worlds. The pre-registered race (docs/a1-designs.md) is 5 seeds on the test worlds.
SEEDS = [0]
WORLDS = "val"
WORK = Path("/kaggle/working")
SRC = WORK / "src"
# tqdm redraws many times a second. Each job's full stream goes to its log file; the console shows one bar a minute.
BAR_EVERY_S = 60

found = sorted(Path("/kaggle/input").rglob("COMMIT"))
print(f"code found: {[str(p) for p in found]}", flush=True)
if len(found) != 1:
    sys.exit(f"expected exactly one COMMIT file under /kaggle/input, found {len(found)}")
root = found[0].parent
print(f"commit {found[0].read_text().strip()}", flush=True)
for pkg in ("scoreboard", "childmind", "mind"):
    shutil.copytree(root / pkg, SRC / pkg, dirs_exist_ok=True)
subprocess.run(["nvidia-smi"], check=False)
n_gpu = torch.cuda.device_count()
print(f"python {sys.version.split()[0]}  torch {torch.__version__}  GPUs {n_gpu}", flush=True)
if n_gpu == 0:
    sys.exit("no GPU attached")

jobs = []
for s in SEEDS:
    c1 = ["childmind.train", "--task", "reading", "--seed", str(s), "--out", str(WORK / f"c1v2-s{s}" / "childmind")]
    jobs += [
        (f"c1v2-s{s}", c1),
        (f"a1-s{s}", ["mind.train", "--seed", str(s), "--out", str(WORK / f"a1-s{s}" / "mind")]),
    ]


def stream(name: str, proc: subprocess.Popen, log) -> None:
    last_bar = 0.0
    for line in proc.stdout:
        log.write(line)
        log.flush()
        # A finished nested bar clears itself with cursor-up codes, which arrive here as empty lines.
        if not line.replace("\x1b[A", "").strip():
            continue
        bar = "it/s]" in line or "s/it]" in line
        if bar and time.monotonic() - last_bar < BAR_EVERY_S:
            continue
        if bar:
            last_bar = time.monotonic()
        print(f"[{name}] {line.rstrip()}", flush=True)


def launch(name: str, args: list[str], gpu: int) -> tuple[subprocess.Popen, threading.Thread, object]:
    cmd = [sys.executable, "-u", "-m", *args]
    print(f"[{name}] gpu {gpu}: {' '.join(cmd)}", flush=True)
    log = (WORK / f"{name}.log").open("w")
    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
    p = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t = threading.Thread(target=stream, args=(name, p, log))
    t.start()
    return p, t, log


# One job per GPU (plan.md §6: run one seed per GPU rather than splitting a small model across both).
pending, running, codes = list(jobs), {}, {}
while pending or running:
    for gpu in range(n_gpu):
        if gpu not in running and pending:
            name, args = pending.pop(0)
            running[gpu] = (name, *launch(name, args, gpu))
    time.sleep(5)
    for gpu, (name, p, t, log) in list(running.items()):
        if p.poll() is not None:
            t.join()
            log.close()
            codes[name] = p.returncode
            print(f"[{name}] exited {p.returncode}", flush=True)
            del running[gpu]
print(f"training exit codes {codes}", flush=True)
if any(codes.values()):
    sys.exit("a training job failed, so the race was not run")

race = [sys.executable, "-u", "-m", "scoreboard.read_race", "--worlds", WORLDS, "--n", "100", "--device", "cuda"]
race += ["--out", str(WORK / "results"), "--childmind"]
race += [str(WORK / f"c1v2-s{s}" / "childmind" / "last.pt") for s in SEEDS]
race += ["--mind", *[str(WORK / f"a1-s{s}" / "mind" / "last.pt") for s in SEEDS]]
print("running: " + " ".join(race), flush=True)
sys.exit(subprocess.run(race, cwd=SRC, check=False).returncode)
