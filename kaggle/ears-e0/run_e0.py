from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import soundfile
import torch

# E0 as pre-registered in docs/ears-e0.md: gate G1 (overfit 16 utterances), then QuartzNet-5x5 from scratch on
# train-clean-100 for seeds 0 and 1, scored on dev-clean and test-clean.
INPUT = Path("/kaggle/input")
EPOCHS, MAX_HOURS = 40, 9.0
WORK = Path("/kaggle/working")
SRC = WORK / "src"
# tqdm redraws many times a second. Each job's full stream goes to its log file; the console shows one bar a minute.
BAR_EVERY_S = 60
# A bar ends with its closing bracket; an eval line printed right after a bar shares its line and must still show.
BAR = re.compile(r"(it/s|s/it)[^\]]*\]\s*$")


def mounted(slug: str) -> list[Path]:
    # Kaggle has mounted inputs at /kaggle/input/<slug> and at /kaggle/input/datasets/<owner>/<slug>. Looking only
    # there, instead of walking the tree, keeps 140k audio files from being scanned just to find one folder.
    return sorted(p for p in [INPUT / slug, *INPUT.glob(f"datasets/*/{slug}")] if p.is_dir())


code, mirror = mounted("ava-zeno-code"), mounted("librispeech-clean")
print(f"code {[str(p) for p in code]}  mirror {[str(p) for p in mirror]}", flush=True)
if len(code) != 1 or len(mirror) != 1:
    sys.exit(f"expected one code and one mirror folder; inputs: {sorted(str(p) for p in INPUT.glob('*/*/*'))}")
print(f"commit {(code[0] / 'COMMIT').read_text().strip()}", flush=True)
shutil.copytree(code[0] / "ears", SRC / "ears", dirs_exist_ok=True)
MIRROR = mirror[0]
subprocess.run(["nvidia-smi"], check=False)
n_gpu = torch.cuda.device_count()
print(
    f"python {sys.version.split()[0]}  torch {torch.__version__}  soundfile {soundfile.__version__}  "
    f"GPUs {n_gpu}  CPUs {os.cpu_count()}",
    flush=True,
)
if n_gpu == 0:
    sys.exit("no GPU attached")

common = ["ears.train", "--root", str(MIRROR)]
jobs = [
    ("e0-s0", [*common, "--epochs", str(EPOCHS), "--max-hours", str(MAX_HOURS), "--seed", "0"]),
    ("g1", [*common, "--overfit", "16", "--batch", "16", "--epochs", "500", "--eval-every", "100", "--seed", "0"]),
    ("e0-s1", [*common, "--epochs", str(EPOCHS), "--max-hours", str(MAX_HOURS), "--seed", "1"]),
]


def stream(name: str, proc: subprocess.Popen, log) -> None:
    last_bar = 0.0
    for line in proc.stdout:
        log.write(line)
        log.flush()
        # A finished nested bar clears itself with cursor-up codes, which arrive here as empty lines.
        if not line.replace("\x1b[A", "").strip():
            continue
        bar = bool(BAR.search(line))
        if bar and time.monotonic() - last_bar < BAR_EVERY_S:
            continue
        if bar:
            last_bar = time.monotonic()
        print(f"[{name}] {line.rstrip()}", flush=True)


def launch(name: str, args: list[str], gpu: int) -> tuple[subprocess.Popen, threading.Thread, object]:
    cmd = [sys.executable, "-u", "-m", *args, "--out", str(WORK / name / "ears")]
    print(f"[{name}] gpu {gpu}: {' '.join(cmd)}", flush=True)
    log = (WORK / f"{name}.log").open("w")
    env = os.environ | {"CUDA_VISIBLE_DEVICES": str(gpu)}
    p = subprocess.Popen(cmd, cwd=SRC, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    t = threading.Thread(target=stream, args=(name, p, log))
    t.start()
    return p, t, log


# One job per GPU: e0-s0 on one, g1 then e0-s1 on the other.
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
print(f"exit codes {codes}", flush=True)
sys.exit(1 if any(codes.values()) else 0)
