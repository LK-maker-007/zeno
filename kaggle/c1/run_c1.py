from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

WORK = Path("/kaggle/working/src")

# Kaggle extracts uploaded archives and has moved the mount point before, so locate the code by a known file.
found = sorted(Path("/kaggle/input").rglob("childmind/train.py"))
print(f"code found: {[str(p) for p in found]}", flush=True)
if len(found) != 1:
    sys.exit(f"expected exactly one childmind/train.py under /kaggle/input, found {len(found)}")
root = found[0].parent.parent
for pkg in ("childmind", "scoreboard"):
    shutil.copytree(root / pkg, WORK / pkg, dirs_exist_ok=True)
print(f"copied to {WORK}: {sorted(str(p.relative_to(WORK)) for p in WORK.rglob('*.py'))}", flush=True)
if shutil.which("nvidia-smi"):
    subprocess.run(["nvidia-smi"], check=False)
else:
    print("nvidia-smi not found: no GPU attached", flush=True)
cmd = [sys.executable, "-u", "-m", "childmind.train", *sys.argv[1:]]
print("running: " + " ".join(cmd), flush=True)
sys.exit(subprocess.run(cmd, cwd=WORK, check=False).returncode)
