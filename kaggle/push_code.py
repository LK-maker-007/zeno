from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATASET = "singarajb/ava-zeno-code"
PATHS = ["scoreboard", "childmind", "mind", "arena", "ears", "pyproject.toml", "LICENSE"]


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, check=True, capture_output=True, text=True).stdout.strip()


def run(a: argparse.Namespace) -> None:
    commit = git("rev-parse", "HEAD")
    git("fetch", "origin", "main")
    # Every Kaggle run must trace to merged code, so an unmerged commit is refused.
    if subprocess.run(["git", "merge-base", "--is-ancestor", commit, "origin/main"], cwd=ROOT).returncode:
        sys.exit(f"HEAD {commit[:7]} is not on origin/main: merge it first")
    out = ROOT / "runs" / "kaggle-code"
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    # git archive packs the commit, never the working tree, so uncommitted edits cannot leak into a run.
    tar = subprocess.run(["git", "archive", "--format=tar", commit, *PATHS], cwd=ROOT, check=True, capture_output=True)
    subprocess.run(["tar", "-x", "-C", str(out)], input=tar.stdout, check=True)
    (out / "COMMIT").write_text(commit + "\n")
    meta = {"title": "ava-zeno-code", "id": DATASET, "licenses": [{"name": "other"}], "description": "MIT licence"}
    (out / "dataset-metadata.json").write_text(json.dumps(meta, indent=2))
    files = sorted(str(p.relative_to(out)) for p in out.rglob("*") if p.is_file())
    print(f"commit {commit}: {len(files)} files\n" + "\n".join(f"  {f}" for f in files), flush=True)
    if a.dry_run:
        return
    exists = subprocess.run(["kaggle", "datasets", "status", DATASET], capture_output=True).returncode == 0
    if exists:
        cmd = ["kaggle", "datasets", "version", "-p", str(out), "-m", f"commit {commit}", "--dir-mode", "zip"]
    else:
        cmd = ["kaggle", "datasets", "create", "-p", str(out), "--dir-mode", "zip"]
    print("running: " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=f"Upload the merged code at HEAD to the Kaggle dataset {DATASET}")
    ap.add_argument("--dry-run", action="store_true", help="build the upload folder and list it, upload nothing")
    run(ap.parse_args())
