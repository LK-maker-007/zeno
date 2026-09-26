from __future__ import annotations

import collections
import sys
import time
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

# Checks a LibriSpeech mirror against the corpus's own files before any training depends on it:
# every transcript line must have its audio, and measured hours per subset must match CHAPTERS.TXT.
SUBSETS = [
    "dev-clean",
    "dev-other",
    "test-clean",
    "test-other",
    "train-clean-100",
    "train-clean-360",
    "train-other-500",
]

print(f"python {sys.version.split()[0]}  soundfile {sf.__version__}", flush=True)
for ds in sorted(Path("/kaggle/input").iterdir()):
    t0 = time.time()
    print(f"\n=== {ds.name}", flush=True)
    trans = sorted(ds.rglob("*.trans.txt"))
    by_subset = collections.defaultdict(list)
    for t in trans:
        subset = next((p for p in t.parts if p in SUBSETS), "?")
        by_subset[subset].append(t)
    print(f"transcript files {len(trans)}: {dict(sorted((k, len(v)) for k, v in by_subset.items()))}", flush=True)
    measured = {}
    for subset, files in sorted(by_subset.items()):
        n = missing = 0
        seconds, rates, chars = 0.0, collections.Counter(), collections.Counter()
        speakers = set()
        for t in tqdm(files, desc=f"  {subset}", mininterval=30):
            for line in t.read_text().splitlines():
                utt, text = line.split(" ", 1)
                n += 1
                speakers.add(utt.split("-")[0])
                chars.update(text)
                audio = t.parent / f"{utt}.flac"
                if not audio.exists():
                    missing += 1
                    continue
                info = sf.info(str(audio))
                seconds += info.duration
                rates[info.samplerate] += 1
        measured[subset] = seconds / 3600
        print(
            f"  {subset}: utterances {n}  missing audio {missing}  speakers {len(speakers)}  "
            f"hours {seconds / 3600:.2f}  sample rates {dict(rates)}",
            flush=True,
        )
        print(f"    characters in transcripts: {''.join(sorted(chars))!r}", flush=True)
    for chapters in sorted(ds.rglob("CHAPTERS.TXT"))[:1]:
        minutes = collections.Counter()
        for line in chapters.read_text().splitlines():
            if line.startswith(";"):
                continue
            cols = [c.strip() for c in line.split("|")]
            minutes[cols[3]] += float(cols[2])
        print(f"  CHAPTERS.TXT ({chapters.relative_to(ds)}): hours per subset vs measured", flush=True)
        for subset in SUBSETS:
            if subset in minutes or subset in measured:
                listed = minutes.get(subset, 0.0) / 60
                got = measured.get(subset, 0.0)
                print(
                    f"    {subset:<16} listed {listed:8.2f}  measured {got:8.2f}  diff {got - listed:+8.2f}",
                    flush=True,
                )
    print(f"  probe took {time.time() - t0:.0f} s", flush=True)
