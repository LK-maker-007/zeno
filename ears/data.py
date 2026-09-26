from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from ears.features import SAMPLE_RATE, log_mel, mel_filterbank
from ears.text import encode


@dataclass(frozen=True)
class Utt:
    audio: Path
    text: str
    speaker: str
    subset: str


def index(root: Path, subsets: list[str]) -> list[Utt]:
    # Mirrors nest a subset at different depths (LibriSpeech/<subset>/..., <subset>/LibriSpeech/<subset>/...).
    # Globbing only those depths avoids walking every audio file: that walk took 21 minutes on Kaggle.
    out = []
    for subset in subsets:
        found: set[Path] = set()
        for depth in ("", "*/", "*/*/"):
            found.update(root.glob(f"{depth}{subset}/*/*/*.trans.txt"))
        for t in sorted(found):
            for line in t.read_text().splitlines():
                utt, text = line.split(" ", 1)
                out.append(Utt(t.parent / f"{utt}.flac", text, utt.split("-")[0], subset))
    return out


class _Features(Dataset):
    def __init__(self, utts: list[Utt]):
        self.utts, self.fb = utts, mel_filterbank()

    def __len__(self) -> int:
        return len(self.utts)

    def __getitem__(self, i: int) -> tuple[torch.Tensor, torch.Tensor]:
        import soundfile as sf

        wave, sr = sf.read(str(self.utts[i].audio), dtype="float32")
        if sr != SAMPLE_RATE:
            raise ValueError(f"{self.utts[i].audio}: {sr} Hz, expected {SAMPLE_RATE}")
        return log_mel(torch.from_numpy(wave), self.fb).half(), torch.tensor(encode(self.utts[i].text))


def _one_thread(_: int) -> None:
    # Each worker computes one utterance at a time; intra-op threads on top of the workers would oversubscribe.
    torch.set_num_threads(1)


def cache(utts: list[Utt], workers: int, desc: str) -> list[tuple[torch.Tensor, torch.Tensor]]:
    # Features for every utterance, held in RAM as fp16: 100 h of 64-bin frames is about 4.6 GB.
    loader = DataLoader(_Features(utts), batch_size=None, num_workers=workers, worker_init_fn=_one_thread)
    return list(tqdm(loader, total=len(utts), desc=desc, mininterval=1))


def batches(lengths: list[int], size: int, rng: random.Random) -> list[list[int]]:
    # Similar lengths share a batch so little of it is padding; the jitter and shuffle change the batches every epoch.
    order = sorted(range(len(lengths)), key=lambda i: lengths[i] * (1 + 0.1 * rng.random()))
    chunks = [order[i : i + size] for i in range(0, len(order), size)]
    rng.shuffle(chunks)
    return chunks


def pad(items: list[tuple[torch.Tensor, torch.Tensor]], device: str) -> tuple[torch.Tensor, ...]:
    lengths = torch.tensor([f.shape[1] for f, _ in items])
    x = torch.zeros(len(items), items[0][0].shape[0], int(lengths.max()))
    for i, (f, _) in enumerate(items):
        x[i, :, : f.shape[1]] = f.float()
    y = torch.cat([t for _, t in items])
    y_lengths = torch.tensor([len(t) for _, t in items])
    return x.to(device), lengths.to(device), y.to(device), y_lengths.to(device)
