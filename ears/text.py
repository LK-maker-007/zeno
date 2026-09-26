from __future__ import annotations

import torch

# Index 0 is the CTC blank. LibriSpeech transcripts use capital letters, the apostrophe and the space.
SYMBOLS = ["<blank>", " ", "'", *"ABCDEFGHIJKLMNOPQRSTUVWXYZ"]
INDEX = {s: i for i, s in enumerate(SYMBOLS)}


def encode(text: str) -> list[int]:
    return [INDEX[ch] for ch in text]


def greedy(logits: torch.Tensor, length: int) -> str:
    # logits: (vocab, T). Best path, then collapse repeats and drop blanks.
    out, prev = [], 0
    for i in logits[:, :length].argmax(0).tolist():
        if i != prev and i != 0:
            out.append(SYMBOLS[i])
        prev = i
    return "".join(out)
