from __future__ import annotations

import math

import torch

SAMPLE_RATE = 16000
N_FFT, WIN, HOP = 512, 400, 160
N_MELS = 64


def mel_filterbank(n_mels: int = N_MELS, n_fft: int = N_FFT, sr: int = SAMPLE_RATE) -> torch.Tensor:
    # HTK mel scale, triangular filters spanning 0 Hz to Nyquist. Shape (n_mels, n_fft // 2 + 1).
    top = 2595.0 * math.log10(1.0 + (sr / 2) / 700.0)
    hz = 700.0 * (10 ** (torch.linspace(0.0, top, n_mels + 2) / 2595.0) - 1.0)
    bins = torch.linspace(0.0, sr / 2, n_fft // 2 + 1)
    lo, mid, hi = hz[:-2, None], hz[1:-1, None], hz[2:, None]
    return torch.clamp(torch.minimum((bins - lo) / (mid - lo), (hi - bins) / (hi - mid)), min=0.0)


def log_mel(wave: torch.Tensor, fb: torch.Tensor) -> torch.Tensor:
    # wave: (samples,) in [-1, 1]. Returns (n_mels, frames) with each channel normalised over the utterance.
    window = torch.hann_window(WIN, device=wave.device)
    spec = torch.stft(wave, N_FFT, HOP, WIN, window=window, center=True, return_complex=True)
    x = torch.log(fb.to(wave.device) @ spec.abs().pow(2) + 1e-6)
    return (x - x.mean(1, keepdim=True)) / (x.std(1, keepdim=True) + 1e-5)


def frames(n_samples: int) -> int:
    # torch.stft with center=True emits one frame per hop, plus one.
    return n_samples // HOP + 1


def mask(x: torch.Tensor, lengths: torch.Tensor, rng: torch.Generator, n_freq=2, freq=15, n_time=2, time=50):
    # SpecAugment-style masking of a padded batch (B, n_mels, T), in place; masks never reach into the padding.
    b, n_mels, _ = x.shape
    for i in range(b):
        n = int(lengths[i])
        for _ in range(n_freq):
            w = int(torch.randint(0, freq + 1, (1,), generator=rng))
            f0 = int(torch.randint(0, max(1, n_mels - w), (1,), generator=rng))
            x[i, f0 : f0 + w, :n] = 0.0
        for _ in range(n_time):
            w = min(int(torch.randint(0, time + 1, (1,), generator=rng)), n // 5)
            t0 = int(torch.randint(0, max(1, n - w), (1,), generator=rng))
            x[i, :, t0 : t0 + w] = 0.0
    return x
