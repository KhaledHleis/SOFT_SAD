"""MFCC feature extraction matching the paper's spec.

Defaults: n_fft=1024, hop=512, n_mels=13, n_mfcc=13 at 16 kHz, i.e. 32 ms
analysis frames every 16 ms. Per-utterance cepstral mean-variance
normalisation is applied so the GRU sees zero-mean unit-variance features
regardless of recording level.
"""
from __future__ import annotations

import numpy as np
import torch
import torchaudio


def make_mfcc_extractor(cfg: dict, sample_rate: int) -> torchaudio.transforms.MFCC:
    """Build the torchaudio MFCC transform configured from `cfg`."""
    n_fft = int(cfg["n_fft"])
    hop = int(cfg["hop_length"])
    n_mels = int(cfg["n_mels"])
    n_mfcc = int(cfg["n_mfcc"])
    fmin = float(cfg.get("fmin", 0))
    fmax = float(cfg.get("fmax", sample_rate / 2))

    return torchaudio.transforms.MFCC(
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        melkwargs=dict(
            n_fft=n_fft,
            hop_length=hop,
            n_mels=n_mels,
            f_min=fmin,
            f_max=fmax,
            center=False,        # so frame i = window starting at i*hop
            power=2.0,
        ),
    )


def expected_num_frames(num_samples: int, n_fft: int, hop: int) -> int:
    """Number of frames torchaudio.MFCC produces with center=False."""
    if num_samples < n_fft:
        return 0
    return 1 + (num_samples - n_fft) // hop


def extract_mfcc(wav: np.ndarray, mfcc: torchaudio.transforms.MFCC) -> np.ndarray:
    """Return float32 features of shape (T, n_mfcc)."""
    x = torch.from_numpy(np.asarray(wav, dtype=np.float32)).unsqueeze(0)  # (1, L)
    with torch.no_grad():
        y = mfcc(x)                # (1, n_mfcc, T)
    y = y.squeeze(0).transpose(0, 1).contiguous().numpy()  # (T, n_mfcc)
    return y.astype(np.float32)


def cmvn(feat: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Per-utterance cepstral mean-variance normalisation."""
    mu = feat.mean(axis=0, keepdims=True)
    sd = feat.std(axis=0, keepdims=True)
    return (feat - mu) / (sd + eps)
