"""Annotations → per-frame labels, and PyTorch Dataset wrappers.

The job of this module is to take an utterance-level annotation file
(`annotations/<utt_id>.json`) and produce, for each frame index `i`, the
ground-truth label code in {SILENCE, SPEECH, NONSPEECH}. We also expose the
per-frame "category" string (e.g. "coughing") for the test-set histogram.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from scipy.io import wavfile
from torch.utils.data import Dataset

from . import NAME_TO_LABEL, SILENCE


# ---------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------

@dataclass
class Annotation:
    """A parsed annotation file."""
    utt_id: str
    duration: float            # seconds
    sample_rate: int
    intervals: list            # list of dicts with keys: start, end, label, category

    @classmethod
    def from_json(cls, path: Path) -> "Annotation":
        with open(path, "r") as f:
            data = json.load(f)
        intervals = data["intervals"]
        # sanity: intervals must cover [0, duration) exactly, no gaps, no overlaps
        intervals = sorted(intervals, key=lambda x: x["start"])
        prev_end = 0.0
        for iv in intervals:
            if not np.isclose(iv["start"], prev_end, atol=1e-3):
                raise ValueError(
                    f"{path}: gap or overlap at t={iv['start']:.3f}s "
                    f"(expected {prev_end:.3f}s)"
                )
            if iv["end"] <= iv["start"]:
                raise ValueError(f"{path}: empty interval {iv}")
            if iv["label"] not in NAME_TO_LABEL:
                raise ValueError(
                    f"{path}: unknown label '{iv['label']}'. "
                    f"Allowed: {list(NAME_TO_LABEL)}"
                )
            prev_end = iv["end"]
        if not np.isclose(prev_end, data["duration"], atol=1e-3):
            raise ValueError(
                f"{path}: intervals end at {prev_end:.3f}s, "
                f"but duration is {data['duration']:.3f}s"
            )
        return cls(
            utt_id=data["utt_id"],
            duration=float(data["duration"]),
            sample_rate=int(data.get("sample_rate", 16000)),
            intervals=intervals,
        )


def frame_labels_from_annotation(
    ann: Annotation,
    n_frames: int,
    hop_length: int,
    sample_rate: int,
    n_fft: int,
) -> tuple[np.ndarray, list[Optional[str]]]:
    """Return (labels[n_frames], categories[n_frames]).

    Frame `i` corresponds to the time-centre of analysis window `i`:
        t_i = (i * hop_length + n_fft / 2) / sample_rate
    The frame is labelled by the interval that contains `t_i`. Intervals
    are half-open `[start, end)`, with the very last interval treated as
    closed on the right so the final frame always gets a label.
    """
    labels = np.full(n_frames, SILENCE, dtype=np.int64)
    categories: list[Optional[str]] = [None] * n_frames

    centres_sec = (np.arange(n_frames) * hop_length + n_fft / 2.0) / sample_rate

    for iv in ann.intervals:
        lbl = NAME_TO_LABEL[iv["label"]]
        cat = iv.get("category")
        s, e = iv["start"], iv["end"]
        mask = (centres_sec >= s) & (centres_sec < e)
        labels[mask] = lbl
        for idx in np.where(mask)[0]:
            categories[idx] = cat

    # In case rounding put the very last frame past the last interval's end,
    # snap it to the final interval's label.
    last = ann.intervals[-1]
    if centres_sec[-1] >= last["end"]:
        labels[-1] = NAME_TO_LABEL[last["label"]]
        categories[-1] = last.get("category")

    return labels, categories


# ---------------------------------------------------------------------
# Audio I/O
# ---------------------------------------------------------------------

def load_wav(path: Path, expected_sr: int) -> np.ndarray:
    """Read a 16-bit mono wav, return float32 in [-1, 1]."""
    sr, x = wavfile.read(path)
    if sr != expected_sr:
        raise ValueError(f"{path}: sample rate {sr} != expected {expected_sr}")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if x.dtype == np.int16:
        x = x.astype(np.float32) / 32768.0
    elif x.dtype == np.int32:
        x = x.astype(np.float32) / 2147483648.0
    elif x.dtype == np.float32:
        pass
    else:
        x = x.astype(np.float32)
    return x


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------

class FeatureDataset(Dataset):
    """Reads cached *.npz files produced by `prepare.py`.

    Each .npz contains:
        feat        : float32 (T, n_mfcc)
        labels      : int64   (T,)
        categories  : object array of length T, or None
        utt_id      : str
    """

    def __init__(self, feature_dir: Path, csv_path: Path):
        import csv

        self.feature_dir = Path(feature_dir)
        self.utt_ids: list[str] = []
        with open(csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                self.utt_ids.append(row["utt_id"])

    def __len__(self) -> int:
        return len(self.utt_ids)

    def __getitem__(self, idx: int):
        utt_id = self.utt_ids[idx]
        path = self.feature_dir / f"{utt_id}.npz"
        data = np.load(path, allow_pickle=True)
        feat = torch.from_numpy(data["feat"].astype(np.float32))
        labels = torch.from_numpy(data["labels"].astype(np.int64))
        return {
            "utt_id": utt_id,
            "feat": feat,           # (T, F)
            "labels": labels,       # (T,)
            "categories": data["categories"],  # object array
        }


def collate_pad(batch):
    """Pad a list of variable-length utterances into a batch.

    Returns:
        feat   : (B, T_max, F) float32
        labels : (B, T_max)    int64
        lengths: (B,)          int64    (true T per item, for masking loss)
        utt_ids: list[str]
        categories_list: list[np.ndarray]
    """
    B = len(batch)
    F = batch[0]["feat"].shape[1]
    lengths = torch.tensor([b["feat"].shape[0] for b in batch], dtype=torch.long)
    T_max = int(lengths.max().item())

    feat = torch.zeros(B, T_max, F, dtype=torch.float32)
    labels = torch.full((B, T_max), -1, dtype=torch.long)  # -1 = padding/ignore
    for i, b in enumerate(batch):
        T = b["feat"].shape[0]
        feat[i, :T] = b["feat"]
        labels[i, :T] = b["labels"]

    return {
        "feat": feat,
        "labels": labels,
        "lengths": lengths,
        "utt_ids": [b["utt_id"] for b in batch],
        "categories_list": [b["categories"] for b in batch],
    }
