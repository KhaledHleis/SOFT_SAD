"""Generate a tiny synthetic dataset to smoke-test the pipeline.

"speech" segments = AM sine bursts around 200 Hz (vowel-like).
"nonspeech" segments = filtered noise bursts (cough-like, broadband).
"silence" segments = low-amplitude white noise.

The result is 30 wavs of variable length with matching JSON annotations and
the three split CSVs.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def gen_speech(sr: int, dur: float) -> np.ndarray:
    t = np.arange(int(sr * dur)) / sr
    f0 = 180 + 40 * np.sin(2 * np.pi * 3 * t)        # vibrato
    am = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)       # syllable-like AM
    # add a few harmonics
    sig = sum((1.0 / k) * np.sin(2 * np.pi * k * f0 * t) for k in (1, 2, 3, 4))
    return (am * sig * 0.4).astype(np.float32)


def gen_nonspeech(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * dur)
    x = rng.standard_normal(n).astype(np.float32)
    # bandpass-ish via two cascaded EMAs
    a = 0.3
    y = np.zeros_like(x); s = 0.0
    for i in range(n):
        s = a * x[i] + (1 - a) * s
        y[i] = x[i] - s        # high-pass
    # attack-decay envelope
    env = np.linspace(0, 1, n // 5).astype(np.float32)
    env = np.concatenate([env, np.exp(-np.linspace(0, 4, n - env.size)).astype(np.float32)])[:n]
    return (env * y * 0.5).astype(np.float32)


def gen_silence(sr: int, dur: float, rng: np.random.Generator) -> np.ndarray:
    n = int(sr * dur)
    return (rng.standard_normal(n).astype(np.float32) * 0.005)


def make_utt(utt_id: str, sr: int, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Build one utterance with 4-8 alternating intervals."""
    n_intervals = rng.integers(4, 9)
    # always start and end with silence
    pieces = []
    intervals = []
    t = 0.0
    last = "silence"
    for i in range(n_intervals):
        if i == 0 or i == n_intervals - 1 or last in ("speech", "nonspeech"):
            lbl = "silence"
        else:
            lbl = rng.choice(["speech", "nonspeech"])
        dur = float(rng.uniform(0.2, 1.2))
        if lbl == "speech":
            x = gen_speech(sr, dur); cat = "audio_books"
        elif lbl == "nonspeech":
            x = gen_nonspeech(sr, dur, rng); cat = rng.choice(["coughing", "laughing", "yawning"])
        else:
            x = gen_silence(sr, dur, rng); cat = None
        pieces.append(x)
        iv = {"start": round(t, 4), "end": round(t + dur, 4), "label": lbl}
        if cat is not None:
            iv["category"] = cat
        intervals.append(iv)
        t += dur
        last = lbl
    wav = np.concatenate(pieces)
    # Pad to a multiple of n_fft to avoid round-down issues
    pad = (1024 - (len(wav) % 1024)) % 1024
    if pad > 0:
        wav = np.concatenate([wav, gen_silence(sr, pad / sr, rng)])
        intervals[-1]["end"] = round(intervals[-1]["end"] + pad / sr, 4)
    duration = round(len(wav) / sr, 4)
    ann = {
        "utt_id": utt_id,
        "duration": duration,
        "sample_rate": sr,
        "intervals": intervals,
    }
    return wav.astype(np.float32), ann


def write_wav(path: Path, wav: np.ndarray, sr: int):
    x16 = np.clip(wav * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(path, sr, x16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="toy_data")
    ap.add_argument("--n", type=int, default=30)
    ap.add_argument("--sr", type=int, default=16000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    random.seed(args.seed)
    out = Path(args.out)
    (out / "wavs").mkdir(parents=True, exist_ok=True)
    (out / "annotations").mkdir(parents=True, exist_ok=True)

    all_ids = []
    for i in range(args.n):
        utt_id = f"toy_{i:03d}"
        wav, ann = make_utt(utt_id, args.sr, rng)
        write_wav(out / "wavs" / f"{utt_id}.wav", wav, args.sr)
        with open(out / "annotations" / f"{utt_id}.json", "w") as f:
            json.dump(ann, f, indent=2)
        all_ids.append((utt_id, ann["duration"]))

    # split 70/15/15
    random.shuffle(all_ids)
    n_train = int(0.7 * len(all_ids))
    n_val   = int(0.15 * len(all_ids))
    splits = {
        "train": all_ids[:n_train],
        "val":   all_ids[n_train:n_train + n_val],
        "test":  all_ids[n_train + n_val:],
    }
    for split, rows in splits.items():
        with open(out / f"{split}.csv", "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "duration", "source"])
            for uid, dur in rows:
                w.writerow([uid, f"{dur:.4f}", "toy"])
    print(f"  wrote {args.n} utterances to {out}")
    for s, rows in splits.items():
        print(f"    {s}: {len(rows)} utts")


if __name__ == "__main__":
    main()
