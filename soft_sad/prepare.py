"""Build cached MFCC feature files from wavs + annotations.

Usage:
    python -m soft_sad.prepare --config config.yaml
    python -m soft_sad.prepare --config config.yaml --check-only
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import yaml
from tqdm import tqdm

from soft_sad.data import Annotation, frame_labels_from_annotation, load_wav
from soft_sad.features import cmvn, expected_num_frames, extract_mfcc, make_mfcc_extractor


def load_split(csv_path: Path) -> list[str]:
    """Read utt_ids from a split CSV."""
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            out.append(row["utt_id"])
    return out


def check_utt(
    utt_id: str,
    wav_dir: Path,
    ann_dir: Path,
    sample_rate: int,
) -> tuple[bool, str]:
    wav_path = wav_dir / f"{utt_id}.wav"
    ann_path = ann_dir / f"{utt_id}.json"
    if not wav_path.exists():
        return False, f"missing wav: {wav_path}"
    if not ann_path.exists():
        return False, f"missing annotation: {ann_path}"
    try:
        wav = load_wav(wav_path, sample_rate)
    except Exception as e:
        return False, f"bad wav {wav_path}: {e}"
    try:
        ann = Annotation.from_json(ann_path)
    except Exception as e:
        return False, f"bad annotation {ann_path}: {e}"
    expected_len = ann.duration * sample_rate
    if abs(len(wav) - expected_len) > sample_rate * 0.05:   # 50 ms tolerance
        return False, (
            f"{utt_id}: wav has {len(wav)} samples, "
            f"annotation says {expected_len:.0f}"
        )
    return True, "ok"


def process_one(
    utt_id: str,
    wav_dir: Path, ann_dir: Path,
    out_dir: Path,
    mfcc,
    sample_rate: int, n_fft: int, hop_length: int,
):
    wav = load_wav(wav_dir / f"{utt_id}.wav", sample_rate)
    ann = Annotation.from_json(ann_dir / f"{utt_id}.json")
    feat = extract_mfcc(wav, mfcc)                  # (T, F)
    feat = cmvn(feat)
    T = feat.shape[0]
    T_check = expected_num_frames(len(wav), n_fft, hop_length)
    if T != T_check:
        # torchaudio rounding can produce off-by-one; trim to the lower.
        T = min(T, T_check)
        feat = feat[:T]
    labels, categories = frame_labels_from_annotation(
        ann, n_frames=T,
        hop_length=hop_length, sample_rate=sample_rate, n_fft=n_fft,
    )
    out_path = out_dir / f"{utt_id}.npz"
    np.savez_compressed(
        out_path,
        feat=feat,
        labels=labels.astype(np.int64),
        categories=np.array(categories, dtype=object),
        utt_id=utt_id,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--check-only", action="store_true",
                    help="walk all files and report problems, write nothing.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_root = Path(cfg["data"]["root"])
    sample_rate = int(cfg["data"]["sample_rate"])
    wav_dir = data_root / "wavs"
    ann_dir = data_root / "annotations"
    feat_dir = Path(cfg["data"]["feature_cache"])
    feat_dir.mkdir(parents=True, exist_ok=True)

    splits = {}
    for name in ("train", "val", "test"):
        csv_path = data_root / f"{name}.csv"
        if csv_path.exists():
            splits[name] = load_split(csv_path)
        else:
            print(f"[warn] {csv_path} not found, skipping {name} split")

    # Sanity-check phase
    print(">>> checking files...")
    all_ok = True
    n_class = {"silence": 0.0, "speech": 0.0, "nonspeech": 0.0}
    for split, utts in splits.items():
        for utt_id in utts:
            ok, msg = check_utt(utt_id, wav_dir, ann_dir, sample_rate)
            if not ok:
                print(f"  [{split}] FAIL {utt_id}: {msg}")
                all_ok = False
                continue
            ann = Annotation.from_json(ann_dir / f"{utt_id}.json")
            for iv in ann.intervals:
                n_class[iv["label"]] += iv["end"] - iv["start"]
    if not all_ok:
        print(">>> aborting: fix the failures above and retry.")
        sys.exit(2)
    total = sum(n_class.values())
    print(">>> file check OK. Class durations (seconds):")
    for k, v in n_class.items():
        print(f"     {k:>10}: {v:8.1f}  ({100*v/max(total,1e-6):.1f}%)")
    if args.check_only:
        return

    # Feature extraction
    mfcc = make_mfcc_extractor(cfg["features"], sample_rate)
    n_fft = int(cfg["features"]["n_fft"])
    hop_length = int(cfg["features"]["hop_length"])

    for split, utts in splits.items():
        print(f">>> extracting features for {split} ({len(utts)} utts)...")
        for utt_id in tqdm(utts):
            try:
                process_one(
                    utt_id, wav_dir, ann_dir, feat_dir, mfcc,
                    sample_rate, n_fft, hop_length,
                )
            except Exception as e:
                print(f"  [{split}] {utt_id}: {e}")
    print(">>> done.")


if __name__ == "__main__":
    main()
