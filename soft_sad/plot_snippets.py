"""
plot_snippets.py
----------------
Generates annotated snippet images (waveform + spectrogram + annotation ribbon)
for a random selection of samples from each label category, based on the
soft-SAD data format and config.yaml.

Usage:
    python plot_snippets.py --config config.yaml [--n 5] [--out snippets]
"""

import argparse
import json
import os
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator
import numpy as np
import yaml
import soundfile as sf
import librosa
import librosa.display

# ── Colour palette for labels ────────────────────────────────────────────────
LABEL_COLORS = {
    "speech":     "#4CAF50",   # green
    "nonspeech":  "#FF9800",   # orange
    "silence":    "#78909C",   # blue-grey
}
LABEL_ALPHA = 0.35


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_split_csv(csv_path: Path) -> list[str]:
    """Return list of utt_ids from a split CSV (skips header)."""
    if not csv_path.exists():
        return []
    utt_ids = []
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue  # header
            parts = line.strip().split(",")
            if parts and parts[0]:
                utt_ids.append(parts[0])
    return utt_ids


def load_annotation(ann_path: Path) -> dict:
    with open(ann_path) as f:
        return json.load(f)


def collect_utt_ids_by_category(data_root: Path) -> dict[str, list[str]]:
    """
    Walk all split CSVs and group utt_ids by the highest-priority label
    found in their annotation (speech > nonspeech > silence).
    Returns {label: [utt_id, ...]}
    """
    all_ids: list[str] = []
    for split in ("train.csv", "val.csv", "test.csv"):
        all_ids.extend(load_split_csv(data_root / split))

    # de-duplicate while preserving order
    seen = set()
    unique_ids = []
    for uid in all_ids:
        if uid not in seen:
            seen.add(uid)
            unique_ids.append(uid)

    by_label: dict[str, list[str]] = {"speech": [], "nonspeech": [], "silence": []}
    for uid in unique_ids:
        ann_path = data_root / "annotations" / f"{uid}.json"
        if not ann_path.exists():
            continue
        ann = load_annotation(ann_path)
        labels_present = {iv["label"] for iv in ann.get("intervals", [])}
        # Assign to the "richest" label present
        if "speech" in labels_present:
            by_label["speech"].append(uid)
        elif "nonspeech" in labels_present:
            by_label["nonspeech"].append(uid)
        else:
            by_label["silence"].append(uid)

    return by_label


# ── Core plotting function ────────────────────────────────────────────────────

def plot_snippet(utt_id: str, data_root: Path, cfg: dict, out_dir: Path) -> None:
    wav_path = data_root / "wavs" / f"{utt_id}.wav"
    ann_path = data_root / "annotations" / f"{utt_id}.json"

    if not wav_path.exists() or not ann_path.exists():
        print(f"  [skip] missing files for {utt_id}")
        return

    # ── Load audio ──────────────────────────────────────────────────────────
    sr_cfg      = cfg["data"]["sample_rate"]
    n_fft       = cfg["features"]["n_fft"]
    hop_length  = cfg["features"]["hop_length"]
    n_mels      = cfg["features"]["n_mels"]
    fmin        = cfg["features"]["fmin"]
    fmax        = cfg["features"]["fmax"]

    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != sr_cfg:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_cfg)
        sr = sr_cfg
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    t_audio  = np.linspace(0, duration, len(audio))

    # ── Mel spectrogram ──────────────────────────────────────────────────────
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr,
        n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)
    t_spec = librosa.frames_to_time(
        np.arange(S_db.shape[1]), sr=sr, hop_length=hop_length
    )

    # ── Annotations ─────────────────────────────────────────────────────────
    ann = load_annotation(ann_path)
    intervals = ann.get("intervals", [])

    # ── Figure layout ───────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        3, 1,
        figsize=(14, 7),
        gridspec_kw={"height_ratios": [1.2, 2.0, 0.4]},
        sharex=True,
    )
    fig.patch.set_facecolor("#1C1C2E")
    for ax in axes:
        ax.set_facecolor("#12121F")
        ax.tick_params(colors="#CCCCDD", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    def add_annotation_spans(ax, ymin=0, ymax=1, transform_y=False):
        for iv in intervals:
            color = LABEL_COLORS.get(iv["label"], "#888888")
            if transform_y:
                ax.axvspan(iv["start"], iv["end"],
                           alpha=LABEL_ALPHA, color=color, linewidth=0)
            else:
                ax.axvspan(iv["start"], iv["end"],
                           alpha=LABEL_ALPHA, color=color, linewidth=0)

    # ── Waveform ─────────────────────────────────────────────────────────────
    ax_wav = axes[0]
    ax_wav.plot(t_audio, audio, color="#7EC8E3", linewidth=0.5, alpha=0.9)
    ax_wav.set_ylabel("Amplitude", color="#AAAACC", fontsize=9)
    ax_wav.set_ylim(-1.05, 1.05)
    ax_wav.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax_wav.grid(axis="y", color="#222244", linewidth=0.4, linestyle="--")
    add_annotation_spans(ax_wav)
    ax_wav.set_title(
        f"{utt_id}   ·   {duration:.2f} s   ·   {sr/1000:.0f} kHz",
        color="#E8E8FF", fontsize=11, fontweight="bold", pad=8,
        fontfamily="monospace",
    )

    # ── Spectrogram ──────────────────────────────────────────────────────────
    ax_spec = axes[1]
    img = ax_spec.pcolormesh(
        t_spec, np.linspace(fmin, fmax, n_mels) / 1000,
        S_db, shading="gouraud", cmap="magma", vmin=-80, vmax=0,
    )
    ax_spec.set_ylabel("Freq (kHz)", color="#AAAACC", fontsize=9)
    add_annotation_spans(ax_spec)

    # Colour-bar
    cbar = fig.colorbar(img, ax=ax_spec, pad=0.01, fraction=0.015)
    cbar.ax.tick_params(colors="#AAAACC", labelsize=7)
    cbar.set_label("dB", color="#AAAACC", fontsize=8)

    # ── Annotation ribbon ────────────────────────────────────────────────────
    ax_ann = axes[2]
    ax_ann.set_yticks([])
    ax_ann.set_ylabel("Labels", color="#AAAACC", fontsize=9)
    ax_ann.set_ylim(0, 1)
    for iv in intervals:
        color = LABEL_COLORS.get(iv["label"], "#888888")
        ax_ann.axvspan(iv["start"], iv["end"], ymin=0, ymax=1,
                       color=color, alpha=0.85, linewidth=0)
        # Category sub-label if present
        cat = iv.get("category", "")
        mid = (iv["start"] + iv["end"]) / 2
        span = iv["end"] - iv["start"]
        txt = cat if cat else iv["label"]
        if span > 0.15:   # only label if wide enough
            ax_ann.text(mid, 0.5, txt, ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold",
                        fontfamily="monospace",
                        clip_on=True)

    ax_ann.set_xlabel("Time (s)", color="#AAAACC", fontsize=9)
    ax_ann.xaxis.set_tick_params(colors="#CCCCDD")

    # ── Legend ───────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=LABEL_COLORS[k], alpha=0.85, label=k.capitalize())
        for k in LABEL_COLORS
    ]
    axes[0].legend(
        handles=legend_patches, loc="upper right",
        framealpha=0.3, facecolor="#1C1C2E", edgecolor="#444466",
        labelcolor="#DDDDEE", fontsize=8,
    )

    plt.tight_layout(rect=[0, 0, 1, 1])
    plt.subplots_adjust(hspace=0.05)

    out_path = out_dir / f"{utt_id}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  saved → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Plot soft-SAD data snippets.")
    parser.add_argument("--config", default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--n", type=int, default=5,
                        help="Max samples per category (default: 5)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    args = parser.parse_args()

    random.seed(args.seed)

    cfg       = load_config(args.config)
    data_root = Path(cfg["data"]["root"])
    out_dir   = Path(cfg["data"]["root"],"snippets")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Data root : {data_root}")
    print(f"Output    : {out_dir}")
    print(f"Samples/category: {args.n}\n")

    by_label = collect_utt_ids_by_category(data_root)
    total = sum(len(v) for v in by_label.values())
    print(f"Found {total} utterances across all splits:")
    for lbl, ids in by_label.items():
        print(f"  {lbl:12s}: {len(ids)} utterances")
    print()

    for label, utt_ids in by_label.items():
        if not utt_ids:
            print(f"[{label}] no utterances found, skipping.\n")
            continue

        chosen = random.sample(utt_ids, min(args.n, len(utt_ids)))
        print(f"[{label}] plotting {len(chosen)} snippet(s):")
        for uid in chosen:
            plot_snippet(uid, data_root, cfg, out_dir)
        print()

    print(f"Done. Images saved in '{out_dir}/'")


if __name__ == "__main__":
    main()