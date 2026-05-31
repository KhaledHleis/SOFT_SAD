"""
plot_snippets.py
----------------
Generates annotated snippet images (waveform + spectrogram + annotation ribbon)
for a random selection of samples from each label category.

Pass --infer to also load the trained model and overlay:
  • the frame-level P(speech) curve
  • the membership function centred on every detected event

Use --membership / --hard to choose which metric's collar to visualise
for ground-truth events (independent of --infer).

Usage:
    # plots only
    python -m soft_sad.plot_snippets --config config.yaml

    # plots + soft membership (asymmetric trapezoid) for GT events
    python -m soft_sad.plot_snippets --config config.yaml --membership

    # plots + hard rectangular collar for GT events
    python -m soft_sad.plot_snippets --config config.yaml --hard

    # full picture: model output + soft membership (GT & detected)
    python -m soft_sad.plot_snippets --config config.yaml --infer

    # full picture: model output + hard collar (GT & detected)
    python -m soft_sad.plot_snippets --config config.yaml --infer --hard

    # --infer alone implies --membership (soft) unless --hard is also given
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
import numpy as np
import torch
import yaml
import soundfile as sf
import librosa

from soft_sad.model import SADGRU
from soft_sad.membership import MembershipParams, membership, hard_collar_membership
from soft_sad.events import detect_events, extract_ground_truth_events
from soft_sad.data import Annotation, frame_labels_from_annotation

# ── Colour palette ────────────────────────────────────────────────────────────
LABEL_COLORS = {
    "speech":    "#4CAF50",
    "nonspeech": "#FF9800",
    "silence":   "#78909C",
}
LABEL_ALPHA       = 0.35
PROB_COLOR        = "#E040FB"
PROB_FILL_ALPHA   = 0.18
THRESH_COLOR      = "#FF5252"
MU_GT_SPEECH      = "#4CAF50"   # green  – GT speech events
MU_GT_NONSPEECH   = "#FF9800"   # orange – GT nonspeech events
MU_DET_COLOR      = "#E040FB"   # purple – detected events (model)
MU_ALPHA_FILL     = 0.10
MU_ALPHA_LINE     = 0.80


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def load_split_csv(csv_path: Path) -> list[str]:
    if not csv_path.exists():
        return []
    utt_ids = []
    with open(csv_path) as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            parts = line.strip().split(",")
            if parts and parts[0]:
                utt_ids.append(parts[0])
    return utt_ids


def load_annotation(ann_path: Path) -> dict:
    with open(ann_path) as f:
        return json.load(f)


def build_membership_params(cfg: dict) -> MembershipParams:
    """Build MembershipParams from config, identical to evaluate.py."""
    return MembershipParams.from_ms(
        t1_ms=float(cfg["metrics"]["t1_ms"]),
        t2_ms=float(cfg["metrics"]["t2_ms"]),
        t3_ms=float(cfg["metrics"]["t3_ms"]),
        t4_ms=float(cfg["metrics"]["t4_ms"]),
        K_ms=float(cfg["metrics"]["K_ms"]),
        steepness=float(cfg["metrics"]["steepness"]),
        hop_length=int(cfg["features"]["hop_length"]),
        sample_rate=int(cfg["data"]["sample_rate"]),
    )


def build_collar_frames(cfg: dict) -> int:
    """Return the hard-collar half-width in frames, identical to evaluate.py."""
    sample_rate = int(cfg["data"]["sample_rate"])
    hop_length  = int(cfg["features"]["hop_length"])
    collar_ms   = cfg["metrics"]["hard_collar_ms"]
    if collar_ms is None:
        collar_ms = (cfg["metrics"]["t3_ms"] - cfg["metrics"]["t2_ms"]) / 2.0
    return int(round(float(collar_ms) * sample_rate / hop_length / 1000.0))


def load_model(cfg: dict, device: torch.device) -> SADGRU:
    """Load best.pt from cfg[training][out_dir], identical to evaluate.py."""
    ckpt_path = Path(cfg["training"]["out_dir"]) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found at {ckpt_path}. "
            "Train the model first with soft_sad.train."
        )
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    input_size = int(ckpt.get("input_size", cfg["features"]["n_mfcc"]))
    model = SADGRU(
        input_size=input_size,
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        bidirectional=bool(cfg["model"]["bidirectional"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def infer_one(utt_id: str, feat_dir: Path, model: SADGRU,
              device: torch.device) -> np.ndarray | None:
    """Return P(speech) array (T,) from the cached .npz, or None if missing."""
    npz_path = feat_dir / f"{utt_id}.npz"
    if not npz_path.exists():
        return None
    data = np.load(npz_path, allow_pickle=True)
    feat = torch.from_numpy(data["feat"].astype(np.float32)).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = model.predict_proba(feat).squeeze(0).cpu().numpy()
    return prob.astype(np.float64)


def collect_utt_ids_by_category(data_root: Path) -> dict[str, list[str]]:
    """Group utt_ids by richest label: speech > nonspeech > silence."""
    all_ids: list[str] = []
    for split in ("train.csv", "val.csv", "test.csv"):
        all_ids.extend(load_split_csv(data_root / split))

    seen: set[str] = set()
    unique_ids: list[str] = []
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
        if "speech" in labels_present:
            by_label["speech"].append(uid)
        elif "nonspeech" in labels_present:
            by_label["nonspeech"].append(uid)
        else:
            by_label["silence"].append(uid)

    return by_label


def _membership_curve(
    event_frame: int,
    n_frames: int,
    params: MembershipParams,
    hop_length: int,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_sec, mu) for the soft membership centred on `event_frame`."""
    frame_grid = np.arange(n_frames)
    delta      = frame_grid - event_frame
    mu         = membership(delta, params)
    t_sec      = librosa.frames_to_time(frame_grid, sr=sample_rate, hop_length=hop_length)
    return t_sec, mu


def _collar_curve(
    event_frame: int,
    n_frames: int,
    collar_frames: int,
    hop_length: int,
    sample_rate: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (t_sec, mu) for the hard rectangular collar centred on `event_frame`."""
    frame_grid = np.arange(n_frames)
    delta      = frame_grid - event_frame
    mu         = hard_collar_membership(delta, collar_frames)
    t_sec      = librosa.frames_to_time(frame_grid, sr=sample_rate, hop_length=hop_length)
    return t_sec, mu


# ── Core plotting function ────────────────────────────────────────────────────

def plot_snippet(
    utt_id: str,
    data_root: Path,
    cfg: dict,
    out_dir: Path,
    prob_array: np.ndarray | None = None,
    show_membership: bool = False,
    metric_mode: str = "soft",   # "soft" | "hard"
) -> None:
    wav_path = data_root / "wavs"         / f"{utt_id}.wav"
    ann_path = data_root / "annotations"  / f"{utt_id}.json"

    if not wav_path.exists() or not ann_path.exists():
        print(f"  [skip] missing wav or annotation for {utt_id}")
        return

    # ── Config values ─────────────────────────────────────────────────────────
    sr_cfg     = cfg["data"]["sample_rate"]
    n_fft      = cfg["features"]["n_fft"]
    hop_length = cfg["features"]["hop_length"]
    n_mels     = cfg["features"]["n_mels"]
    fmin       = cfg["features"]["fmin"]
    fmax       = cfg["features"]["fmax"]

    # ── Audio ─────────────────────────────────────────────────────────────────
    audio, sr = sf.read(str(wav_path), dtype="float32", always_2d=False)
    if sr != sr_cfg:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=sr_cfg)
        sr = sr_cfg
    if audio.ndim > 1:
        audio = audio.mean(axis=1)

    duration = len(audio) / sr
    t_audio  = np.linspace(0, duration, len(audio))

    # ── Mel spectrogram ───────────────────────────────────────────────────────
    S = librosa.feature.melspectrogram(
        y=audio, sr=sr, n_fft=n_fft, hop_length=hop_length,
        n_mels=n_mels, fmin=fmin, fmax=fmax,
    )
    S_db   = librosa.power_to_db(S, ref=np.max)
    n_spec_frames = S_db.shape[1]
    t_spec = librosa.frames_to_time(np.arange(n_spec_frames), sr=sr, hop_length=hop_length)

    # ── Annotations ───────────────────────────────────────────────────────────
    ann_dict  = load_annotation(ann_path)
    intervals = ann_dict.get("intervals", [])

    # ── Frame-level GT labels (needed for event extraction) ───────────────────
    ann_obj = Annotation.from_json(ann_path)
    gt_labels, _ = frame_labels_from_annotation(
        ann_obj, n_frames=n_spec_frames,
        hop_length=hop_length, sample_rate=sr, n_fft=n_fft,
    )
    gt_events = extract_ground_truth_events(gt_labels)

    # ── Membership params ─────────────────────────────────────────────────────
    mu_params     = None
    collar_frames = None
    if show_membership:
        if metric_mode == "hard":
            collar_frames = build_collar_frames(cfg)
        else:
            mu_params = build_membership_params(cfg)

    # ── Prob / detection setup ────────────────────────────────────────────────
    show_prob = prob_array is not None
    if show_prob:
        # align prob time axis to the same frame grid as the spectrogram
        n_prob = len(prob_array)
        t_prob = librosa.frames_to_time(np.arange(n_prob), sr=sr, hop_length=hop_length)
        det_events = detect_events(
            prob_array,
            threshold=0.5,
            min_gap_frames=int(cfg["events"]["min_gap_frames"]),
            smoothing_frames=int(cfg["events"]["smoothing_frames"]),
        )
    else:
        det_events = np.array([], dtype=np.int64)

    # ── Figure layout ─────────────────────────────────────────────────────────
    #   rows: waveform | spectrogram | [P(speech)] | [membership] | annotation
    height_ratios = [1.2, 2.0]
    if show_prob:
        height_ratios.append(1.0)
    if show_membership:
        height_ratios.append(1.2)
    height_ratios.append(0.4)   # annotation ribbon always last

    fig, axes = plt.subplots(
        len(height_ratios), 1,
        figsize=(14, 4 + sum(height_ratios) * 0.9),
        gridspec_kw={"height_ratios": height_ratios},
        sharex=True,
    )
    fig.patch.set_facecolor("#1C1C2E")
    for ax in axes:
        ax.set_facecolor("#12121F")
        ax.tick_params(colors="#CCCCDD", labelsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333355")

    # assign axes by position
    row = 0
    ax_wav  = axes[row]; row += 1
    ax_spec = axes[row]; row += 1
    ax_prob = axes[row] if show_prob      else None; row += (1 if show_prob      else 0)
    ax_mu   = axes[row] if show_membership else None; row += (1 if show_membership else 0)
    ax_ann  = axes[row]

    def add_label_spans(ax):
        for iv in intervals:
            ax.axvspan(iv["start"], iv["end"],
                       alpha=LABEL_ALPHA,
                       color=LABEL_COLORS.get(iv["label"], "#888888"),
                       linewidth=0)

    # ── Waveform ──────────────────────────────────────────────────────────────
    ax_wav.plot(t_audio, audio, color="#7EC8E3", linewidth=0.5, alpha=0.9)
    ax_wav.set_ylabel("Amplitude", color="#AAAACC", fontsize=9)
    ax_wav.set_ylim(-1.05, 1.05)
    ax_wav.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax_wav.grid(axis="y", color="#222244", linewidth=0.4, linestyle="--")
    add_label_spans(ax_wav)
    ax_wav.set_title(
        f"{utt_id}   ·   {duration:.2f} s   ·   {sr/1000:.0f} kHz",
        color="#E8E8FF", fontsize=11, fontweight="bold", pad=8,
        fontfamily="monospace",
    )

    # ── Spectrogram ───────────────────────────────────────────────────────────
    img = ax_spec.pcolormesh(
        t_spec, np.linspace(fmin, fmax, n_mels) / 1000,
        S_db, shading="gouraud", cmap="magma", vmin=-80, vmax=0,
    )
    ax_spec.set_ylabel("Freq (kHz)", color="#AAAACC", fontsize=9)
    add_label_spans(ax_spec)
    cbar = fig.colorbar(img, ax=ax_spec, pad=0.01, fraction=0.015)
    cbar.ax.tick_params(colors="#AAAACC", labelsize=7)
    cbar.set_label("dB", color="#AAAACC", fontsize=8)

    # ── P(speech) ─────────────────────────────────────────────────────────────
    if show_prob:
        ax_prob.fill_between(t_prob, prob_array,
                             color=PROB_COLOR, alpha=PROB_FILL_ALPHA, zorder=2)
        ax_prob.plot(t_prob, prob_array,
                     color=PROB_COLOR, linewidth=1.2, alpha=0.95, zorder=3)
        ax_prob.axhline(0.5, color=THRESH_COLOR, linewidth=0.8,
                        linestyle="--", alpha=0.7, zorder=4)
        ax_prob.set_ylabel("P(speech)", color="#AAAACC", fontsize=9)
        ax_prob.set_ylim(-0.05, 1.05)
        ax_prob.set_yticks([0.0, 0.5, 1.0])
        ax_prob.yaxis.set_tick_params(colors="#CCCCDD")
        ax_prob.grid(axis="y", color="#222244", linewidth=0.4, linestyle="--")
        add_label_spans(ax_prob)
        # mark detected event onsets with a vertical tick
        for ef in det_events:
            t_e = librosa.frames_to_time(ef, sr=sr, hop_length=hop_length)
            ax_prob.axvline(t_e, color=THRESH_COLOR, linewidth=0.7,
                            alpha=0.5, linestyle=":")

    # ── Membership / collar subplot ───────────────────────────────────────────
    if show_membership:
        ylabel = "collar (hard)" if metric_mode == "hard" else "μ (soft)"
        ax_mu.set_ylabel(ylabel, color="#AAAACC", fontsize=9)
        ax_mu.set_ylim(-0.05, 1.15)
        ax_mu.set_yticks([0.0, 0.5, 1.0])
        ax_mu.yaxis.set_tick_params(colors="#CCCCDD")
        ax_mu.grid(axis="y", color="#222244", linewidth=0.4, linestyle="--")
        add_label_spans(ax_mu)

        def curve(ef):
            if metric_mode == "hard":
                return _collar_curve(ef, n_spec_frames, collar_frames, hop_length, sr)
            return _membership_curve(ef, n_spec_frames, mu_params, hop_length, sr)

        # ── GT speech events — membership/collar curves ───────────────────────
        for ef in gt_events["speech"]:
            t_c, mu_c = curve(ef)
            ax_mu.fill_between(t_c, mu_c,
                               color=MU_GT_SPEECH, alpha=MU_ALPHA_FILL, zorder=2)
            ax_mu.plot(t_c, mu_c,
                       color=MU_GT_SPEECH, linewidth=1.0, alpha=MU_ALPHA_LINE, zorder=3)

        # ── GT nonspeech events — membership/collar curves ────────────────────
        for ef in gt_events["nonspeech"]:
            t_c, mu_c = curve(ef)
            ax_mu.fill_between(t_c, mu_c,
                               color=MU_GT_NONSPEECH, alpha=MU_ALPHA_FILL, zorder=2)
            ax_mu.plot(t_c, mu_c,
                       color=MU_GT_NONSPEECH, linewidth=1.0, alpha=MU_ALPHA_LINE, zorder=3)

        # ── Detections — vertical markers showing their score on the GT curves -
        # The membership is centred on GT events; detections are just points
        # being evaluated against those curves.  We mark each detection as a
        # vertical dashed line so the user can read off its score visually.
        for ef in det_events:
            t_d = librosa.frames_to_time(ef, sr=sr, hop_length=hop_length)
            ax_mu.axvline(t_d, color=MU_DET_COLOR, linewidth=0.9,
                          linestyle="--", alpha=0.7, zorder=4)

        ax_mu.axhline(1.0, color="#555577", linewidth=0.5, linestyle=":", zorder=1)

    # ── Annotation ribbon ─────────────────────────────────────────────────────
    ax_ann.set_yticks([])
    ax_ann.set_ylabel("Labels", color="#AAAACC", fontsize=9)
    ax_ann.set_ylim(0, 1)
    for iv in intervals:
        color = LABEL_COLORS.get(iv["label"], "#888888")
        ax_ann.axvspan(iv["start"], iv["end"], ymin=0, ymax=1,
                       color=color, alpha=0.85, linewidth=0)
        cat  = iv.get("category", "")
        txt  = cat if cat else iv["label"]
        span = iv["end"] - iv["start"]
        mid  = (iv["start"] + iv["end"]) / 2
        if span > 0.15:
            ax_ann.text(mid, 0.5, txt, ha="center", va="center",
                        color="white", fontsize=7, fontweight="bold",
                        fontfamily="monospace", clip_on=True)

    ax_ann.set_xlabel("Time (s)", color="#AAAACC", fontsize=9)
    ax_ann.xaxis.set_tick_params(colors="#CCCCDD")

    # ── Legend ────────────────────────────────────────────────────────────────
    patches = [
        mpatches.Patch(color=LABEL_COLORS[k], alpha=0.85, label=k.capitalize())
        for k in LABEL_COLORS
    ]
    if show_prob:
        patches.append(mpatches.Patch(color=PROB_COLOR,      alpha=0.85, label="P(speech)"))
    if show_membership:
        mu_label = "collar" if metric_mode == "hard" else "μ"
        patches.append(mpatches.Patch(color=MU_GT_SPEECH,    alpha=0.85, label=f"{mu_label} GT speech"))
        patches.append(mpatches.Patch(color=MU_GT_NONSPEECH, alpha=0.85, label=f"{mu_label} GT nonspeech"))
        if det_events.size > 0:
            import matplotlib.lines as mlines
            patches.append(mlines.Line2D([], [], color=MU_DET_COLOR, linewidth=1.2,
                                         linestyle="--", label="detection onset"))
    axes[0].legend(handles=patches, loc="upper right",
                   framealpha=0.3, facecolor="#1C1C2E", edgecolor="#444466",
                   labelcolor="#DDDDEE", fontsize=8)

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
    parser.add_argument("--config",     default="config.yaml",
                        help="Path to config.yaml (default: config.yaml)")
    parser.add_argument("--n",          type=int, default=5,
                        help="Max samples per category (default: 5)")
    parser.add_argument("--seed",       type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--infer",      action="store_true",
                        help=(
                            "Load best.pt and overlay the P(speech) curve "
                            "and detected-event membership/collar functions. "
                            "Implies --membership (soft) unless --hard is also given."
                        ))
    parser.add_argument("--membership", action="store_true",
                        help=(
                            "Overlay the soft asymmetric membership function "
                            "for every ground-truth event (no model needed)."
                        ))
    parser.add_argument("--hard",       action="store_true",
                        help=(
                            "Use the hard rectangular collar instead of the soft "
                            "membership function. Can be combined with --infer or "
                            "--membership."
                        ))
    args = parser.parse_args()

    # resolve which subplot to show and which metric to use
    show_membership = args.membership or args.infer or args.hard
    metric_mode     = "hard" if args.hard else "soft"

    random.seed(args.seed)

    cfg       = load_config(args.config)
    data_root = Path(cfg["data"]["root"])
    feat_dir  = Path(cfg["data"]["feature_cache"])
    out_dir   = data_root / "snippets"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model if requested ───────────────────────────────────────────────
    model  = None
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.infer:
        model = load_model(cfg, device)
        ckpt_path = Path(cfg["training"]["out_dir"]) / "best.pt"
        print(f"Model     : {ckpt_path}  ({model.num_parameters()} params, device={device})")

    if show_membership:
        if metric_mode == "hard":
            cf = build_collar_frames(cfg)
            print(f"Metric    : hard collar  ±{cf} frames")
        else:
            mp = build_membership_params(cfg)
            print(f"Metric    : soft  t1={mp.t1} t2={mp.t2} t3={mp.t3} t4={mp.t4} K={mp.K} frames  s={mp.steepness}")

    print(f"Data root : {data_root}")
    print(f"Output    : {out_dir}")
    print(f"Inference : {'yes' if model else 'no'}")
    print(f"Metric    : {metric_mode if show_membership else 'none'}\n")

    by_label = collect_utt_ids_by_category(data_root)
    total = sum(len(v) for v in by_label.values())
    print(f"Found {total} utterances across all splits:")
    for lbl, ids in by_label.items():
        print(f"  {lbl:12s}: {len(ids)}")
    print()

    for label, utt_ids in by_label.items():
        if not utt_ids:
            print(f"[{label}] no utterances found, skipping.\n")
            continue

        chosen = random.sample(utt_ids, min(args.n, len(utt_ids)))
        print(f"[{label}] plotting {len(chosen)} snippet(s):")
        for uid in chosen:
            prob_array = None
            if model is not None:
                prob_array = infer_one(uid, feat_dir, model, device)
                if prob_array is None:
                    print(f"  [warn] no cached features for {uid}, skipping model output")
            plot_snippet(uid, data_root, cfg, out_dir,
                         prob_array=prob_array,
                         show_membership=show_membership,
                         metric_mode=metric_mode)
        print()

    print(f"Done. Images saved in '{out_dir}/'")


if __name__ == "__main__":
    main()