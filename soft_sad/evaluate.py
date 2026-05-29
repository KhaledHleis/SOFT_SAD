"""Evaluation: hard vs Soft-SAD metrics with threshold sweep and data-size scaling.

Produces:
  - F1 vs threshold curve (PNG + CSV)
  - confusion matrix at best-F1 threshold (CSV)
  - ROC curve under both scorings (PNG + CSV)
  - per-category accuracy histogram at best-F1 threshold (PNG + CSV)
  - [NEW] F1 / P / R / FAR vs number of utterances at best-F1 threshold (PNG + CSV)

Usage:
    python -m soft_sad.evaluate --config config.yaml --checkpoint runs/best.pt

Config additions (all optional, with defaults shown):
    metrics:
      datasize_step: 1000       # utterances between each scaling checkpoint
      datasize_n_boot: 0        # bootstrap resamples per checkpoint (0 = no bands)
      datasize_seed: 42         # RNG seed for bootstrap
"""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from . import LABEL_NAMES, NONSPEECH, SILENCE, SPEECH
from soft_sad.data import FeatureDataset, collate_pad
from soft_sad.events import detect_events, extract_ground_truth_events
from soft_sad.membership import MembershipParams
from soft_sad.metrics_hard import hard_confusion
from soft_sad.metrics_soft import aggregate_confusions, compute_event_confusion
from soft_sad.model import SADGRU


# ---------------------------------------------------------------------
# Inference: probability stream per utterance
# ---------------------------------------------------------------------

@torch.no_grad()
def infer_probs(model, loader, device) -> list[dict]:
    """Return list of dicts: {utt_id, prob (T,), labels (T,), categories (T,)}."""
    model.eval()
    out = []
    for batch in loader:
        feat    = batch["feat"].to(device)
        labels  = batch["labels"]                # cpu int64, -1 padded
        lengths = batch["lengths"].to(device)
        logits  = model(feat, lengths)
        probs   = torch.sigmoid(logits).cpu().numpy()
        for i, utt_id in enumerate(batch["utt_ids"]):
            T = int(batch["lengths"][i].item())
            out.append({
                "utt_id":     utt_id,
                "prob":       probs[i, :T].astype(np.float64),
                "labels":     labels[i, :T].numpy().astype(np.int64),
                "categories": batch["categories_list"][i][:T],
            })
    return out


# ---------------------------------------------------------------------
# Helpers shared by sweep() and the scaling loop
# ---------------------------------------------------------------------

def _eval_one_utterance(
    u: dict,
    cfg: dict,
    speech_params: MembershipParams,
    hard_collar_frames: int,
    tau: float,
) -> tuple[dict, dict]:
    """Return (soft_conf, hard_conf) for a single utterance at threshold tau."""
    prob   = u["prob"]
    labels = u["labels"]
    det    = detect_events(
        prob, tau,
        min_gap_frames=int(cfg["events"]["min_gap_frames"]),
        smoothing_frames=int(cfg["events"]["smoothing_frames"]),
    )
    gt          = extract_ground_truth_events(labels)
    pred_labels = np.where(prob >= tau, SPEECH, NONSPEECH).astype(np.int64)

    soft = compute_event_confusion(
        detections=det,
        speech_events=gt["speech"],
        nonspeech_events=gt["nonspeech"],
        nonspeech_intervals=gt["nonspeech_intervals"],
        speech_params=speech_params,
        rigorous_nonspeech=bool(cfg["metrics"]["rigorous_nonspeech"]),
        pred_labels=pred_labels,
        gt_labels=labels,
        enable_dummy=True,
    )
    hard = hard_confusion(
        detections=det,
        speech_events=gt["speech"],
        nonspeech_events=gt["nonspeech"],
        nonspeech_intervals=gt["nonspeech_intervals"],
        collar_frames=hard_collar_frames,
        pred_labels=pred_labels,
        gt_labels=labels,
    )
    return soft, hard


# ---------------------------------------------------------------------
# Threshold sweep
# ---------------------------------------------------------------------

def sweep(
    predictions: list[dict],
    cfg: dict,
    speech_params: MembershipParams,
    hard_collar_frames: int,
    thresholds: np.ndarray,
) -> dict:
    """For each threshold, aggregate hard + soft confusion matrices across all utts."""
    soft_by_tau = []
    hard_by_tau = []
    for tau in thresholds:
        soft_per_utt = []
        hard_per_utt = []
        for u in predictions:
            soft, hard = _eval_one_utterance(u, cfg, speech_params, hard_collar_frames, tau)
            soft_per_utt.append(soft)
            if cfg["metrics"]["also_compute_hard"]:
                hard_per_utt.append(hard)

        soft_by_tau.append(aggregate_confusions(soft_per_utt))
        hard_by_tau.append(aggregate_confusions(hard_per_utt) if hard_per_utt else None)

    return {"soft": soft_by_tau, "hard": hard_by_tau, "tau": thresholds}


# ---------------------------------------------------------------------
# Per-category accuracy at a chosen threshold
# ---------------------------------------------------------------------

def per_category_accuracy(
    predictions: list[dict],
    cfg: dict,
    speech_params: MembershipParams,
    hard_collar_frames: int,
    tau: float,
) -> dict:
    """Map category string -> {soft_TP, hard_TP, n_events}.

    Categories come from the annotation 'category' field (e.g. 'coughing').
    Untyped non-speech events fall into category 'nonspeech'; untyped speech
    events into 'speech'.
    """
    per_cat = defaultdict(lambda: {"soft_TP": 0.0, "hard_TP": 0, "n": 0})

    for u in predictions:
        cats = u["categories"]
        soft, hard = _eval_one_utterance(u, cfg, speech_params, hard_collar_frames, tau)

        for (ev_frame, score_soft, _) in soft["match_speech"]:
            cat = cats[ev_frame] if 0 <= ev_frame < len(cats) and cats[ev_frame] else "speech"
            per_cat[cat]["soft_TP"] += float(score_soft)
            per_cat[cat]["n"] += 1
        for (ev_frame, score_soft, _) in soft["match_nonspeech"]:
            cat = cats[ev_frame] if 0 <= ev_frame < len(cats) and cats[ev_frame] else "nonspeech"
            per_cat[cat]["soft_TP"] += float(score_soft)
            per_cat[cat]["n"] += 1

        for (ev_frame, score_hard, _) in hard["match_speech"]:
            cat = cats[ev_frame] if 0 <= ev_frame < len(cats) and cats[ev_frame] else "speech"
            per_cat[cat]["hard_TP"] += 1 if score_hard >= 1.0 else 0
        for (ev_frame, score_hard, _) in hard["match_nonspeech"]:
            cat = cats[ev_frame] if 0 <= ev_frame < len(cats) and cats[ev_frame] else "nonspeech"
            per_cat[cat]["hard_TP"] += 1 if score_hard >= 1.0 else 0

    return dict(per_cat)


# ---------------------------------------------------------------------
# [NEW] Data-size scaling
# ---------------------------------------------------------------------

def _incremental_slices(predictions: list[dict], step: int):
    """Yield (n, slice) for n = step, 2*step, ..., N (N always included)."""
    N = len(predictions)
    checkpoints = list(range(step, N, step))
    if not checkpoints or checkpoints[-1] != N:
        checkpoints.append(N)
    for end in checkpoints:
        yield end, predictions[:end]


def _aggregate_subset(
    subset: list[dict],
    cfg: dict,
    speech_params: MembershipParams,
    hard_collar_frames: int,
    tau: float,
) -> tuple[dict, dict]:
    """Aggregate soft and hard confusions over a subset at fixed tau."""
    soft_confs, hard_confs = [], []
    for u in subset:
        s, h = _eval_one_utterance(u, cfg, speech_params, hard_collar_frames, tau)
        soft_confs.append(s)
        hard_confs.append(h)
    return aggregate_confusions(soft_confs), aggregate_confusions(hard_confs)


def sweep_by_datasize(
    predictions: list[dict],
    cfg: dict,
    speech_params: MembershipParams,
    hard_collar_frames: int,
    tau: float,
    step: int = 1000,
    n_boot: int = 0,
    seed: int = 42,
) -> list[dict]:
    """Evaluate metrics at growing data subsets (fixed tau).

    Parameters
    ----------
    predictions : list of per-utterance dicts (output of infer_probs).
    cfg, speech_params, hard_collar_frames : forwarded to _eval_one_utterance.
    tau : float
        Fixed operating threshold (use the best_tau from the full-set sweep).
        Not re-optimised per slice — that would conflate metric stability with
        threshold sensitivity.
    step : int
        Number of utterances between consecutive checkpoints.
    n_boot : int
        Number of bootstrap resamples per checkpoint for confidence intervals.
        0 disables bootstrap (no std columns in the output).
    seed : int
        RNG seed for reproducible bootstrap sampling.

    Returns
    -------
    List of dicts, one per checkpoint::

        {
          "n":        int,          # number of utterances in this slice
          "soft":     dict,         # aggregated soft confusion (9 keys)
          "hard":     dict,         # aggregated hard confusion (9 keys)
          # present only when n_boot > 0:
          "soft_std": dict,         # std of each metric across bootstrap draws
          "hard_std": dict,
        }
    """
    rng = random.Random(seed)
    rows = []

    for n, subset in _incremental_slices(predictions, step):
        soft_agg, hard_agg = _aggregate_subset(
            subset, cfg, speech_params, hard_collar_frames, tau
        )
        row = {"n": n, "soft": soft_agg, "hard": hard_agg}

        if n_boot > 0:
            # Bootstrap: resample utterances with replacement, re-aggregate.
            keys = ["F1", "P", "R", "TAR", "FAR"]
            boot_soft = {k: [] for k in keys}
            boot_hard = {k: [] for k in keys}
            for _ in range(n_boot):
                sample = rng.choices(subset, k=len(subset))
                s_b, h_b = _aggregate_subset(
                    sample, cfg, speech_params, hard_collar_frames, tau
                )
                for k in keys:
                    boot_soft[k].append(s_b[k])
                    boot_hard[k].append(h_b[k])
            row["soft_std"] = {k: float(np.std(boot_soft[k])) for k in keys}
            row["hard_std"] = {k: float(np.std(boot_hard[k])) for k in keys}

        rows.append(row)
        print(f"    n={n:5d}  soft_F1={soft_agg['F1']:.3f}  hard_F1={hard_agg['F1']:.3f}")

    return rows


# ---------------------------------------------------------------------
# [NEW] Output helpers for data-size scaling
# ---------------------------------------------------------------------

def save_datasize_curve(rows: list[dict], out_dir: Path) -> None:
    """Save F1 / P / R / FAR vs n plots (PNG) and the full table (CSV).

    Layout: 2×2 subplots (F1, Precision, Recall, FAR), soft and hard
    overlaid on each panel.  When bootstrap std is present, a ±1σ shaded
    band is drawn around each curve.
    """
    ns       = [r["n"]           for r in rows]
    metrics  = ["F1", "P", "R", "FAR"]
    labels   = {"F1": "F1", "P": "Precision", "R": "Recall", "FAR": "FAR"}
    has_boot = "soft_std" in rows[0]

    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    axes = axes.flatten()

    for ax, key in zip(axes, metrics):
        soft_vals = np.array([r["soft"][key] for r in rows])
        hard_vals = np.array([r["hard"][key] for r in rows])

        ax.plot(ns, soft_vals, label="soft", lw=2, color="indianred")
        ax.plot(ns, hard_vals, label="hard", lw=2, color="steelblue", linestyle="--")

        if has_boot:
            soft_std = np.array([r["soft_std"][key] for r in rows])
            hard_std = np.array([r["hard_std"][key] for r in rows])
            ax.fill_between(ns,
                            soft_vals - soft_std, soft_vals + soft_std,
                            alpha=0.2, color="indianred")
            ax.fill_between(ns,
                            hard_vals - hard_std, hard_vals + hard_std,
                            alpha=0.2, color="steelblue")

        ax.set_ylabel(labels[key])
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

    for ax in axes[2:]:
        ax.set_xlabel("n utterances")

    title = "Metrics vs data size"
    if has_boot:
        title += "  (±1σ bootstrap band)"
    fig.suptitle(title, fontsize=11)
    plt.tight_layout()
    plt.savefig(out_dir / "datasize_curve.png", dpi=160)
    plt.close()

    # --- CSV ---
    fieldnames = ["n",
                  "soft_F1", "hard_F1",
                  "soft_P",  "hard_P",
                  "soft_R",  "hard_R",
                  "soft_FAR","hard_FAR"]
    if has_boot:
        fieldnames += ["soft_F1_std", "hard_F1_std",
                       "soft_P_std",  "hard_P_std",
                       "soft_R_std",  "hard_R_std",
                       "soft_FAR_std","hard_FAR_std"]

    with open(out_dir / "datasize_curve.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(fieldnames)
        for r in rows:
            s, h = r["soft"], r["hard"]
            row_vals = [
                r["n"],
                f"{s['F1']:.4f}", f"{h['F1']:.4f}",
                f"{s['P']:.4f}",  f"{h['P']:.4f}",
                f"{s['R']:.4f}",  f"{h['R']:.4f}",
                f"{s['FAR']:.4f}",f"{h['FAR']:.4f}",
            ]
            if has_boot:
                ss, hs = r["soft_std"], r["hard_std"]
                row_vals += [
                    f"{ss['F1']:.4f}", f"{hs['F1']:.4f}",
                    f"{ss['P']:.4f}",  f"{hs['P']:.4f}",
                    f"{ss['R']:.4f}",  f"{hs['R']:.4f}",
                    f"{ss['FAR']:.4f}",f"{hs['FAR']:.4f}",
                ]
            w.writerow(row_vals)


# ---------------------------------------------------------------------
# output helpers 
# ---------------------------------------------------------------------

def save_f1_curve(taus, soft_by_tau, hard_by_tau, out_dir: Path):
    soft_f1 = [c["F1"] for c in soft_by_tau]
    hard_f1 = [c["F1"] if c is not None else None for c in hard_by_tau]
    plt.figure(figsize=(5, 3))
    plt.plot(taus, soft_f1, label="soft", lw=2)
    if hard_by_tau[0] is not None:
        plt.plot(taus, hard_f1, label="hard", lw=2)
    plt.xlabel("threshold $\\tau$"); plt.ylabel("F1")
    plt.title("F1 vs threshold")
    plt.legend(); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "f1_vs_tau.png", dpi=160); plt.close()

    with open(out_dir / "f1_vs_tau.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "soft_F1", "soft_P", "soft_R", "hard_F1", "hard_P", "hard_R"])
        for i, tau in enumerate(taus):
            s = soft_by_tau[i]; h = hard_by_tau[i]
            w.writerow([
                f"{tau:.3f}", f"{s['F1']:.4f}", f"{s['P']:.4f}", f"{s['R']:.4f}",
                f"{h['F1']:.4f}" if h is not None else "",
                f"{h['P']:.4f}" if h is not None else "",
                f"{h['R']:.4f}" if h is not None else "",
            ])


def save_roc(taus, soft_by_tau, hard_by_tau, out_dir: Path):
    """Save ROC. Points are sorted by FAR (then -TAR) so the plot is monotone."""
    def _sorted_xy(confs):
        far = np.array([c["FAR"] for c in confs])
        tar = np.array([c["TAR"] for c in confs])
        order = np.lexsort((-tar, far))
        return far[order], tar[order]

    plt.figure(figsize=(4.5, 4.5))
    sfar, star = _sorted_xy(soft_by_tau)
    plt.plot(sfar, star, label="soft", lw=2, marker="o", markersize=3)
    if hard_by_tau[0] is not None:
        hfar, htar = _sorted_xy(hard_by_tau)
        plt.plot(hfar, htar, label="hard", lw=2, marker="s", markersize=3)
    plt.plot([0, 1], [0, 1], "k--", lw=0.5)
    plt.xlim(-0.02, 1.02); plt.ylim(-0.02, 1.02)
    plt.xlabel("FAR"); plt.ylabel("TAR")
    plt.title("ROC")
    plt.legend(loc="lower right"); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(out_dir / "roc.png", dpi=160); plt.close()

    with open(out_dir / "roc.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["tau", "soft_TAR", "soft_FAR", "hard_TAR", "hard_FAR"])
        for i, tau in enumerate(taus):
            s = soft_by_tau[i]; h = hard_by_tau[i]
            w.writerow([
                f"{tau:.3f}", f"{s['TAR']:.4f}", f"{s['FAR']:.4f}",
                f"{h['TAR']:.4f}" if h is not None else "",
                f"{h['FAR']:.4f}" if h is not None else "",
            ])


def save_confusion(conf: dict, name: str, out_dir: Path):
    with open(out_dir / f"confusion_{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["TP", "FN", "FP", "TN", "P", "R", "F1", "TAR", "FAR"])
        w.writerow([
            f"{conf['TP']:.3f}", f"{conf['FN']:.3f}",
            f"{conf['FP']:.3f}", f"{conf['TN']:.3f}",
            f"{conf['P']:.4f}", f"{conf['R']:.4f}", f"{conf['F1']:.4f}",
            f"{conf['TAR']:.4f}", f"{conf['FAR']:.4f}",
        ])


def save_per_category(per_cat: dict, out_dir: Path):
    cats  = sorted(per_cat.keys())
    soft  = [per_cat[c]["soft_TP"] / max(per_cat[c]["n"], 1) for c in cats]
    hard  = [per_cat[c]["hard_TP"] / max(per_cat[c]["n"], 1) for c in cats]

    x = np.arange(len(cats)); width = 0.4
    plt.figure(figsize=(max(6, len(cats) * 0.6), 3))
    plt.bar(x - width / 2, hard, width, label="hard", color="steelblue")
    plt.bar(x + width / 2, soft, width, label="soft", color="indianred")
    plt.xticks(x, cats, rotation=45, ha="right")
    plt.ylabel("acceptance rate"); plt.ylim(0, 1.05)
    plt.legend(); plt.tight_layout()
    plt.savefig(out_dir / "per_category.png", dpi=160); plt.close()

    with open(out_dir / "per_category.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["category", "n_events", "hard_acceptance", "soft_acceptance"])
        for c in cats:
            n = per_cat[c]["n"]
            w.writerow([
                c, n,
                f"{per_cat[c]['hard_TP']/max(n,1):.4f}",
                f"{per_cat[c]['soft_TP']/max(n,1):.4f}",
            ])


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config",     type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split",      type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--out",        type=str, default=None,
                    help="output directory; default: alongside checkpoint")
    args = ap.parse_args()

    cfg  = yaml.safe_load(open(args.config))
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    input_size = int(ckpt.get("input_size", cfg["features"]["n_mfcc"]))

    out_dir = Path(args.out) if args.out else Path(args.checkpoint).parent / f"eval_{args.split}"
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = SADGRU(
        input_size=input_size,
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        bidirectional=bool(cfg["model"]["bidirectional"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    feat_dir  = Path(cfg["data"]["feature_cache"])
    data_root = Path(cfg["data"]["root"])
    ds        = FeatureDataset(feat_dir, data_root / f"{args.split}.csv")
    loader    = DataLoader(
        ds, batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False, num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_pad,
    )
    preds = infer_probs(model, loader, device)
    print(f"  inferred {len(preds)} utterances on split={args.split}")

    # Membership params
    sample_rate = int(cfg["data"]["sample_rate"])
    hop_length  = int(cfg["features"]["hop_length"])
    p = MembershipParams.from_ms(
        t1_ms=float(cfg["metrics"]["t1_ms"]),
        t2_ms=float(cfg["metrics"]["t2_ms"]),
        t3_ms=float(cfg["metrics"]["t3_ms"]),
        t4_ms=float(cfg["metrics"]["t4_ms"]),
        K_ms=float(cfg["metrics"]["K_ms"]),
        steepness=float(cfg["metrics"]["steepness"]),
        hop_length=hop_length, sample_rate=sample_rate,
    )
    collar_ms = cfg["metrics"]["hard_collar_ms"]
    if collar_ms is None:
        collar_ms = (cfg["metrics"]["t3_ms"] - cfg["metrics"]["t2_ms"]) / 2.0
    collar_frames = int(round(float(collar_ms) * sample_rate / hop_length / 1000.0))
    print(f"  soft membership: t1={p.t1} t2={p.t2} t3={p.t3} t4={p.t4} K={p.K} (frames)")
    print(f"  hard collar     : {collar_frames} frames ({collar_ms} ms)")

    # ----------------------------------------------------------------
    # Threshold sweep (unchanged)
    # ----------------------------------------------------------------
    n_grid = int(cfg["metrics"]["threshold_grid"])
    taus   = np.linspace(0.0, 1.0, n_grid)
    swept  = sweep(preds, cfg, p, collar_frames, taus)
    save_f1_curve(taus, swept["soft"], swept["hard"], out_dir)
    save_roc     (taus, swept["soft"], swept["hard"], out_dir)

    # Best F1 (under soft scoring) and full diagnostics there
    soft_f1  = [c["F1"] for c in swept["soft"]]
    best_idx = int(np.argmax(soft_f1))
    best_tau = float(taus[best_idx])
    print(f"  best soft F1 = {soft_f1[best_idx]:.3f} at tau = {best_tau:.3f}")
    save_confusion(swept["soft"][best_idx], "soft_at_best_tau", out_dir)
    if swept["hard"][best_idx] is not None:
        save_confusion(swept["hard"][best_idx], "hard_at_best_tau", out_dir)
    per_cat = per_category_accuracy(preds, cfg, p, collar_frames, best_tau)
    save_per_category(per_cat, out_dir)

    # ----------------------------------------------------------------
    # [NEW] Data-size scaling curve
    # ----------------------------------------------------------------
    ds_step   = int(cfg["metrics"].get("datasize_step",   1000))
    ds_n_boot = int(cfg["metrics"].get("datasize_n_boot", 0))
    ds_seed   = int(cfg["metrics"].get("datasize_seed",   42))

    print(f"  computing data-size scaling (step={ds_step}, n_boot={ds_n_boot})...")
    scaling_rows = sweep_by_datasize(
        preds, cfg, p, collar_frames,
        tau=best_tau,
        step=ds_step,
        n_boot=ds_n_boot,
        seed=ds_seed,
    )
    save_datasize_curve(scaling_rows, out_dir)
    print(f"  data-size scaling written to {out_dir / 'datasize_curve.png'}")

    # ----------------------------------------------------------------
    # Diagnostics summary (extended with scaling info)
    # ----------------------------------------------------------------
    summary = {
        "split":         args.split,
        "best_tau":      best_tau,
        "soft_at_best":  swept["soft"][best_idx],
        "hard_at_best":  swept["hard"][best_idx],
        "membership_frames": dict(
            t1=p.t1, t2=p.t2, t3=p.t3, t4=p.t4, K=p.K, s=p.steepness
        ),
        "collar_frames":      collar_frames,
        "rigorous_nonspeech": bool(cfg["metrics"]["rigorous_nonspeech"]),
        # New: record the scaling config so results are reproducible
        "datasize_scaling": {
            "step":   ds_step,
            "n_boot": ds_n_boot,
            "seed":   ds_seed,
            "final_n": scaling_rows[-1]["n"] if scaling_rows else 0,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  outputs written to {out_dir}")


if __name__ == "__main__":
    main()