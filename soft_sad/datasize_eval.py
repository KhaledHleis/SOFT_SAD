"""Standalone data-size (iterative) evaluation for Soft-SAD.

This module isolates the *iterative* metric-vs-data-size study from the rest
of the evaluation pipeline (F1-vs-tau, ROC, per-category). It can be run on
its own:

    python -m soft_sad.datasize_eval --config config.yaml --checkpoint runs/best.pt

What it does
------------
1. Loads the checkpoint and runs inference on the chosen split
   (via ``evaluate.load_predictions``).
2. Picks the operating threshold to hold fixed across the data-size sweep.
   By default this is the ROC-optimal point on the FULL dataset, i.e. the
   threshold whose ``(FAR, TAR)`` is closest to the ideal corner ``(0, 1)``
       dist(tau) = sqrt( FAR(tau)^2 + (1 - TAR(tau))^2 )
   rather than the F1-maximising threshold. Pass ``--tau`` to fix it by hand
   and skip the ROC sweep entirely.
3. Runs the incremental data-size sweep at that fixed threshold and writes
   ``datasize_curve.png`` / ``datasize_curve.csv`` plus a small
   ``datasize_meta.json`` recording the threshold and how it was chosen.

Why a fixed threshold (not re-optimised per slice)
--------------------------------------------------
Re-optimising the threshold at every data-size checkpoint would conflate
metric stability with threshold sensitivity. Fixing tau at the full-dataset
ROC-optimal point isolates "how does the score move as the (eval) set grows"
from "how does the best threshold move".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

from soft_sad.evaluate import (
    build_membership,
    load_predictions,
    roc_optimal_threshold,
    save_datasize_curve,
    sweep,
    sweep_by_datasize,
)


def choose_threshold(
    preds: list[dict],
    cfg: dict,
    speech_params,
    collar_frames: int,
    tau_override: float | None = None,
) -> tuple[float, dict]:
    """Return (tau, meta). If ``tau_override`` is given, use it directly;
    otherwise sweep thresholds on the full dataset and take the ROC point
    closest to (0, 1)."""
    if tau_override is not None:
        return float(tau_override), {
            "selection": "manual",
            "tau": float(tau_override),
        }

    n_grid = int(cfg["metrics"]["threshold_grid"])
    taus   = np.linspace(0.0, 1.0, n_grid)
    swept  = sweep(preds, cfg, speech_params, collar_frames, taus)
    tau, idx, dist = roc_optimal_threshold(swept["soft"], taus)
    meta = {
        "selection": "roc_closest_to_(0,1)",
        "tau": tau,
        "FAR": swept["soft"][idx]["FAR"],
        "TAR": swept["soft"][idx]["TAR"],
        "dist": dist,
        "threshold_grid": n_grid,
    }
    return tau, meta


def main():
    ap = argparse.ArgumentParser(
        description="Soft-SAD data-size scaling (standalone)."
    )
    ap.add_argument("--config",     type=str, required=True)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--split",      type=str, default="test",
                    choices=["train", "val", "test"])
    ap.add_argument("--out",        type=str, default=None,
                    help="output directory; default: alongside checkpoint")
    ap.add_argument("--tau", type=float, default=None,
                    help="fix the operating threshold directly and skip the "
                         "ROC sweep; default selects the full-dataset ROC "
                         "point closest to (0, 1)")
    ap.add_argument("--step", type=int, default=None,
                    help="utterances between scaling checkpoints "
                         "(default: metrics.datasize_step from config)")
    ap.add_argument("--n-boot", type=int, default=None,
                    help="bootstrap resamples per checkpoint for ±1σ bands "
                         "(default: metrics.datasize_n_boot from config)")
    ap.add_argument("--seed", type=int, default=None,
                    help="bootstrap RNG seed (default: metrics.datasize_seed)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))

    out_dir = (Path(args.out) if args.out
               else Path(args.checkpoint).parent / f"datasize_{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    preds = load_predictions(cfg, args.checkpoint, args.split)
    print(f"  inferred {len(preds)} utterances on split={args.split}")

    p, collar_frames = build_membership(cfg)

    tau, tau_meta = choose_threshold(preds, cfg, p, collar_frames, args.tau)
    print(f"  fixed operating threshold tau = {tau:.3f} "
          f"({tau_meta['selection']})")

    ds_step   = args.step   if args.step   is not None else int(cfg["metrics"].get("datasize_step",   1000))
    ds_n_boot = args.n_boot if args.n_boot is not None else int(cfg["metrics"].get("datasize_n_boot", 0))
    ds_seed   = args.seed   if args.seed   is not None else int(cfg["metrics"].get("datasize_seed",   42))

    print(f"  data-size scaling (step={ds_step}, n_boot={ds_n_boot}, tau={tau:.3f})...")
    rows = sweep_by_datasize(
        preds, cfg, p, collar_frames,
        tau=tau, step=ds_step, n_boot=ds_n_boot, seed=ds_seed,
    )
    save_datasize_curve(rows, out_dir)

    meta = {
        "split": args.split,
        "threshold": tau_meta,
        "datasize_scaling": {
            "tau": tau, "step": ds_step, "n_boot": ds_n_boot, "seed": ds_seed,
            "final_n": rows[-1]["n"] if rows else 0,
        },
    }
    with open(out_dir / "datasize_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  outputs written to {out_dir}")


if __name__ == "__main__":
    main()