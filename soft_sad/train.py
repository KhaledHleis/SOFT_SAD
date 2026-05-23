"""Train the GRU SAD model with per-frame weighted BCE.

The binary target at frame t is `1` if the frame label is SPEECH else `0`.
NONSPEECH and SILENCE frames are both negative; the weight on SILENCE
frames is `weight_silence` (default 0.5), encouraging the model to lean
on the harder speech/non-speech boundary.

Usage:
    python -m soft_sad.train --config config.yaml
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from . import NONSPEECH, SILENCE, SPEECH
from soft_sad.data import FeatureDataset, collate_pad
from soft_sad.model import SADGRU


def set_seed(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_weights_and_targets(labels: torch.Tensor, cfg: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """labels: (B, T) int with -1 padding.

    Returns:
        target  : (B, T) float in {0, 1}
        weight  : (B, T) float, zero on padded positions
    """
    pad_mask = (labels == -1)
    target = (labels == SPEECH).float()
    w = torch.zeros_like(target)
    w[labels == SPEECH]    = float(cfg["training"]["weight_speech"])
    w[labels == NONSPEECH] = float(cfg["training"]["weight_nonspeech"])
    w[labels == SILENCE]   = float(cfg["training"]["weight_silence"])
    w[pad_mask] = 0.0
    return target, w


def run_epoch(
    model: SADGRU,
    loader: DataLoader,
    cfg: dict,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    desc: str = "",
) -> dict:
    """One pass over `loader`. If `optimizer` is None, evaluation mode."""
    train_mode = optimizer is not None
    model.train(train_mode)

    total_loss = 0.0
    total_weight = 0.0
    total_correct_speech = 0.0
    total_count_speech = 0.0

    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        feat = batch["feat"].to(device)
        labels = batch["labels"].to(device)
        lengths = batch["lengths"].to(device)
        target, weight = make_weights_and_targets(labels, cfg)

        if train_mode:
            optimizer.zero_grad()
        logits = model(feat, lengths)
        loss_per_frame = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        loss = (loss_per_frame * weight).sum() / weight.sum().clamp(min=1e-8)

        if train_mode:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(cfg["training"]["grad_clip"])
            )
            optimizer.step()

        # Bookkeeping
        with torch.no_grad():
            total_loss += float(loss.item()) * float(weight.sum().item())
            total_weight += float(weight.sum().item())
            preds = (torch.sigmoid(logits) >= 0.5).float()
            spm = (labels == SPEECH).float()
            total_correct_speech += float(((preds == 1) * spm).sum().item())
            total_count_speech += float(spm.sum().item())

        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return {
        "loss": total_loss / max(total_weight, 1e-8),
        "speech_recall_05": (
            total_correct_speech / total_count_speech
            if total_count_speech > 0 else float("nan")
        ),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    set_seed(int(cfg["training"]["seed"]))

    feat_dir = Path(cfg["data"]["feature_cache"])
    data_root = Path(cfg["data"]["root"])
    train_ds = FeatureDataset(feat_dir, data_root / "train.csv")
    val_ds   = FeatureDataset(feat_dir, data_root / "val.csv")
    print(f"  train: {len(train_ds)} utts   val: {len(val_ds)} utts")

    train_loader = DataLoader(
        train_ds, batch_size=int(cfg["training"]["batch_size"]),
        shuffle=True, num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_pad, drop_last=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=int(cfg["training"]["batch_size"]),
        shuffle=False, num_workers=int(cfg["training"]["num_workers"]),
        collate_fn=collate_pad, drop_last=False,
    )

    # Infer input feature size from one sample
    sample = train_ds[0]
    input_size = sample["feat"].shape[1]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SADGRU(
        input_size=input_size,
        hidden_size=int(cfg["model"]["hidden_size"]),
        num_layers=int(cfg["model"]["num_layers"]),
        bidirectional=bool(cfg["model"]["bidirectional"]),
        dropout=float(cfg["model"]["dropout"]),
    ).to(device)
    print(f"  model: {model.num_parameters()} parameters on {device}")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["training"]["learning_rate"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )

    out_dir = Path(cfg["training"]["out_dir"]); out_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")
    bad_epochs = 0
    history = []
    for epoch in range(int(cfg["training"]["num_epochs"])):
        tr = run_epoch(model, train_loader, cfg, device, optimizer, desc=f"epoch {epoch} train")
        va = run_epoch(model, val_loader,   cfg, device, None,      desc=f"epoch {epoch} val  ")
        msg = (
            f"epoch {epoch:3d}  "
            f"train_loss={tr['loss']:.4f}  val_loss={va['loss']:.4f}  "
            f"val_speech_rec@0.5={va['speech_recall_05']:.3f}"
        )
        print(msg)
        history.append({"epoch": epoch, "train": tr, "val": va})

        # Best checkpoint by val loss
        improved = va["loss"] < best_val - 1e-5
        if improved:
            best_val = va["loss"]
            bad_epochs = 0
            torch.save({
                "state_dict": model.state_dict(),
                "config": cfg,
                "input_size": input_size,
                "epoch": epoch,
                "val_loss": va["loss"],
            }, out_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= int(cfg["training"]["early_stop_patience"]):
                print(f"  early stop at epoch {epoch}")
                break

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

    print(f"  best val_loss = {best_val:.4f}, checkpoint at {out_dir/'best.pt'}")


if __name__ == "__main__":
    main()
