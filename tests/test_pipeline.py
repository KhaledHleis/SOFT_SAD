"""End-to-end smoke test using the toy dataset.

Verifies that:
    - make_toy_data writes valid wavs + annotations + splits
    - prepare extracts features without errors
    - train converges (loss decreases over a couple of epochs)
    - evaluate produces the expected output files
"""
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml


def _run(cmd: list[str], cwd: Path):
    """Run a subprocess, fail loudly."""
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print("STDOUT:", result.stdout)
        print("STDERR:", result.stderr)
        raise AssertionError(f"command {cmd} failed")
    return result


def _write_config(toy_dir: Path, cfg_path: Path, runs_dir: Path):
    cfg = {
        "data": {
            "root": str(toy_dir),
            "sample_rate": 16000,
            "feature_cache": str(toy_dir / "features"),
        },
        "features": {
            "n_fft": 1024, "hop_length": 512,
            "n_mels": 13, "n_mfcc": 13,
            "fmin": 20, "fmax": 8000,
        },
        "model": {"hidden_size": 5, "num_layers": 1, "bidirectional": False, "dropout": 0.0},
        "training": {
            "batch_size": 4,
            "num_epochs": 3,
            "learning_rate": 3e-3,
            "weight_decay": 0.0,
            "grad_clip": 1.0,
            "weight_speech": 1.0,
            "weight_nonspeech": 1.0,
            "weight_silence": 0.5,
            "num_workers": 0,
            "seed": 0,
            "log_every": 50,
            "out_dir": str(runs_dir),
            "early_stop_patience": 99,
        },
        "events": {"min_gap_frames": 1, "smoothing_frames": 1},
        "metrics": {
            "t1_ms": -240, "t2_ms": -80, "t3_ms": 80, "t4_ms": 400, "K_ms": 1600,
            "steepness": 2.0,
            "threshold_grid": 11,
            "rigorous_nonspeech": True,
            "also_compute_hard": True,
            "hard_collar_ms": None,
        },
    }
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)


def test_end_to_end_pipeline(tmp_path: Path):
    repo_root = Path(__file__).resolve().parent.parent

    toy = tmp_path / "toy"
    runs = tmp_path / "runs"
    cfg_path = tmp_path / "config.yaml"

    # 1) generate toy data
    _run([
        sys.executable, "-m", "soft_sad.make_toy_data",
        "--out", str(toy), "--n", "20", "--seed", "0",
    ], cwd=repo_root)
    assert (toy / "wavs").is_dir()
    assert (toy / "train.csv").exists()
    assert (toy / "val.csv").exists()
    assert (toy / "test.csv").exists()

    # 2) write a config that points at the toy data
    _write_config(toy, cfg_path, runs)

    # 3) prepare
    _run([sys.executable, "-m", "soft_sad.prepare", "--config", str(cfg_path)], cwd=repo_root)
    feats = list((toy / "features").glob("*.npz"))
    assert len(feats) == 20

    # 4) train
    _run([sys.executable, "-m", "soft_sad.train", "--config", str(cfg_path)], cwd=repo_root)
    assert (runs / "best.pt").exists()
    history = yaml.safe_load(open(runs / "history.json"))  # actually JSON but yaml parses superset
    losses = [h["train"]["loss"] for h in history]
    assert losses[-1] <= losses[0] * 1.05, f"training loss did not decrease: {losses}"

    # 5) evaluate
    _run([
        sys.executable, "-m", "soft_sad.evaluate",
        "--config", str(cfg_path),
        "--checkpoint", str(runs / "best.pt"),
        "--split", "test",
    ], cwd=repo_root)
    out = runs / "best.pt"
    eval_dir = (runs / "eval_test")
    assert (eval_dir / "f1_vs_tau.png").exists()
    assert (eval_dir / "f1_vs_tau.csv").exists()
    assert (eval_dir / "roc.png").exists()
    assert (eval_dir / "roc.csv").exists()
    assert (eval_dir / "summary.json").exists()
    assert (eval_dir / "per_category.png").exists()
