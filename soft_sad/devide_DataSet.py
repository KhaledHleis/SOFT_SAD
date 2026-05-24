import pandas as pd
import numpy as np
import argparse, yaml, os
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    data_root = Path(cfg["data"]["root"])
    train_percentage = cfg["training"]["per_train"]
    seed = cfg["training"].get("seed", 42)
    csv_path = os.path.join(data_root, "ALL.csv")

    df = pd.read_csv(csv_path)

    rng = np.random.default_rng(seed)

    train_splits = []
    val_splits = []

    # stratify by category so the ratio holds inside every category
    for category, group in df.groupby("category"):
        group = group.sample(frac=1, random_state=int(rng.integers(0, 2**31)))  # shuffle within category

        n_train = max(1, round(len(group) * train_percentage))

        train_splits.append(group.iloc[:n_train])
        val_splits.append(group.iloc[n_train:])

    train_df = pd.concat(train_splits).sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    val_df   = pd.concat(val_splits).sample(frac=1, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)

    train_path = os.path.join(data_root, "train.csv")
    val_path   = os.path.join(data_root, "val.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path,   index=False)

    # summary
    print(f"Total   : {len(df)} samples")
    print(f"Train   : {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"Val     : {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print()
    print("Per-category breakdown:")
    summary = df.groupby("category").size().rename("total")
    summary = pd.concat([
        summary,
        train_df.groupby("category").size().rename("train"),
        val_df.groupby("category").size().rename("val"),
    ], axis=1).fillna(0).astype(int)
    print(summary.to_string())


if __name__ == "__main__":
    main()