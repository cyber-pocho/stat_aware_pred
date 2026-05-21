"""XGBoost lithology classifier training script."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.data.loader import load_all_wells, CORE_LOG_CURVES, IDX_TO_NAME, N_CLASSES
from src.data.features import add_derived_features, DERIVED_FEATURES
from src.data.splits import get_well_collar_coords, spatial_kfold
from src.eval.metrics import macro_f1, per_class_f1
from src.models.baseline import XGBoostLithologyClassifier, build_window_features


def train(config: dict) -> None:
    df = load_all_wells(config["data"]["data_dir"], verbose=True)

    if config["data"].get("add_derived", True):
        df = add_derived_features(df)

    base_features = CORE_LOG_CURVES + (DERIVED_FEATURES if config["data"].get("add_derived") else [])
    base_features = [c for c in base_features if c in df.columns]

    if config["data"].get("add_window_features", True):
        df, feature_cols = build_window_features(df, base_features)
    else:
        feature_cols = base_features

    print(f"Total features: {len(feature_cols)}")

    # Well split — use same seed as Stage 1 for a comparable val set
    import torch
    all_wells = sorted(df["WELL"].unique().tolist())
    n_val = max(1, int(len(all_wells) * config["data"]["val_fraction"]))
    rng  = torch.Generator().manual_seed(config["training"]["seed"])
    perm = torch.randperm(len(all_wells), generator=rng).tolist()
    val_wells   = [all_wells[i] for i in perm[:n_val]]
    train_wells = [all_wells[i] for i in perm[n_val:]]

    train_df = df[df["WELL"].isin(train_wells)]
    val_df   = df[df["WELL"].isin(val_wells)]
    print(f"Train rows: {len(train_df):,}  Val rows: {len(val_df):,}")

    clf = XGBoostLithologyClassifier(**config["model"])
    clf.fit(train_df, val_df, feature_cols,
            verbose=config["training"].get("verbose", True))

    # Evaluate
    y_true = val_df["LITHOLOGY_IDX"].values
    y_pred = clf.predict(val_df)

    f1 = macro_f1(y_true, y_pred, N_CLASSES)
    per_cls = per_class_f1(y_true, y_pred, N_CLASSES)
    print(f"\nVal macro F1: {f1:.4f}")
    print("\nPer-class F1:")
    for i, score in enumerate(per_cls):
        if (y_true == i).any():
            print(f"  {IDX_TO_NAME[i]:25s}: {score:.4f}")

    ckpt = Path(config["output"]["checkpoint_dir"]) / "xgboost.pkl"
    clf.save(str(ckpt))
    print(f"\nSaved: {ckpt}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
