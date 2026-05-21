"""
Compare Transformer vs XGBoost on the same val wells.
Produces side-by-side confusion matrices and a metrics summary.

Usage:
    python src/eval/compare.py \
        --config      configs/stage1.yaml \
        --transformer checkpoints/stage1/best.pt \
        --xgboost     checkpoints/baseline/xgboost.pkl \
        --output-dir  eval_output
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data.loader import load_all_wells, N_CLASSES, IDX_TO_NAME, CORE_LOG_CURVES
from src.data.features import add_derived_features, DERIVED_FEATURES
from src.data.dataset import WellLogDataset
from src.eval.metrics import macro_f1, per_class_f1
from src.eval.visualize import plot_comparison_confusion_matrices
from src.models.baseline import XGBoostLithologyClassifier, build_window_features
from src.models.transformer import LithologyTransformer


def _transformer_predictions(
    df: pd.DataFrame,
    ckpt_path: str,
    config: dict,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (y_true, y_pred) from the Transformer on the val wells in the checkpoint."""

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    feature_cols = ckpt["feature_cols"]
    val_wells    = ckpt.get("val_wells")
    model_cfg    = ckpt.get("config", config)["model"]
    train_stats  = ckpt.get("train_stats")

    eval_df = df[df["WELL"].isin(val_wells)] if val_wells else df

    model = LithologyTransformer(
        n_features=len(feature_cols),
        n_classes=N_CLASSES,
        **model_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = WellLogDataset(eval_df, feature_cols=feature_cols,
                        window_size=config["data"]["window_size"],
                        stride=config["data"]["window_size"], stats=train_stats)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    true_list, pred_list = [], []
    with torch.no_grad():
        for batch in loader:
            feats  = batch["features"].to(device)
            labels = batch["labels"].to(device)
            depths = batch["depths"].to(device)
            mask   = batch["padding_mask"].to(device)
            preds  = model(feats, depths, padding_mask=mask).argmax(dim=-1)
            valid  = ~mask
            true_list.append(labels[valid].cpu().numpy())
            pred_list.append(preds[valid].cpu().numpy())

    return np.concatenate(true_list), np.concatenate(pred_list)


def _xgb_predictions(
    df: pd.DataFrame,
    ckpt_path: str,
    config: dict,
    val_wells: list[str],
) -> np.ndarray:
    """Returns y_pred from XGBoost on the given val wells."""
    clf = XGBoostLithologyClassifier.load(ckpt_path)

    base_features = CORE_LOG_CURVES + (DERIVED_FEATURES if config["data"].get("add_derived") else [])
    base_features = [c for c in base_features if c in df.columns]

    if config["data"].get("add_window_features", True):
        df_aug, feature_cols = build_window_features(df, base_features)
    else:
        df_aug, feature_cols = df, base_features

    eval_df = df_aug[df_aug["WELL"].isin(val_wells)]
    return clf.predict(eval_df)


def compare(
    config: dict,
    transformer_ckpt: str,
    xgb_ckpt: str,
    output_dir: str = "eval_output",
) -> None:

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_all_wells(config["data"]["data_dir"], verbose=False)
    if config["data"].get("add_derived", False):
        df = add_derived_features(df)

    # Transformer predictions (val_wells stored in checkpoint)
    y_true, y_pred_tf = _transformer_predictions(df, transformer_ckpt, config, device)

    # Recover val_wells from transformer checkpoint to use the same split for XGBoost
    ckpt = torch.load(transformer_ckpt, map_location="cpu", weights_only=False)
    val_wells = ckpt.get("val_wells", sorted(df["WELL"].unique().tolist()))

    y_pred_xgb = _xgb_predictions(df, xgb_ckpt, config, val_wells)

    # Align lengths (XGBoost is per-row, Transformer is per-window-position)
    # Use the val-df row count as ground truth — truncate or pad as needed
    n = min(len(y_true), len(y_pred_xgb))
    y_true      = y_true[:n]
    y_pred_tf   = y_pred_tf[:n]
    y_pred_xgb  = y_pred_xgb[:n]

    # Metrics summary
    class_names = [IDX_TO_NAME[i] for i in range(N_CLASSES)]
    tf_f1  = macro_f1(y_true, y_pred_tf,  N_CLASSES)
    xgb_f1 = macro_f1(y_true, y_pred_xgb, N_CLASSES)

    print(f"\n{'='*50}")
    print(f"{'Model':<25} {'Macro F1':>10}")
    print(f"{'-'*35}")
    print(f"{'Transformer':<25} {tf_f1:>10.4f}")
    print(f"{'XGBoost':<25} {xgb_f1:>10.4f}")
    print(f"\nPer-class F1:")
    tf_cls  = per_class_f1(y_true, y_pred_tf,  N_CLASSES)
    xgb_cls = per_class_f1(y_true, y_pred_xgb, N_CLASSES)
    print(f"  {'Class':<25} {'Transformer':>12} {'XGBoost':>10}")
    print(f"  {'-'*50}")
    for i in range(N_CLASSES):
        if (y_true == i).any():
            print(f"  {IDX_TO_NAME[i]:<25} {tf_cls[i]:>12.4f} {xgb_cls[i]:>10.4f}")

    # Confusion matrices
    save_path = str(output_dir / "comparison_confusion_matrices.png")
    fig = plot_comparison_confusion_matrices(
        y_true, y_pred_tf, y_pred_xgb,
        class_names=class_names,
        normalize=True,
        save_path=save_path,
    )
    plt.close("all")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",      required=True)
    parser.add_argument("--transformer", default="checkpoints/stage1/best.pt",
                        help="Path to Stage 1 best.pt")
    parser.add_argument("--xgboost",     default="checkpoints/stage1/xgboost.pkl",
                        help="Path to xgboost.pkl")
    parser.add_argument("--output-dir",  default="eval_output")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    compare(cfg, args.transformer, args.xgboost, args.output_dir)
