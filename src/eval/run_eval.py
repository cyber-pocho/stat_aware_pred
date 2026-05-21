"""Evaluate a trained Stage 1 checkpoint and report metrics + confusion matrix."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml

from src.data.loader import load_all_wells, N_CLASSES, IDX_TO_NAME, CORE_LOG_CURVES
from src.data.features import add_derived_features, DERIVED_FEATURES
from src.data.dataset import WellLogDataset
from src.eval.metrics import macro_f1, per_class_f1, expected_calibration_error, reliability_diagram_data
from src.eval.visualize import plot_confusion_matrix, plot_reliability_diagram
from src.models.transformer import LithologyTransformer

import matplotlib.pyplot as plt


def run_eval(config: dict, checkpoint_path: str, output_dir: str = "eval_output") -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    feature_cols = ckpt["feature_cols"]
    val_wells    = ckpt.get("val_wells")
    model_cfg    = ckpt.get("config", config)["model"]

    print(f"Checkpoint: epoch {ckpt['epoch']}  val_loss={ckpt.get('val_loss', '?'):.4f}")
    if val_wells:
        print(f"Val wells ({len(val_wells)}): {val_wells}")

    # Load data
    df = load_all_wells(config["data"]["data_dir"], verbose=False)
    if config["data"].get("add_derived", False):
        df = add_derived_features(df)

    if val_wells:
        eval_df = df[df["WELL"].isin(val_wells)]
    else:
        print("No val_wells in checkpoint — evaluating on all wells (optimistic).")
        eval_df = df

    # Build model and load weights
    model = LithologyTransformer(
        n_features=len(feature_cols),
        n_classes=N_CLASSES,
        **model_cfg,
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Build dataset (no augmentation / shuffling)
    ds = WellLogDataset(
        eval_df,
        feature_cols=feature_cols,
        window_size=config["data"]["window_size"],
        stride=config["data"]["window_size"],   # non-overlapping for eval
        stats=None,
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=2)

    all_true, all_pred, all_conf = [], [], []

    with torch.no_grad():
        for batch in loader:
            features = batch["features"].to(device)
            labels   = batch["labels"].to(device)
            depths   = batch["depths"].to(device)
            mask     = batch["padding_mask"].to(device)

            logits = model(features, depths, padding_mask=mask)
            probs  = torch.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1)
            conf   = probs.max(dim=-1).values

            valid = ~mask
            all_true.append(labels[valid].cpu().numpy())
            all_pred.append(preds[valid].cpu().numpy())
            all_conf.append(conf[valid].cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    y_conf = np.concatenate(all_conf)

    # Metrics
    f1 = macro_f1(y_true, y_pred, N_CLASSES)
    per_cls = per_class_f1(y_true, y_pred, N_CLASSES)
    correct = (y_true == y_pred)
    ece = expected_calibration_error(y_conf, correct)

    print(f"\n{'='*50}")
    print(f"Macro F1:  {f1:.4f}")
    print(f"ECE:       {ece:.4f}")
    print(f"\nPer-class F1:")
    for i, score in enumerate(per_cls):
        if (y_true == i).any():
            n = (y_true == i).sum()
            print(f"  {IDX_TO_NAME[i]:25s}: {score:.4f}  (n={n:,})")

    # Confusion matrix
    class_names = [IDX_TO_NAME[i] for i in range(N_CLASSES)]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    plot_confusion_matrix(y_true, y_pred, class_names,
                          title="Transformer — row-normalised confusion matrix",
                          normalize=True, ax=axes[0])

    bin_centres, bin_acc, bin_counts = reliability_diagram_data(y_conf, correct)
    plot_reliability_diagram(bin_centres, bin_acc, bin_counts,
                             label=f"Transformer (ECE={ece:.3f})", ax=axes[1])

    fig.tight_layout()
    out_path = output_dir / "eval_transformer.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved figures → {out_path}")
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config",     required=True,  help="Path to stage1.yaml")
    parser.add_argument("--checkpoint", required=True,  help="Path to best.pt checkpoint")
    parser.add_argument("--output-dir", default="eval_output")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_eval(cfg, args.checkpoint, args.output_dir)
