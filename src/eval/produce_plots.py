"""Generate result plots for notebooks/results/.

Produces:
    confusion_matrix.png   — row-normalised confusion matrix on the val well
    calibration_curve.png  — reliability diagram with ECE
    attention_weights.png  — layer-1 self-attention heatmap
    macro_f1_per_fold.png  — spatial k-fold CV bar chart (Transformer vs XGBoost)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import scienceplots  # noqa: F401 — registers styles as a side-effect
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

plt.style.use(["science", "no-latex"])

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.dataset import WellLogDataset
from src.data.features import add_derived_features, DERIVED_FEATURES
from src.data.loader import (
    CORE_LOG_CURVES, IDX_TO_NAME, N_CLASSES, load_all_wells,
)
from src.data.splits import get_well_collar_coords, spatial_kfold
from src.eval.metrics import (
    expected_calibration_error, macro_f1, reliability_diagram_data,
)
from src.eval.visualize import plot_confusion_matrix, plot_reliability_diagram
from src.models.baseline import XGBoostLithologyClassifier, build_window_features
from src.models.transformer import LithologyTransformer
from src.train.stage1 import focal_loss

OUTPUT_DIR = Path("notebooks/results")
CKPT_PATH  = Path("checkpoints/stage1/best.pt")
CONFIG_PATH = Path("configs/stage1.yaml")
CV_EPOCHS  = 15
CV_FOLDS   = 5


# ── helpers ──────────────────────────────────────────────────────────────────

def _load_model_and_data(config: dict, device: torch.device):
    """Return (model, feature_cols, train_stats, eval_df, y_true, y_pred, y_conf)."""
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    feature_cols = ckpt["feature_cols"]
    val_wells    = ckpt.get("val_wells")
    model_cfg    = ckpt.get("config", config)["model"]
    train_stats  = ckpt.get("train_stats")

    df = load_all_wells(config["data"]["data_dir"], verbose=False)
    if config["data"].get("add_derived", False):
        df = add_derived_features(df)

    eval_df = df[df["WELL"].isin(val_wells)] if val_wells else df

    model = LithologyTransformer(
        n_features=len(feature_cols), n_classes=N_CLASSES, **model_cfg
    ).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    ds = WellLogDataset(eval_df, feature_cols=feature_cols,
                        window_size=config["data"]["window_size"],
                        stride=config["data"]["window_size"],
                        stats=train_stats)
    loader = torch.utils.data.DataLoader(ds, batch_size=128, shuffle=False, num_workers=0)

    true_l, pred_l, conf_l = [], [], []
    with torch.no_grad():
        for batch in loader:
            feats = batch["features"].to(device)
            labs  = batch["labels"].to(device)
            deps  = batch["depths"].to(device)
            mask  = batch["padding_mask"].to(device)
            logits = model(feats, deps, padding_mask=mask)
            probs  = torch.softmax(logits, dim=-1)
            preds  = probs.argmax(dim=-1)
            conf   = probs.max(dim=-1).values
            valid  = ~mask
            true_l.append(labs[valid].cpu().numpy())
            pred_l.append(preds[valid].cpu().numpy())
            conf_l.append(conf[valid].cpu().numpy())

    return (model, feature_cols, train_stats, eval_df, df,
            np.concatenate(true_l), np.concatenate(pred_l), np.concatenate(conf_l))


def _extract_attn_weights(model, eval_df, feature_cols, train_stats, config, device):
    """Extract layer-1 self-attention weights for the first window of the val well."""
    ds = WellLogDataset(eval_df, feature_cols=feature_cols,
                        window_size=config["data"]["window_size"],
                        stride=config["data"]["window_size"],
                        stats=train_stats)
    loader = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0)

    model.eval()
    with torch.no_grad():
        batch = next(iter(loader))
        feats = batch["features"].to(device)
        deps  = batch["depths"].to(device)
        mask  = batch["padding_mask"].to(device)

        x = model.input_proj(feats)                # [1, L, d_model]
        x = x + model.pos_enc(deps)

        layer = model.encoder.layers[0]
        _, attn_w = layer.self_attn(
            x, x, x,
            key_padding_mask=mask,
            need_weights=True,
            average_attn_weights=True,
        )  # attn_w: [1, L, L]

        valid = ~mask[0]
        n_valid = valid.sum().item()
        attn = attn_w[0, :n_valid, :n_valid].cpu().numpy()
        depths = deps[0, :n_valid].cpu().numpy()

    return attn, depths


def _cv_fold_transformer(df, train_wells, val_wells, feature_cols, config, device):
    train_ds, val_ds = WellLogDataset.train_val_split(
        df, train_wells=train_wells, val_wells=val_wells,
        feature_cols=feature_cols,
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
    )
    if len(train_ds) == 0 or len(val_ds) == 0:
        return None

    model = LithologyTransformer(
        n_features=len(feature_cols), n_classes=N_CLASSES, **config["model"]
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=config["training"]["lr"],
                            weight_decay=config["training"]["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=CV_EPOCHS)
    gamma = config["training"]["focal_loss_gamma"]

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=config["training"]["batch_size"],
        shuffle=True, num_workers=0)

    for _ in range(CV_EPOCHS):
        model.train()
        for batch in train_loader:
            feats = batch["features"].to(device)
            labs  = batch["labels"].to(device)
            deps  = batch["depths"].to(device)
            mask  = batch["padding_mask"].to(device)
            loss  = focal_loss(model(feats, deps, padding_mask=mask), labs, gamma, mask)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()

    model.eval()
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=128, shuffle=False, num_workers=0)
    true_l, pred_l = [], []
    with torch.no_grad():
        for batch in val_loader:
            feats = batch["features"].to(device)
            labs  = batch["labels"].to(device)
            deps  = batch["depths"].to(device)
            mask  = batch["padding_mask"].to(device)
            preds = model(feats, deps, padding_mask=mask).argmax(dim=-1)
            valid = ~mask
            true_l.append(labs[valid].cpu().numpy())
            pred_l.append(preds[valid].cpu().numpy())

    return macro_f1(np.concatenate(true_l), np.concatenate(pred_l), N_CLASSES)


def _cv_fold_xgb(df, train_wells, val_wells, base_features, config):
    import pandas as _pd
    df_aug, feat_cols = (
        build_window_features(df, base_features)
        if config["data"].get("add_window_features", True)
        else (df, base_features)
    )
    train_df = df_aug[df_aug["WELL"].isin(train_wells)]
    val_df   = df_aug[df_aug["WELL"].isin(val_wells)]
    if len(train_df) == 0 or len(val_df) == 0:
        return None

    # XGBoost sklearn wrapper requires contiguous 0..N-1 labels in training.
    # When rare-class wells are held out, add one dummy row per missing class
    # so num_class is inferred correctly (negligible effect on 10k+ train rows).
    missing = set(range(N_CLASSES)) - set(train_df["LITHOLOGY_IDX"].unique())
    if missing:
        dummy = _pd.concat(
            [train_df.iloc[[0]].assign(LITHOLOGY_IDX=cls) for cls in sorted(missing)],
            ignore_index=True,
        )
        train_df = _pd.concat([train_df, dummy], ignore_index=True)

    clf = XGBoostLithologyClassifier(**config.get("baseline_model", {}))
    clf.fit(train_df, val_df, feat_cols, verbose=False)
    return macro_f1(val_df["LITHOLOGY_IDX"].values, clf.predict(val_df), N_CLASSES)


# ── plot functions ────────────────────────────────────────────────────────────

def save_confusion_matrix(y_true, y_pred):
    class_names = [IDX_TO_NAME[i] for i in range(N_CLASSES)]
    fig, ax = plt.subplots(figsize=(10, 8))
    plot_confusion_matrix(y_true, y_pred, class_names,
                          title="Transformer — Confusion Matrix (row-normalised)",
                          normalize=True, ax=ax)
    fig.tight_layout()
    out = OUTPUT_DIR / "confusion_matrix.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_calibration_curve(y_true, y_pred, y_conf):
    correct = y_true == y_pred
    ece = expected_calibration_error(y_conf, correct)
    centres, acc, counts = reliability_diagram_data(y_conf, correct)
    fig, ax = plt.subplots(figsize=(6, 6))
    plot_reliability_diagram(centres, acc, counts,
                             label=f"Transformer (ECE={ece:.3f})", ax=ax)
    fig.tight_layout()
    out = OUTPUT_DIR / "calibration_curve.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  ECE={ece:.4f}")
    return ece


def save_attention_heatmap(attn, depths):
    n = attn.shape[0]
    tick_idx = np.linspace(0, n - 1, min(12, n), dtype=int)

    fig, ax = plt.subplots(figsize=(12, 7))
    im = ax.imshow(attn, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    plt.colorbar(im, ax=ax, label="Attention weight")
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{depths[i]:.0f} m" for i in tick_idx], rotation=45, ha="right")
    ax.set_yticks(tick_idx)
    ax.set_yticklabels([f"{depths[i]:.0f} m" for i in tick_idx])
    ax.set_xlabel("Key depth")
    ax.set_ylabel("Query depth")
    ax.set_title("Layer-1 self-attention weights — val well 31/2-21 S")
    fig.tight_layout()
    out = OUTPUT_DIR / "attention_weights.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def save_macro_f1_per_fold(tf_scores, xgb_scores):
    n = len(tf_scores)
    fig, ax = plt.subplots(figsize=(max(7, n * 1.4), 5))
    x = np.arange(n)
    w = 0.35
    b1 = ax.bar(x - w / 2, tf_scores,  w, label="Transformer", color="steelblue")
    b2 = ax.bar(x + w / 2, xgb_scores, w, label="XGBoost",     color="coral")

    for bar in (*b1, *b2):
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.005,
                f"{v:.3f}", ha="center", va="bottom", fontsize=8)

    tf_mean,  tf_std  = np.mean(tf_scores),  np.std(tf_scores)
    xgb_mean, xgb_std = np.mean(xgb_scores), np.std(xgb_scores)
    ax.axhline(tf_mean,  ls="--", color="steelblue", alpha=0.7,
               label=f"TF mean={tf_mean:.3f} ±{tf_std:.3f}")
    ax.axhline(xgb_mean, ls="--", color="coral",     alpha=0.7,
               label=f"XGB mean={xgb_mean:.3f} ±{xgb_std:.3f}")

    ax.set_xlabel("Spatial Fold")
    ax.set_ylabel("Macro F1")
    ax.set_title(f"Macro F1 per Spatial Fold — {CV_EPOCHS}-epoch Transformer vs XGBoost")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Fold {i + 1}" for i in range(n)])
    ax.set_ylim(0, min(1.0, max(max(tf_scores), max(xgb_scores)) * 1.15 + 0.05))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    out = OUTPUT_DIR / "macro_f1_per_fold.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")
    return tf_mean, tf_std, xgb_mean, xgb_std


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # ── 1. confusion matrix + calibration ────────────────────────────────────
    print("=== [1/4] Confusion matrix & calibration ===")
    (model, feature_cols, train_stats, eval_df, df,
     y_true, y_pred, y_conf) = _load_model_and_data(config, device)

    f1_single = macro_f1(y_true, y_pred, N_CLASSES)
    print(f"Single-fold macro F1 (val well): {f1_single:.4f}")
    save_confusion_matrix(y_true, y_pred)
    ece = save_calibration_curve(y_true, y_pred, y_conf)

    # ── 2. attention weights ──────────────────────────────────────────────────
    print("\n=== [2/4] Attention weights ===")
    attn, depths = _extract_attn_weights(model, eval_df, feature_cols, train_stats, config, device)
    save_attention_heatmap(attn, depths)

    # ── 3. spatial k-fold CV ──────────────────────────────────────────────────
    print(f"\n=== [3/4] Spatial {CV_FOLDS}-fold CV ({CV_EPOCHS} epochs/fold) ===")

    # Wells with valid collar coordinates for spatial clustering
    collar = get_well_collar_coords(df).dropna(subset=["X_LOC", "Y_LOC"])
    spatial_wells = set(collar["WELL"].tolist())
    # Anomalous wells (NaN coords) are always in training, never in val
    always_train  = sorted(set(df["WELL"].unique()) - spatial_wells)
    print(f"Spatial wells : {sorted(spatial_wells)}")
    print(f"Always-train  : {always_train}")

    base_features = CORE_LOG_CURVES + (
        DERIVED_FEATURES if config["data"].get("add_derived") else []
    )
    base_features = [c for c in base_features if c in df.columns]

    tf_scores, xgb_scores = [], []
    for fold, (sp_train, sp_val) in enumerate(
        spatial_kfold(collar, n_folds=CV_FOLDS, random_state=42)
    ):
        if not sp_val:
            continue
        train_wells = sp_train + always_train
        val_wells   = sp_val
        print(f"\n  Fold {fold + 1}: val={val_wells}, n_train_wells={len(train_wells)}")

        tf_f1  = _cv_fold_transformer(df, train_wells, val_wells, feature_cols, config, device)
        xgb_f1 = _cv_fold_xgb(df, train_wells, val_wells, base_features, config)

        if tf_f1 is None or xgb_f1 is None:
            print("    skipped (empty split)")
            continue
        print(f"    TF={tf_f1:.4f}  XGB={xgb_f1:.4f}")
        tf_scores.append(tf_f1)
        xgb_scores.append(xgb_f1)

    # ── 4. save CV bar chart ──────────────────────────────────────────────────
    print("\n=== [4/4] Saving fold bar chart ===")
    tf_mean, tf_std, xgb_mean, xgb_std = save_macro_f1_per_fold(tf_scores, xgb_scores)

    print(f"\n{'='*50}")
    print(f"Single-val-well macro F1: Transformer={f1_single:.4f}")
    print(f"CV macro F1 ({CV_EPOCHS}-ep): Transformer={tf_mean:.4f}±{tf_std:.4f}  XGBoost={xgb_mean:.4f}±{xgb_std:.4f}")
    print(f"Calibration ECE: {ece:.4f}")

    return {
        "tf_single_f1": f1_single,
        "tf_cv_mean": tf_mean, "tf_cv_std": tf_std,
        "xgb_cv_mean": xgb_mean, "xgb_cv_std": xgb_std,
        "ece": ece,
    }


if __name__ == "__main__":
    results = main()
