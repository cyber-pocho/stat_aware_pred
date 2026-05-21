"""Visualization: log tracks, confusion matrices, attention heatmaps, grade maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns


# ── colour palette: one colour per lithology index (up to 12) ──────────────
_LITH_COLOURS = [
    "#f4a460",  # Sandstone   — sandy brown
    "#708090",  # Shale       — slate grey
    "#a9a9a9",  # Shale/sand  — light grey
    "#87ceeb",  # Limestone   — sky blue
    "#b0c4de",  # Lst/clay    — steel blue
    "#fffacd",  # Chalk       — lemon
    "#d2b48c",  # Marl        — tan
    "#ff69b4",  # Halite      — pink
    "#dda0dd",  # Anhydrite   — plum
    "#90ee90",  # Tuff        — light green
    "#2f4f4f",  # Coal        — dark slate
    "#ffd700",  # extra       — gold
]


def plot_log_track(
    well_df: pd.DataFrame,
    curves: list[str],
    depth_col: str = "DEPTH_MD",
    label_col: str | None = "LITHOLOGY_IDX",
    class_names: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Vertical log track plot: one subplot per curve, depth on the y-axis.
    A shaded lithology column is prepended when label_col is provided.
    """
    n_tracks = len(curves) + (1 if label_col else 0)
    fig, axes = plt.subplots(1, n_tracks,
                             figsize=figsize or (2.5 * n_tracks, 10),
                             sharey=True)
    if n_tracks == 1:
        axes = [axes]

    depth = well_df[depth_col].values
    col_idx = 0

    if label_col and label_col in well_df.columns:
        ax = axes[col_idx]
        labels = well_df[label_col].values
        for i in range(len(depth) - 1):
            c = _LITH_COLOURS[int(labels[i]) % len(_LITH_COLOURS)]
            ax.fill_betweenx([depth[i], depth[i + 1]], 0, 1, color=c)
        ax.set_xlim(0, 1)
        ax.set_xticks([])
        ax.set_ylabel("Depth (m)")
        ax.set_title("Lithology")
        ax.invert_yaxis()
        col_idx += 1

    for curve in curves:
        ax = axes[col_idx]
        if curve in well_df.columns:
            vals = well_df[curve].values
            ax.plot(vals, depth, lw=0.6, color="black")
            ax.set_xlabel(curve)
            ax.set_title(curve)
        ax.invert_yaxis()
        col_idx += 1

    fig.tight_layout()
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: list[str],
    title: str = "Confusion Matrix",
    normalize: bool = True,
    ax: plt.Axes | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> plt.Axes:
    """
    Plot a confusion matrix heatmap.

    Args:
        y_true:       ground-truth class indices
        y_pred:       predicted class indices
        class_names:  label for each class index
        normalize:    if True, normalise rows (recall per class); avoids shale dominating
        ax:           existing axes to draw into; creates a new figure if None
    """
    n = len(class_names)
    cm = np.zeros((n, n), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    present = [i for i in range(n) if cm[i].sum() > 0]
    cm = cm[np.ix_(present, present)]
    names = [class_names[i] for i in present]

    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
        cm_plot = cm.astype(float) / row_sums
        fmt, vmax = ".2f", 1.0
    else:
        cm_plot = cm
        fmt, vmax = "d", None

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)

    sns.heatmap(
        cm_plot, annot=True, fmt=fmt, cmap="Blues",
        xticklabels=names, yticklabels=names,
        vmin=0, vmax=vmax, ax=ax,
        linewidths=0.3, linecolor="lightgrey",
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    return ax


def plot_comparison_confusion_matrices(
    y_true: np.ndarray,
    y_pred_transformer: np.ndarray,
    y_pred_xgb: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    save_path: str | None = None,
) -> plt.Figure:
    """
    Side-by-side confusion matrices for Transformer vs XGBoost.
    Normalised by row (recall per class) so class imbalance doesn't distort the visual.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    plot_confusion_matrix(y_true, y_pred_transformer, class_names,
                          title="Transformer", normalize=normalize, ax=ax1)
    plot_confusion_matrix(y_true, y_pred_xgb, class_names,
                          title="XGBoost", normalize=normalize, ax=ax2)

    fig.suptitle("Lithology Classification — Transformer vs XGBoost\n"
                 "(row-normalised: each cell = recall for that class)", fontsize=13)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved: {save_path}")

    return fig


def plot_attention_heatmap(
    attn_weights: np.ndarray,
    depths: np.ndarray,
    query_labels: list[str] | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Heatmap of cross-attention weights over depth intervals.

    attn_weights: [n_queries, seq_len]
    depths:       [seq_len] measured depth in metres
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(12, max(3, len(attn_weights) * 0.6)))

    sns.heatmap(
        attn_weights, cmap="YlOrRd",
        xticklabels=False, ax=ax,
        cbar_kws={"label": "Attention weight"},
    )
    n_ticks = min(10, len(depths))
    tick_idx = np.linspace(0, len(depths) - 1, n_ticks, dtype=int)
    ax.set_xticks(tick_idx)
    ax.set_xticklabels([f"{depths[i]:.0f} m" for i in tick_idx], rotation=45)
    ax.set_xlabel("Depth")
    ax.set_ylabel("Query location")
    ax.set_title("Cross-attention weights over depth")
    if query_labels:
        ax.set_yticklabels(query_labels, rotation=0)
    return ax


def plot_grade_map(
    easting: np.ndarray,
    northing: np.ndarray,
    grade: np.ndarray,
    uncertainty: np.ndarray | None = None,
    element: str = "Grade",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    2D scatter map of predicted grade.
    Marker size encodes uncertainty when provided (larger = more uncertain).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 8))

    sizes = 80 + 200 * (uncertainty / uncertainty.max()) if uncertainty is not None else 80
    sc = ax.scatter(easting, northing, c=grade, s=sizes,
                    cmap="plasma", edgecolors="grey", linewidths=0.3)
    plt.colorbar(sc, ax=ax, label=element)
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.set_title(f"{element} prediction" +
                 (" (marker size ∝ uncertainty)" if uncertainty is not None else ""))
    return ax


def plot_reliability_diagram(
    bin_centres: np.ndarray,
    bin_accuracy: np.ndarray,
    bin_counts: np.ndarray,
    label: str = "Model",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Reliability diagram: actual accuracy vs predicted confidence per bin.
    A perfectly calibrated model lies on the diagonal.
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))

    valid = ~np.isnan(bin_accuracy)
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
    ax.bar(bin_centres[valid], bin_accuracy[valid],
           width=bin_centres[1] - bin_centres[0], alpha=0.6,
           color="steelblue", edgecolor="white", label=label)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability Diagram")
    ax.legend()
    return ax
