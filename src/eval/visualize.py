"""Visualization: log curve tracks, attention heatmaps, grade prediction maps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_log_track(
    well_df: pd.DataFrame,
    curves: list[str],
    depth_col: str = "DEPTH_MD",
    label_col: str | None = "LITHOLOGY_IDX",
    figsize: tuple[int, int] | None = None,
) -> plt.Figure:
    """
    Vertical log track plot: one subplot per curve, depth on the y-axis.
    If label_col is provided, colours the depth track by lithology class.
    """
    raise NotImplementedError


def plot_attention_heatmap(
    attn_weights: np.ndarray,
    depths: np.ndarray,
    query_locations: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Heatmap of cross-attention weights over depth intervals.

    attn_weights: [n_query_locations, seq_len]
    depths:       [seq_len] measured depth in metres
    """
    raise NotImplementedError


def plot_grade_map(
    easting: np.ndarray,
    northing: np.ndarray,
    grade: np.ndarray,
    uncertainty: np.ndarray | None = None,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    2D scatter map of predicted grade with optional uncertainty as marker size.
    """
    raise NotImplementedError


def plot_reliability_diagram(
    bin_centres: np.ndarray,
    bin_accuracy: np.ndarray,
    bin_counts: np.ndarray,
    label: str = "Model",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """
    Plots the reliability diagram alongside a perfectly calibrated diagonal.
    """
    raise NotImplementedError
