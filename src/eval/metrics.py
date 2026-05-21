"""Evaluation metrics: F1, RMSE, MAE, ECE, and reliability diagram data."""

from __future__ import annotations

import numpy as np


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 across all lithology classes."""
    scores = per_class_f1(y_true, y_pred, n_classes)
    present = [s for i, s in enumerate(scores) if np.any(y_true == i)]
    return float(np.mean(present)) if present else 0.0


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class F1 scores; index i corresponds to class i."""
    scores = np.zeros(n_classes)
    for c in range(n_classes):
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        fn = np.sum((y_pred != c) & (y_true == c))
        denom = 2 * tp + fp + fn
        scores[c] = (2 * tp / denom) if denom > 0 else 0.0
    return scores


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def reliability_diagram_data(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        bin_centres:  [n_bins] midpoints of confidence bins
        bin_accuracy: [n_bins] mean accuracy within each bin (nan if empty)
        bin_counts:   [n_bins] number of samples in each bin
    """
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2
    accuracy = np.full(n_bins, np.nan)
    counts = np.zeros(n_bins, dtype=int)
    for i in range(n_bins):
        mask = (confidences >= bins[i]) & (confidences < bins[i + 1])
        counts[i] = mask.sum()
        if counts[i] > 0:
            accuracy[i] = correctness[mask].mean()
    return centres, accuracy, counts


def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """ECE: weighted average of |accuracy - confidence| over confidence bins."""
    _, accuracy, counts = reliability_diagram_data(confidences, correctness, n_bins)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centres = (bins[:-1] + bins[1:]) / 2
    valid = counts > 0
    ece = np.sum(counts[valid] * np.abs(accuracy[valid] - centres[valid])) / counts.sum()
    return float(ece)
