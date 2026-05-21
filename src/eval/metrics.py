"""Evaluation metrics: F1, RMSE, MAE, ECE, and reliability diagram data."""

from __future__ import annotations

import numpy as np


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Macro-averaged F1 across all lithology classes."""
    raise NotImplementedError


def per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Per-class F1 scores; index i corresponds to class i."""
    raise NotImplementedError


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.mean(np.abs(y_true - y_pred)))


def expected_calibration_error(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    ECE: weighted average of |accuracy - confidence| over confidence bins.

    Args:
        confidences: [N] predicted confidence (max softmax probability).
        correctness: [N] bool, True if the predicted class was correct.
        n_bins:      number of equally spaced confidence bins in [0, 1].
    Returns:
        ECE scalar.
    """
    raise NotImplementedError


def reliability_diagram_data(
    confidences: np.ndarray,
    correctness: np.ndarray,
    n_bins: int = 10,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
        bin_centres: [n_bins] midpoints of confidence bins
        bin_accuracy: [n_bins] mean accuracy within each bin
        bin_counts:   [n_bins] number of samples in each bin
    """
    raise NotImplementedError
