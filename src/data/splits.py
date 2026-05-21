"""Spatial cross-validation split generation for well log data."""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Iterator


def spatial_kfold(
    well_coords: pd.DataFrame,
    n_folds: int = 5,
    well_col: str = "WELL",
    x_col: str = "X_LOC",
    y_col: str = "Y_LOC",
    random_state: int = 42,
) -> Iterator[tuple[list[str], list[str]]]:
    """
    Yield (train_wells, val_wells) splits where val wells are geographically
    isolated from training wells.

    Uses k-means clustering on well collar coordinates to form spatially
    contiguous fold groups — held-out wells are a full spatial cluster, not
    randomly sampled nearby neighbours.

    Args:
        well_coords: one row per well with collar x/y columns.
        n_folds: number of spatial folds.
        well_col: column identifying each well.
        x_col, y_col: easting/northing columns used for clustering.
        random_state: seed for k-means initialisation.

    Yields:
        (train_well_names, val_well_names) for each fold.
    """
    raise NotImplementedError
