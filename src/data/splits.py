"""Spatial cross-validation split generation for well log data."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


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

    Uses k-means clustering on collar coordinates so each fold is a spatially
    contiguous group — held-out wells are a full spatial cluster, not
    randomly sampled nearby neighbours. This avoids the optimistic bias of
    random splits in spatial problems (nearby wells share geology).

    Args:
        well_coords: one row per well with collar x/y columns.
        n_folds:     number of spatial folds (k-means clusters).
        well_col:    column identifying each well.
        x_col, y_col: easting/northing columns used for clustering.
        random_state: seed for k-means initialisation.

    Yields:
        (train_well_names, val_well_names) for each fold.
    """
    coords = well_coords[[x_col, y_col]].values.astype(float)
    well_names = well_coords[well_col].values

    kmeans = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    labels = kmeans.fit_predict(coords)

    for fold in range(n_folds):
        val_mask   = labels == fold
        val_wells  = well_names[val_mask].tolist()
        train_wells = well_names[~val_mask].tolist()
        yield train_wells, val_wells


def get_well_collar_coords(df: pd.DataFrame,
                            well_col: str = "WELL",
                            x_col: str = "X_LOC",
                            y_col: str = "Y_LOC") -> pd.DataFrame:
    """Return one row per well with the collar (shallowest depth) coordinates."""
    return (
        df.sort_values("DEPTH_MD")
        .groupby(well_col, sort=False)
        .first()
        .reset_index()[[well_col, x_col, y_col]]
    )
