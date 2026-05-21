from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from typing import Optional

from src.data.loader import CORE_LOG_CURVES


class WellLogDataset(Dataset):
    """
    Sliding-window Dataset for per-depth lithology classification.

    Each sample is a fixed-length window of depth intervals from one well.
    Wells shorter than window_size are zero-padded; padding_mask flags those
    positions as True so the Transformer can ignore them via src_key_padding_mask.

    Args:
        df: DataFrame from loader.load_all_wells — must contain feature_cols,
            depth_col, label_col, and well_col.
        feature_cols: ordered list of log curve column names.
        window_size: depth intervals per window.
        stride: step between windows within a well. Ignored for short wells.
        depth_col: measured depth column — returned as-is for positional encoding.
        label_col: integer lithology class index column.
        well_col: column that groups rows by well.
        stats: {'mean': pd.Series, 'std': pd.Series} for z-score normalisation.
               If None, computed from df — always pass training stats to val/test.
    """

    def __init__(
        self,
        df: pd.DataFrame,
        feature_cols: list[str] = CORE_LOG_CURVES,
        window_size: int = 128,
        stride: int = 64,
        depth_col: str = "DEPTH_MD",
        label_col: str = "LITHOLOGY_IDX",
        well_col: str = "WELL",
        stats: Optional[dict] = None,
    ) -> None:
        self.feature_cols = feature_cols
        self.window_size = window_size
        self.n_features = len(feature_cols)

        if stats is None:
            stats = _compute_stats(df, feature_cols)
        self.stats = stats

        self._features: list[torch.Tensor] = []
        self._labels: list[torch.Tensor] = []
        self._depths: list[torch.Tensor] = []
        self._padding_masks: list[torch.Tensor] = []

        for _, well_df in df.groupby(well_col, sort=False):
            well_df = well_df.sort_values(depth_col).reset_index(drop=True)

            raw = well_df[feature_cols].ffill().bfill().fillna(0.0)
            normed = ((raw - stats["mean"]) / stats["std"]).values.astype(np.float32)
            labels = well_df[label_col].values.astype(np.int64)
            depths = well_df[depth_col].values.astype(np.float32)
            n = len(well_df)

            if n <= window_size:
                self._append_padded(normed, labels, depths, n)
            else:
                for s in range(0, n - window_size + 1, stride):
                    e = s + window_size
                    self._features.append(torch.from_numpy(normed[s:e]))
                    self._labels.append(torch.from_numpy(labels[s:e]))
                    self._depths.append(torch.from_numpy(depths[s:e]))
                    self._padding_masks.append(torch.zeros(window_size, dtype=torch.bool))

    def _append_padded(
        self,
        normed: np.ndarray,
        labels: np.ndarray,
        depths: np.ndarray,
        n: int,
    ) -> None:
        pad = self.window_size - n
        self._features.append(
            torch.from_numpy(np.concatenate([normed, np.zeros((pad, self.n_features), dtype=np.float32)]))
        )
        self._labels.append(
            torch.from_numpy(np.concatenate([labels, np.zeros(pad, dtype=np.int64)]))
        )
        self._depths.append(
            torch.from_numpy(np.concatenate([depths, np.zeros(pad, dtype=np.float32)]))
        )
        mask = torch.zeros(self.window_size, dtype=torch.bool)
        mask[n:] = True
        self._padding_masks.append(mask)

    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, idx: int) -> dict:
        return {
            "features": self._features[idx],        # [window_size, n_features]
            "labels": self._labels[idx],             # [window_size]  int64
            "depths": self._depths[idx],             # [window_size]  float32 metres
            "padding_mask": self._padding_masks[idx],# [window_size]  bool, True=pad
        }

    @classmethod
    def train_val_split(
        cls,
        df: pd.DataFrame,
        train_wells: list[str],
        val_wells: list[str],
        well_col: str = "WELL",
        **kwargs,
    ) -> tuple[WellLogDataset, WellLogDataset]:
        """
        Build train and val datasets from disjoint well lists.
        Normalization stats are fitted on training wells only and shared with val.
        """
        train_df = df[df[well_col].isin(train_wells)]
        val_df = df[df[well_col].isin(val_wells)]
        train_ds = cls(train_df, well_col=well_col, **kwargs)
        val_ds = cls(val_df, well_col=well_col, stats=train_ds.stats, **kwargs)
        return train_ds, val_ds


def _compute_stats(df: pd.DataFrame, feature_cols: list[str]) -> dict:
    mean = df[feature_cols].mean()
    std = df[feature_cols].std().replace(0.0, 1.0)
    return {"mean": mean, "std": std}
