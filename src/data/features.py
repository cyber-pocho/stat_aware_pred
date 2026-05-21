"""Log curve preprocessing, normalization, and derived petrophysical features."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived log curve features and append them as new columns.

    Derived features:
        VP_VS_RATIO   — DTC / DTS; separates fluid types and lithologies.
        ND_XPLOT      — NPHI - RHOB crossplot proxy; isolates porosity from
                        lithology effect on the neutron-density crossplot.
        RD_RM_RATIO   — RDEP / RMED; invasion indicator; deviations from 1
                        flag filtrate invasion and permeable beds.
        GR_NORM       — GR normalised to [0, 1] per-well; removes tool
                        calibration drift between wells.
    """
    out = df.copy()

    if {"DTC", "DTS"}.issubset(df.columns):
        out["VP_VS_RATIO"] = df["DTC"] / df["DTS"].replace(0, np.nan)

    if {"NPHI", "RHOB"}.issubset(df.columns):
        out["ND_XPLOT"] = df["NPHI"] - df["RHOB"]

    if {"RDEP", "RMED"}.issubset(df.columns):
        out["RD_RM_RATIO"] = df["RDEP"] / df["RMED"].replace(0, np.nan)

    if "GR" in df.columns and "WELL" in df.columns:
        out["GR_NORM"] = df.groupby("WELL")["GR"].transform(
            lambda s: (s - s.min()) / (s.max() - s.min() + 1e-9)
        )

    return out


DERIVED_FEATURES = ["VP_VS_RATIO", "ND_XPLOT", "RD_RM_RATIO", "GR_NORM"]
