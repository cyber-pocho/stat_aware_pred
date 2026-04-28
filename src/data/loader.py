import os
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

LITHOLOGY_MAP={
    30000:"Sandstone",
    65000:"Shale",
    65030:"Shale with sand",
    70000:"Limestone",
    70032:"Limestone with clay",
    80000:"Chalk",
    90000:"Halite",
    90000:"Anhydrite",
}

LITHOLOGY_TO_IDX={code: i for i, code in enumerate(sorted(LITHOLOGY_MAP.keys()))}
IDX_TO_LITHOLOGY={i: code for code, i in LITHOLOGY_TO_IDX.items()}
IDX_TO_NAME={i:LITHOLOGY_MAP[code] for code, i in LITHOLOGY_TO_IDX.items()}
N_CLASSES=len(LITHOLOGY_MAP)

CORE_LOG_CURVES=[
    "GR", # Gamma Ray
    "RDEP", # Deep resistivity
    "RMED", # Medium resistivity
    "RHOB", # Bulk density
    "NPHI", # Neutron porosity
    "DTC", # Compressional Sonic
    "DTS", # Shear Sonic
    "PEF", # Photoelectric factor
    "CALI", # Caliper
    "DCAL", # Differential correction
    "DRHO", # Density correction
    "BS", # Bit size
    "ROP", # Rate of penetration
    "ROPA", # Average ROP
]

SPATIAL_COLS=["X_LOC", "Y_LOC", "Z_LOC", "DEPTH_MD"]
META_COLS=["WELL","GROUP","FORMATION"]
LABEL_COLS=["FORCE_2020_LITHOFACIES_LITHOLOGY", "FORCE_2020_LITHOFACIES_CONFIDENCE"]

def normalize_anomalous(df:pd.DataFrame, filepath:str)->pd.DataFrame:
    """
    Normalize two anomalous wells (31_6-5, 31_6-8) to match
    the standard schema. These files have:
        - An unamed: 0 index column.
        - DEPT as duplicate depth column
        - Missing GROUP, FORMATION, DTS, ROP, DCAL, ROPA, etc.
    """
    drop_cols=[c for c in ["Unnamed: 0", "DEPT"] if c in df.columns]
    df=df.drop(columns=drop_cols)

    for col in ["GROUP","FORMATION"]:
        if col not in df.columns:
            df[col]=""
    for col in CORE_LOG_CURVES + SPATIAL_COLS:
        if col not in df.columns:
            df[col]=np.nan
    if "WELL" not in df.columns:
        df["WELL"]=Path(filepath).stem

    return df
