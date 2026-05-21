
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
    74000:"Chalk",
    80000:"Marl",
    86000:"Halite",
    88000:"Anhydrite",
    90000:"Tuff",
    99000:"Coal",
}

LITHOLOGY_TO_IDX={code: i for i, code in enumerate(sorted(LITHOLOGY_MAP.keys()))}
IDX_TO_LITHOLOGY={i: code for code, i in LITHOLOGY_TO_IDX.items()}
IDX_TO_NAME={i:LITHOLOGY_MAP[code] for code, i in LITHOLOGY_TO_IDX.items()}
N_CLASSES=len(LITHOLOGY_MAP)

CORE_LOG_CURVES=[
    "GR", # Gamma Ray - natural emission of radiation of the rock. 
    "RDEP", # Deep resistivity - determines if there's crude oil or water. 
    "RMED", # Medium resistivity - mixture of perforation mud and formation fluid (crude oil or water). 
    "RHOB", # Bulk density - density of the rock. 
    "NPHI", # Neutron porosity - Electronic density ~ rock density. 
    "DTC", # Compressional Sonic - 
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

def _normalize_anomalous(df:pd.DataFrame, filepath:str)->pd.DataFrame:
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
def load_single_well(filepath:str)-> Optional[pd.DataFrame]: 
    """
    Load a single well CSV file
    Returns None if the file has no lithology LABEL_COLS
    Returns a normalized DataFrame with consistent columns otherwise. 
    """
    df=pd.read_csv(filepath)

    if "FORCE_2020_LITHOFACIES_LITHOLOGY" not in df.columns:
        return None
    fname=Path(filepath).name 
    if fname in {"31_6-5.csv", "31_6-8.csv"}:
        df=_normalize_anomalous(df, filepath)
    keep=META_COLS+SPATIAL_COLS+CORE_LOG_CURVES+LABEL_COLS
    keep=[c for c in keep if c in df.columns]
    df=df[keep].copy()
    df["LITHOLOGY_IDX"]=df["FORCE_2020_LITHOFACIES_LITHOLOGY"].map(LITHOLOGY_TO_IDX)

    unknown_mask=df["LITHOLOGY_IDX"].isna()
    if unknown_mask.sum()>0:
        print(f"  [{fname}] Dropping {unknown_mask.sum()} rows with unknown lithology codes: "
              f"{df.loc[unknown_mask, 'FORCE_2020_LITHOFACIES_LITHOLOGY'].unique()}")
        df = df[~unknown_mask].copy()
 
    df["LITHOLOGY_IDX"] = df["LITHOLOGY_IDX"].astype(int)
    df = df.sort_values("DEPTH_MD").reset_index(drop=True)
 
    return df
def load_all_wells(data_dir: str, verbose:bool=True)->pd.DataFrame: 
    """
    Load all labeled wells from data_dir into a single DataFrame. 
    Parameters: 
        data_dir: str, # Directory containing per-well CSV files. 
        verbose: bool # Print per-well loading stats. 
    Returns: 
        pd.DataFrame #Combined DataFrame with all labeled wells, sorted by well then depth. 
    """
    data_dir=Path(data_dir)
    files=sorted(data_dir.glob("*.csv"))

    dfs=[]
    skipped=[]
    for f in files: 
        df=load_single_well(str(f))
        if df is None: 
            skipped.append(f.name)
            continue
        if verbose:
            lith_counts=df["LITHOLOGY_IDX"].value_counts().sort_index()
            lith_str=", ".join(
                    f"{IDX_TO_NAME[i]}={n}" for i, n in lith_counts.items()
                    )
            null_pct=df[CORE_LOG_CURVES].isnull().mean().mean()*100
            print(f"Loaded {f.name:.35s}, {len(df):6d} rows", f"null={null_pct:.1f}, {lith_str}")
        dfs.append(df)
    if verbose: 
        print(f"\nSkipped (no labels): {skipped}")
    if not dfs: 
        raise RuntimeError(f"No labeled wells found in {data_dir}")
    combined=pd.concat(dfs, ignore_index=True)
    if verbose:
        print(f"\nTotal labeled rows : {len(combined)}")
        print(f"Total labeled wells: {combined['WELL'].nunique()}")
        print(f"\nGlobal lithology distribution:")
        for idx, count in combined["LITHOLOGY_IDX"].value_counts().sort_index().items():
            pct = count / len(combined) * 100
            print(f"  {IDX_TO_NAME[idx]:25s}: {count:6d} ({pct:.1f}%)")
 
    return combined
def get_well_list(data_dir: str) -> list: 
    """
    Return sorted list of labeled well names. 
    """
    df=load_all_wells(data_dir, verbose=False)
    return sorted(df["WELL"].unique().tolist())

if __name__=="__main__": 
    df=load_all_wells("data/force2020_full", verbose=True)
    print("Columns: ", df.columns.tolist())

