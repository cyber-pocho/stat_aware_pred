# Statistically-Aware Lithology Prediction from Well Logs

> **Status:** Work in Progress — data loading and EDA complete; feature engineering and model training in progress.

A deep learning framework for lithology classification from wireline well log data, incorporating geostatistical awareness (spatial correlation, kriging-based features) to improve prediction robustness across the Norwegian Continental Shelf.

Built on the [FORCE 2020 Machine Learning Competition](https://github.com/bolgebrygg/Force-2021-Machine-Learning-competition) dataset.

---

## Overview

Lithology prediction from well logs is a core task in subsurface characterization. Standard approaches treat each depth sample independently, ignoring spatial structure and inter-well correlation. This project explores whether statistically-informed features (derived from geostatistics and spatial modeling) can improve classification accuracy — particularly in data-sparse zones and across heterogeneous formations.

**Target classes:**

| Code   | Lithology            |
|--------|----------------------|
| 30000  | Sandstone            |
| 65000  | Shale                |
| 65030  | Shale with sand      |
| 70000  | Limestone            |
| 70032  | Limestone with clay  |
| 80000  | Chalk                |
| 90000  | Halite / Anhydrite   |

---

## Dataset

**FORCE 2020 Well Log Dataset** — a multi-well dataset from the Norwegian North Sea with labeled lithofacies.

**Log curves used:**

| Curve  | Description              | Curve  | Description           |
|--------|--------------------------|--------|-----------------------|
| GR     | Gamma Ray                | DTC    | Compressional Sonic   |
| RDEP   | Deep Resistivity         | DTS    | Shear Sonic           |
| RMED   | Medium Resistivity       | PEF    | Photoelectric Factor  |
| RHOB   | Bulk Density             | CALI   | Caliper               |
| NPHI   | Neutron Porosity         | DCAL   | Differential Caliper  |
| DRHO   | Density Correction       | BS     | Bit Size              |
| ROP    | Rate of Penetration      | ROPA   | Average ROP           |

Raw data lives in `data/force2020_full/` (one CSV per well, not committed to this repo).

---

## Project Structure

```
stat_aware_pred/
├── data/
│   └── force2020_full/     # Per-well CSV files (not tracked by git)
├── nbs/
│   └── nb.ipynb            # EDA notebook
├── src/
│   └── data/
│       ├── __init__.py
│       ├── loader.py       # Well loading, normalization, lithology mapping
│       └── features.py     # Feature engineering pipeline (WIP)
└── requirements.txt
```

---

## Installation

```bash
git clone <repo-url>
cd stat_aware_pred
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, CUDA-capable GPU recommended.

Key dependencies: `torch`, `pandas`, `numpy`, `scikit-learn`, `pykrige`, `lasio`, `plotly`.

---

## Usage

### Load all labeled wells

```python
from src.data.loader import load_all_wells

df = load_all_wells("data/force2020_full", verbose=True)
# -> Combined DataFrame: all labeled wells sorted by well then depth
```

### Load a single well

```python
from src.data.loader import load_single_well

df = load_single_well("data/force2020_full/15_9-23.csv")
```

### Get labeled well list

```python
from src.data.loader import get_well_list

wells = get_well_list("data/force2020_full")
```

---

## Roadmap

- [x] Data loading pipeline (`src/data/loader.py`)
- [x] Schema normalization for anomalous wells
- [x] EDA notebook (distributions, correlation matrix, scatter matrix)
- [ ] Feature engineering — log-derived features, spatial coords (`src/data/features.py`)
- [ ] Geostatistical feature extraction (variogram, kriging interpolation via `pykrige`)
- [ ] Baseline models (XGBoost, Random Forest)
- [ ] Deep learning model (1D CNN / Transformer on depth sequences)
- [ ] Statistically-aware model variant
- [ ] Evaluation and comparison

---

## Hardware

Developed on:
- **GPU:** NVIDIA GeForce RTX 3050 6GB (Laptop)
- **CUDA:** 13.0 / PyTorch 2.11.0+cu130
