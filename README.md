# Stratigraphic-Aware Grade Prediction with Uncertainty Quantification

**Author:** Julian Alfonso  
**Collaborator:** [Roommate name], Geologist — stratigraphy and subsurface interpretation

---

## Overview

A two-stage deep learning framework for mineral grade prediction that uses geological structure as a conditioning signal. Standard geostatistical interpolation (kriging, IDW) assumes spatial continuity without geological context. This project replaces that assumption with an explicit stratigraphic prior: lithology is classified first, then used to condition grade prediction at undrilled locations.

**Stage 1 — Lithology Classifier**  
A 1D Transformer encoder maps a depth sequence of wireline log curves to per-interval lithology probabilities. Depth-aware positional encoding preserves the physical meaning of depth (a 10 m interval at 500 m is geologically distinct from the same interval at 2000 m). Focal loss handles the severe class imbalance characteristic of well log datasets.

**Stage 2 — Grade Predictor with Stratigraphic Conditioning**  
A cross-attention fusion network combines a depth sequence of assay values (conditioned on Stage 1 lithology probabilities) with a spatial context vector (collar coordinates + surface geophysics). The spatial branch queries the depth sequence via cross-attention, producing interpretable attention weights that identify which stratigraphic intervals are most predictive at each surface location. Uncertainty is quantified via deep ensembles and MC Dropout, evaluated against kriging variance as a non-neural baseline.

---

## Dataset

### FORCE 2020 — Stage 1

Norwegian petroleum wells with full petrophysical log suites and expert-labeled lithology at 0.1524 m intervals. Released as a Kaggle competition benchmark with published baselines for direct comparison.

**Log curves:**

| Curve | Description              | Curve | Description           |
|-------|--------------------------|-------|-----------------------|
| GR    | Gamma Ray                | DTC   | Compressional Sonic   |
| RDEP  | Deep Resistivity         | DTS   | Shear Sonic           |
| RMED  | Medium Resistivity       | PEF   | Photoelectric Factor  |
| RHOB  | Bulk Density             | CALI  | Caliper               |
| NPHI  | Neutron Porosity         | DCAL  | Differential Caliper  |
| DRHO  | Density Correction       | BS    | Bit Size              |
| ROP   | Rate of Penetration      | ROPA  | Average ROP           |

**Derived features** (computed in `src/data/features.py`):

| Feature     | Formula        | Purpose                                                |
|-------------|----------------|--------------------------------------------------------|
| VP_VS_RATIO | DTC / DTS      | Separates fluid types and lithologies                  |
| ND_XPLOT    | NPHI − RHOB    | Isolates lithology effect on neutron-density crossplot |
| RD_RM_RATIO | RDEP / RMED    | Invasion indicator; flags permeable beds               |
| GR_NORM     | per-well [0,1] | Removes inter-well tool calibration drift              |

**Lithology classes** (11 codes verified in dataset):

| Code  | Class               | Code  | Class               |
|-------|---------------------|-------|---------------------|
| 30000 | Sandstone           | 86000 | Halite              |
| 65000 | Shale               | 88000 | Anhydrite           |
| 65030 | Shale with sand     | 90000 | Tuff                |
| 70000 | Limestone           | 99000 | Coal                |
| 70032 | Limestone with clay | 74000 | Chalk               |
| 80000 | Marl                |       |                     |

### GSWA Open Drillhole Database — Stage 2

Western Australian public mineral exploration drill holes with assay data (Au, Cu, Ni, Zn), collar coordinates, and lithology logs. Used for grade prediction and uncertainty evaluation in a sparse-data regime representative of real exploration programs.

---

## Architecture

### Stage 1: Lithology Classifier

```
Input  [batch, L, n_curves]
  └─ Linear projection ──────────────────────────────── [batch, L, d_model]
  └─ Depth-aware positional encoding
  └─ N × TransformerEncoderLayer (self-attn + FFN)
  └─ Classification head ────────────────────────────── [batch, L, n_classes]
```

### Stage 2: Cross-Attention Fusion

```
Depth branch
  assay + lith_probs  [batch, L, n_assay + n_classes]
  └─ Linear projection + Transformer encoder ─────────── [batch, L, d_model]

Spatial branch
  collar (x,y,z) + surface geophysics  [batch, n_spatial]
  └─ MLP encoder ─────────────────────────────────────── [batch, d_model]

Fusion
  └─ Cross-attention: spatial queries depth ──────────── [batch, d_model]
  └─ Regression head ─────────────────────────────────── [batch, 1]  (grade)
  └─ Uncertainty: deep ensemble or MC Dropout
```

---

## Repository Structure

```
stat_aware_pred/
├── data/
│   ├── force2020/          # FORCE 2020 per-well CSVs (not committed)
│   └── gswa/               # GSWA drill holes (not committed)
├── notebooks/
│   └── 01_eda.ipynb        # Exploratory data analysis
├── src/
│   ├── data/
│   │   ├── loader.py       # Well loading and lithology mapping
│   │   ├── features.py     # Derived petrophysical features
│   │   ├── dataset.py      # Sliding-window PyTorch Dataset
│   │   └── splits.py       # Spatial cross-validation splits
│   ├── models/
│   │   ├── transformer.py  # Depth-sequence Transformer encoder
│   │   ├── fusion.py       # Cross-attention fusion network
│   │   ├── heads.py        # Classification and regression heads
│   │   └── uncertainty.py  # MC Dropout and deep ensemble wrappers
│   ├── train/
│   │   ├── stage1.py       # Lithology classifier training loop
│   │   └── stage2.py       # Grade prediction training loop
│   └── eval/
│       ├── metrics.py      # F1, RMSE, ECE, reliability diagrams
│       └── visualize.py    # Log tracks, attention heatmaps, grade maps
├── configs/
│   ├── stage1.yaml         # Stage 1 hyperparameters
│   └── stage2.yaml         # Stage 2 hyperparameters
├── requirements.txt
└── README.md
```

---

## Quickstart

```bash
git clone https://github.com/cyber-pocho/stratigraphic-grade-prediction
cd stat_aware_pred
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Place FORCE 2020 per-well CSVs in `data/force2020/`.

**Train Stage 1:**
```bash
python src/train/stage1.py --config configs/stage1.yaml
```

**Evaluate Stage 1:**
```bash
python src/eval/run_eval.py --config configs/stage1.yaml --checkpoint checkpoints/stage1/best.pt
```

**Train Stage 2** (requires Stage 1 checkpoint):
```bash
python src/train/stage2.py --config configs/stage2.yaml
```

---

## Evaluation

**Stage 1 — Lithology Classification**
- Primary metric: macro F1 across all 11 classes (penalises minority class failures equally)
- Per-class F1 with geological interpretation of confusion patterns
- Comparison against FORCE 2020 Kaggle winning solution (gradient-boosted trees)
- Spatial cross-validation: held-out wells are geographically isolated from training wells

**Stage 2 — Grade Prediction**
- RMSE and MAE on spatially held-out drill holes
- Comparison against ordinary kriging on the same data
- Calibration: reliability diagrams and Expected Calibration Error (ECE)
- Attention weight visualisation over depth intervals

---

## Geological Contributions

The geological collaborator provides:

- Identification of which log curve combinations are diagnostic for each lithology
- Review of Stage 1 confusion matrices for geological plausibility (e.g. shale/tight carbonate confusion is acceptable; coal/anhydrite is not)
- Annotation of stratigraphic sequence boundaries in FORCE 2020 wells
- Validation of Stage 2 attention weights against known geological controls
- Assessment of grade predictions against established mineralisation models in the GSWA dataset

---

## Dependencies

```
torch>=2.0   numpy   pandas   scipy   scikit-learn
lasio   matplotlib   plotly   pykrige   tqdm   pyyaml
```

Hardware: NVIDIA GeForce RTX 3050 6 GB · CUDA 13.0 · PyTorch 2.11.0+cu130

---

## References

Bormann et al. (2020). *FORCE Machine Learning Competition: Well Log and Lithofacies Dataset.* Zenodo.  
Lakshminarayanan et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.  
Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.  
Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
