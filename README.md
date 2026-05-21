# Stratigraphic-Aware Grade Prediction with Uncertainty Quantification

**Author:** Julian Alfonso  
**Collaborator:** [Roommate name], Geologist — stratigraphy and subsurface interpretation  
**Status:** In development — data pipeline and dataset complete; model stubs in place; training not yet run  
**Target application:** Mineral exploration — direct methodological contribution to multi-source subsurface prediction problems

---

## Problem Statement

Predicting the grade (concentration) of a mineral at undrilled locations is the central problem of mineral exploration. The industry standard is geostatistical interpolation — kriging, inverse distance weighting, radial basis functions. These methods share a fundamental limitation: they assume spatial continuity without geological context. A grade value 50 meters away is treated as informative regardless of whether it sits in the same rock unit or a completely different one separated by a fault or unconformity.

This project addresses that limitation directly. Mineral deposits are not spatially random. They are controlled by geology — by which rock units are present, how those units are arranged in stratigraphic sequence, and how deformation has modified that sequence. A model that does not know which rock unit it is predicting inside is ignoring the single most important categorical variable in the problem.

The approach here is a two-stage hierarchical model:

1. Classify the lithology (rock type) at each depth interval using geophysical well log curves.
2. Use the predicted lithology as a conditioning signal for grade prediction at undrilled locations.

The second model knows what it is predicting inside. This is stratigraphic conditioning.

---

## Why This Matters for Mineflow

Mineflow's documented architecture fuses multiple data types simultaneously — drill assays, geophysics, geochemistry, structural data — and explicitly identifies the failure of traditional methods to capture non-linear, multi-variable relationships. The stratigraphic conditioning layer proposed here is a direct instantiation of that design philosophy: use geological structure (stratigraphy) as an organizing prior that constrains and informs the grade prediction.

Concretely, this project contributes to two open problems visible in Mineflow's prediction pipeline:

**Multi-modal heterogeneous fusion.** Drill hole data (1D depth sequences) and surface geophysics (2D spatial grids) have different spatial resolutions, different units, and different coverage. The cross-attention mechanism implemented here provides one principled approach to fusing them: depth sequences attend to spatial context, and spatial context attends to the most geologically coherent depth intervals.

**Uncertainty quantification on sparse data.** Drill data is extremely sparse. A typical exploration program covers tens of square kilometers with fewer than 100 drill holes. Standard deep learning uncertainty methods (MC Dropout, deep ensembles) are poorly calibrated in this regime. This project implements and evaluates both, reports calibration explicitly, and compares against the kriging variance as a baseline.

Additionally, this project uses Western Australian public drill data — directly relevant to Mineflow's Australian Tenements feature launched in April 2026.

---

## Data

### Primary: FORCE 2020 Well Log Lithology Dataset

Source: Bormann et al. (2020), released via Kaggle  
URL: https://www.kaggle.com/datasets/ala2obstantsmachines/force-well-logs

The FORCE 2020 dataset contains Norwegian petroleum wells with full petrophysical log suites and expert-labeled lithology at 0.1524 m (0.5 ft) depth intervals. It was released as a machine learning competition benchmark, which means published baseline results exist for direct comparison.

**Log curves used:**

| Curve | Description               | Curve | Description              |
|-------|---------------------------|-------|--------------------------|
| GR    | Gamma Ray                 | DTC   | Compressional Sonic      |
| RDEP  | Deep Resistivity          | DTS   | Shear Sonic              |
| RMED  | Medium Resistivity        | PEF   | Photoelectric Factor     |
| RHOB  | Bulk Density              | CALI  | Caliper                  |
| NPHI  | Neutron Porosity          | DCAL  | Differential Caliper     |
| DRHO  | Density Correction        | BS    | Bit Size                 |
| ROP   | Rate of Penetration       | ROPA  | Average ROP              |

**Derived features** (computed in `src/data/features.py`):

| Feature      | Formula          | Geological use                                    |
|--------------|------------------|---------------------------------------------------|
| VP_VS_RATIO  | DTC / DTS        | Separates fluid types and lithologies             |
| ND_XPLOT     | NPHI − RHOB      | Isolates porosity from lithology on neutron-density crossplot |
| RD_RM_RATIO  | RDEP / RMED      | Invasion indicator; flags permeable beds          |
| GR_NORM      | per-well [0, 1]  | Removes tool calibration drift between wells      |

**Lithology classes** (11 codes verified in the dataset):

| Code  | Lithology           |
|-------|---------------------|
| 30000 | Sandstone           |
| 65000 | Shale               |
| 65030 | Shale with sand     |
| 70000 | Limestone           |
| 70032 | Limestone with clay |
| 74000 | Chalk               |
| 80000 | Marl                |
| 86000 | Halite              |
| 88000 | Anhydrite           |
| 90000 | Tuff                |
| 99000 | Coal                |

The dataset has known class imbalance — shale dominates. Handling this explicitly (focal loss) is part of the methodology.

### Secondary: GSWA Open Drillhole Database

Source: Geological Survey of Western Australia  
URL: https://dasc.dmp.wa.gov.au/dasc/

Public mineral exploration drill holes from Western Australia with assay data (gold, copper, nickel, zinc in ppm/ppb), collar coordinates, and lithology logs where available. Used for the spatial grade prediction component and uncertainty evaluation on real sparse mineral exploration data.

---

## Model Architecture

### Stage 1: Lithology Classifier

Input: a depth sequence of log curve values, shape `[depth_intervals, n_curves]`  
Output: per-interval probability distribution over 11 lithology classes

Architecture: 1D Transformer encoder

```
Input [L, C]
  -> Linear projection to d_model
  -> Depth-aware positional encoding (actual depth in metres, not sequence index)
  -> N x TransformerEncoderLayer (self-attention + feedforward)
  -> Per-position classification head [L, n_classes]
```

Loss: Focal loss (`gamma=2.0` default) to handle class imbalance.  
Baseline: FORCE 2020 Kaggle winning solution (gradient boosted trees on engineered features).

### Stage 2: Grade Prediction with Stratigraphic Conditioning

Input:
- Drill hole assay sequence: `[depth_intervals, n_elements]`
- Lithology probabilities from Stage 1: `[depth_intervals, n_classes]`
- Collar spatial coordinates: `[x, y, z]`
- Surface geophysical features: `[n_geophys_features]`

Output: predicted grade + uncertainty estimate (std or 90% credible interval)

Architecture: Cross-attention fusion network

```
Depth branch:
  assay values + lithology probabilities -> 1D Transformer -> [L, d_model]

Spatial branch:
  collar (x,y,z) + surface geophysics -> MLP -> [d_model]

Fusion:
  cross-attention: spatial context queries depth sequence -> [d_model]
  attention weights indicate which stratigraphic intervals are most predictive

Output head:
  MLP -> predicted grade
  MC Dropout (inference) or Deep Ensemble -> uncertainty
```

### Uncertainty Quantification

Two methods implemented and compared:

- **Deep Ensembles** (primary): five independently trained models; variance across members is the uncertainty estimate. Better calibrated than MC Dropout in low-data regions.
- **MC Dropout**: T=50 stochastic forward passes at inference. Cheaper but known to underestimate uncertainty far from training distribution.

Both are evaluated against kriging variance as a non-neural baseline. Calibration is reported explicitly via reliability diagrams and Expected Calibration Error (ECE).

---

## Evaluation

### Stage 1

- Macro F1 across all 11 lithology classes (primary — penalises poor performance on minority classes equally)
- Per-class F1
- Comparison against FORCE 2020 Kaggle winning solution
- Confusion matrix with geological interpretation of plausible vs implausible misclassifications

### Stage 2

- RMSE and MAE on held-out drill holes (spatial cross-validation — held-out holes are geographically separated from training holes)
- Comparison against ordinary kriging
- Calibration curves for uncertainty estimates
- Attention weight visualisation over depth intervals

---

## Repository Structure

```
stat_aware_pred/
├── data/
│   ├── force2020/          # FORCE 2020 per-well CSVs (not committed — 21 wells)
│   └── gswa/               # Western Australian drill data (not yet downloaded)
├── notebooks/
│   └── 01_eda.ipynb        # EDA — log curve distributions, class balance, correlation
├── src/
│   ├── data/
│   │   ├── loader.py       # Well loading, schema normalisation, lithology mapping
│   │   ├── features.py     # Derived log features (Vp/Vs, ND crossplot, GR norm)
│   │   ├── dataset.py      # WellLogDataset — sliding-window PyTorch Dataset
│   │   └── splits.py       # Spatial cross-validation split generation [stub]
│   ├── models/
│   │   ├── transformer.py  # Depth-sequence Transformer encoder [stub]
│   │   ├── fusion.py       # Cross-attention fusion network [stub]
│   │   ├── heads.py        # Classification and regression output heads
│   │   └── uncertainty.py  # MC Dropout and deep ensemble wrappers [stub]
│   ├── train/
│   │   ├── stage1.py       # Lithology classifier training loop [stub]
│   │   └── stage2.py       # Grade prediction training loop [stub]
│   └── eval/
│       ├── metrics.py      # F1, RMSE, MAE, ECE, reliability diagram [stub]
│       └── visualize.py    # Log track plots, attention heatmaps, grade maps [stub]
├── configs/
│   ├── stage1.yaml         # Hyperparameters for lithology classifier
│   └── stage2.yaml         # Hyperparameters for grade prediction model
├── requirements.txt
└── README.md
```

Files marked `[stub]` have their signatures and docstrings in place but are not yet implemented.

---

## Implementation Status

| Component               | Status      |
|-------------------------|-------------|
| Data loader             | Complete    |
| Derived feature engineering | Complete |
| PyTorch Dataset (sliding window) | Complete |
| Spatial CV splits       | Stub        |
| Stage 1 Transformer     | Stub        |
| Stage 2 fusion network  | Stub        |
| Output heads            | Complete    |
| Uncertainty wrappers    | Stub        |
| Training loops          | Stub        |
| Metrics + calibration   | Stub        |
| EDA notebook            | Complete    |
| Baseline notebook       | Not started |

---

## Geological Contributions

This project is a collaboration between a physics/ML practitioner and a geologist specialising in stratigraphy. The geological contributions are:

- Interpretation of which log curve combinations are diagnostic of each lithology class
- Review of Stage 1 confusion matrices for geological plausibility
- Identification of stratigraphic sequences in the FORCE 2020 wells
- Assessment of whether cross-attention weights correspond to geologically meaningful intervals
- Ground-truth validation of grade predictions against known geological controls in the Western Australian dataset

---

## Dependencies

```
torch>=2.0
numpy
pandas
scipy
scikit-learn
lasio
matplotlib
plotly
pykrige
tqdm
pyyaml
```

**Hardware used:** NVIDIA GeForce RTX 3050 6 GB (Laptop), CUDA 13.0, PyTorch 2.11.0+cu130

---

## Reproduction

```bash
git clone https://github.com/cyber-pocho/stratigraphic-grade-prediction
cd stat_aware_pred
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Place FORCE 2020 per-well CSVs in data/force2020/
# (download from https://www.kaggle.com/datasets/ala2obstantsmachines/force-well-logs)

# Run training (once stubs are implemented)
python src/train/stage1.py --config configs/stage1.yaml
python src/train/stage2.py --config configs/stage2.yaml
```

---

## What This Is Not

This project does not claim to reproduce or replicate Mineflow's internal models. It assumes clean, structured input data and focuses exclusively on the modelling and uncertainty quantification components. The FORCE 2020 dataset is petroleum well logs, not mineral exploration drill holes — the lithology classification results on FORCE 2020 establish methodological validity; the Western Australian data is where grade prediction is tested in a regime closer to Mineflow's actual use case.

---

## References

Bormann, P. et al. (2020). FORCE Machine Learning Competition: Well Log and Lithofacies Dataset. Zenodo.

Grana, D., Fjeldstad, T., and Omre, H. (2017). Bayesian Gaussian Mixture Linear Inversion for Geophysical Inverse Problems. Mathematical Geosciences.

Lakshminarayanan, B., Pritzel, A., and Blundell, C. (2017). Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles. NeurIPS.

Lin, T.Y. et al. (2017). Focal Loss for Dense Object Detection. ICCV.

Vaswani, A. et al. (2017). Attention Is All You Need. NeurIPS.
