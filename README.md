# Stratigraphic-Aware Lithology Classification

A 1D Transformer that classifies rock type at every depth interval in a wireline well log, using self-attention to capture stratigraphic context across the full depth sequence.

Built on the [FORCE 2020](https://github.com/bolgebrygg/Force-2021-Machine-Learning-competition) Norwegian well log benchmark.

---

## Motivation

Standard lithology classifiers treat each depth sample independently. A shale layer sitting above a sandstone is a meaningful stratigraphic relationship — but a per-sample classifier ignores it. Self-attention lets the model learn these inter-depth dependencies directly from data, without manually encoding geological rules.

A second contribution is uncertainty-aware grade prediction: lithology predictions from Stage 1 condition a cross-attention fusion network that predicts mineral grade at undrilled locations, with calibrated uncertainty estimates from deep ensembles.

---

## Model

### Stage 1 — Lithology Classifier

```
Well log curves  [batch, depth, n_curves]
    │
    ├─ Linear projection → d_model
    ├─ Depth-aware positional encoding   (actual metres, not sequence index)
    └─ Transformer encoder (N layers, multi-head self-attention)
                │
    Per-interval classification head → [batch, depth, n_classes]
```

Focal loss handles the severe class imbalance (shale dominates most wells).

### Stage 2 — Grade Predictor

```
Depth branch:  assay values + Stage 1 lith probs → Transformer → [batch, L, d]
Spatial branch: collar (x,y,z) + surface geophysics → MLP → [batch, d]
                                    │
                    Cross-attention: spatial queries depth
                                    │
                    Regression head → predicted grade + uncertainty
```

Uncertainty is estimated via deep ensembles (5 members) and benchmarked against kriging variance.

---

## Dataset

**Stage 1 — [FORCE 2020](https://www.kaggle.com/datasets/ala2obstantsmachines/force-well-logs)**  
Norwegian petroleum wells with expert-labeled lithofacies at 0.1524 m intervals.

| Curve | Description           | Curve | Description          |
|-------|-----------------------|-------|----------------------|
| GR    | Gamma Ray             | DTC   | Compressional Sonic  |
| RDEP  | Deep Resistivity      | DTS   | Shear Sonic          |
| RMED  | Medium Resistivity    | PEF   | Photoelectric Factor |
| RHOB  | Bulk Density          | CALI  | Caliper              |
| NPHI  | Neutron Porosity      | DRHO  | Density Correction   |

Four derived features are added: Vp/Vs ratio, neutron-density crossplot, resistivity invasion ratio, and per-well normalised GR.

**Lithology classes (11):** Sandstone · Shale · Shale with sand · Limestone · Limestone with clay · Chalk · Marl · Halite · Anhydrite · Tuff · Coal

**Stage 2 — [GSWA Open Drillhole Database](https://dasc.dmp.wa.gov.au/dasc/)**  
Western Australian public mineral exploration drill holes with Au/Cu/Ni/Zn assay data.

---

## Quickstart

```bash
git clone https://github.com/cyber-pocho/stratigraphic-grade-prediction
cd stat_aware_pred
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Download the FORCE 2020 per-well CSVs from Kaggle and place them in `data/force2020/`.

**Train:**
```bash
python src/train/stage1.py --config configs/stage1.yaml
```

**Evaluate:**
```bash
python src/eval/run_eval.py --config configs/stage1.yaml --checkpoint checkpoints/stage1/best.pt
```

Key config options in `configs/stage1.yaml`:

```yaml
model:
  d_model: 128
  n_heads: 4
  n_layers: 4

training:
  epochs: 50
  batch_size: 64
  focal_loss_gamma: 2.0
```

---

## Evaluation

Stage 1 is evaluated against the FORCE 2020 Kaggle winning solution (gradient-boosted trees) using macro F1 across all 11 classes. Held-out wells are spatially isolated from training wells — random splits are optimistic in spatial problems because nearby wells are correlated.

Stage 2 reports RMSE and MAE on spatially held-out drill holes, with calibration curves (ECE) for uncertainty estimates compared against ordinary kriging variance.

---

## Repository Structure

```
stat_aware_pred/
├── data/
│   ├── force2020/       # FORCE 2020 CSVs (one per well, not committed)
│   └── gswa/            # GSWA drill holes (not committed)
├── notebooks/
│   └── 01_eda.ipynb     # EDA — class balance, log distributions, correlations
├── src/
│   ├── data/
│   │   ├── loader.py    # Well loading and lithology mapping
│   │   ├── features.py  # Derived petrophysical features
│   │   ├── dataset.py   # Sliding-window PyTorch Dataset
│   │   └── splits.py    # Spatial cross-validation splits
│   ├── models/
│   │   ├── transformer.py
│   │   ├── fusion.py
│   │   ├── heads.py
│   │   └── uncertainty.py
│   ├── train/
│   │   ├── stage1.py
│   │   └── stage2.py
│   └── eval/
│       ├── metrics.py
│       └── visualize.py
├── configs/
│   ├── stage1.yaml
│   └── stage2.yaml
└── requirements.txt
```

---

## References

Bormann et al. (2020). *FORCE Machine Learning Competition: Well Log and Lithofacies Dataset.* Zenodo.  
Lakshminarayanan et al. (2017). *Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles.* NeurIPS.  
Lin et al. (2017). *Focal Loss for Dense Object Detection.* ICCV.  
Vaswani et al. (2017). *Attention Is All You Need.* NeurIPS.
