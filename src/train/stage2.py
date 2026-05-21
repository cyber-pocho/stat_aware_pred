"""Grade prediction training loop with stratigraphic conditioning (Stage 2)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader, TensorDataset

from src.models.fusion import CrossAttentionFusion
from src.models.heads import RegressionHead
from src.models.transformer import LithologyTransformer
from src.models.uncertainty import DeepEnsemble


class GradePredictionModel(nn.Module):
    """
    Full Stage 2 model: depth sequence + spatial context → grade + uncertainty.

    At inference, wrap this in DeepEnsemble (5 members) for uncertainty.
    """

    def __init__(
        self,
        n_depth_features: int,
        n_spatial_features: int,
        n_lith_classes: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_depth_layers: int = 2,
        dropout: float = 0.1,
        regression_hidden: int = 64,
    ) -> None:
        super().__init__()
        self.fusion = CrossAttentionFusion(
            n_depth_features=n_depth_features,
            n_spatial_features=n_spatial_features,
            n_lith_classes=n_lith_classes,
            d_model=d_model,
            n_heads=n_heads,
            n_depth_layers=n_depth_layers,
            dropout=dropout,
        )
        self.head = RegressionHead(d_model, hidden_dim=regression_hidden, dropout=dropout)

    def forward(
        self,
        assay: torch.Tensor,
        lith_probs: torch.Tensor,
        spatial: torch.Tensor,
        padding_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        fused, attn_w = self.fusion(assay, lith_probs, spatial, padding_mask)
        return self.head(fused), attn_w  # [B], [B, 1, L]


def train(config: dict) -> None:
    """
    Train Stage 2 grade prediction model.

    Requires:
        - A trained Stage 1 checkpoint to generate lithology probabilities.
        - GSWA drill hole data in config["data"]["data_dir"].
          The dataset should expose batches with keys:
            assay       [B, L, n_elements]
            spatial     [B, n_spatial]
            grade       [B]              target grade at query location
            padding_mask [B, L]          True at padded depth intervals
          See src/data/gswa_loader.py (to be implemented when data is available).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load Stage 1 to generate lith probs — frozen during Stage 2 training
    stage1_ckpt = torch.load(config["data"]["stage1_checkpoint"], map_location=device)
    from src.data.loader import N_CLASSES
    stage1 = LithologyTransformer(
        n_features=len(stage1_ckpt["feature_cols"]),
        n_classes=N_CLASSES,
        **stage1_ckpt["config"]["model"],
    ).to(device)
    stage1.load_state_dict(stage1_ckpt["model"])
    stage1.eval()
    for p in stage1.parameters():
        p.requires_grad_(False)

    n_lith = N_CLASSES
    n_depth = config["data"]["n_assay_elements"]
    n_spatial = config["data"]["n_spatial_features"]

    n_ensemble = config["uncertainty"]["n_ensemble_members"]
    ensemble_members: list[GradePredictionModel] = []
    ckpt_dir = Path(config["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for member_idx in range(n_ensemble):
        print(f"\n--- Ensemble member {member_idx + 1}/{n_ensemble} ---")

        torch.manual_seed(config["training"]["seed"] + member_idx)

        model = GradePredictionModel(
            n_depth_features=n_depth,
            n_spatial_features=n_spatial,
            n_lith_classes=n_lith,
            **{k: v for k, v in config["model"].items()},
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config["training"]["lr"],
            weight_decay=config["training"]["weight_decay"],
        )
        criterion = nn.MSELoss()

        # --- Replace the block below with real GSWA DataLoaders ---
        # from src.data.gswa_loader import build_dataloaders
        # train_loader, val_loader = build_dataloaders(config)
        # ----------------------------------------------------------
        raise NotImplementedError(
            "GSWA data loader not yet implemented. "
            "Download GSWA data and implement src/data/gswa_loader.py, "
            "then replace the placeholder above."
        )

    ensemble = DeepEnsemble(ensemble_members)
    torch.save({"ensemble": [m.state_dict() for m in ensemble_members],
                "config": config}, ckpt_dir / "ensemble.pt")
    print(f"\nSaved ensemble to {ckpt_dir}/ensemble.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
