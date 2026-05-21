"""Lithology classifier training loop (Stage 1)."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.data.loader import load_all_wells, N_CLASSES, CORE_LOG_CURVES
from src.data.features import add_derived_features, DERIVED_FEATURES
from src.data.dataset import WellLogDataset
from src.models.transformer import LithologyTransformer


def focal_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    gamma: float,
    padding_mask: torch.Tensor,
) -> torch.Tensor:
    B, L, C = logits.shape
    logits_flat = logits.view(B * L, C)
    targets_flat = targets.view(B * L)
    valid = ~padding_mask.view(B * L)

    log_probs = F.log_softmax(logits_flat, dim=-1)
    pt = log_probs.exp().gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    ce = -log_probs.gather(1, targets_flat.unsqueeze(1)).squeeze(1)
    loss = ((1 - pt) ** gamma) * ce
    return loss[valid].mean()


def train(config: dict) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_all_wells(config["data"]["data_dir"], verbose=True)
    if config["data"].get("add_derived", False):
        df = add_derived_features(df)

    feature_cols = CORE_LOG_CURVES + (DERIVED_FEATURES if config["data"].get("add_derived") else [])
    feature_cols = [c for c in feature_cols if c in df.columns]

    all_wells = sorted(df["WELL"].unique().tolist())
    n_val = max(1, int(len(all_wells) * config["data"]["val_fraction"]))
    rng = torch.Generator().manual_seed(config["training"]["seed"])
    perm = torch.randperm(len(all_wells), generator=rng).tolist()
    val_wells = [all_wells[i] for i in perm[:n_val]]
    train_wells = [all_wells[i] for i in perm[n_val:]]

    train_ds, val_ds = WellLogDataset.train_val_split(
        df,
        train_wells=train_wells,
        val_wells=val_wells,
        feature_cols=feature_cols,
        window_size=config["data"]["window_size"],
        stride=config["data"]["stride"],
    )
    print(f"Train windows: {len(train_ds)}  Val windows: {len(val_ds)}")

    nw = config["training"].get("num_workers", 2)
    train_loader = DataLoader(train_ds, batch_size=config["training"]["batch_size"],
                              shuffle=True, num_workers=nw, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["training"]["batch_size"],
                              shuffle=False, num_workers=nw, pin_memory=True)

    model = LithologyTransformer(
        n_features=len(feature_cols),
        n_classes=N_CLASSES,
        **config["model"],
    ).to(device)
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=config["training"]["lr"],
                                  weight_decay=config["training"]["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config["training"]["epochs"])

    gamma     = config["training"]["focal_loss_gamma"]
    log_every = config["output"]["log_every"]
    ckpt_dir  = Path(config["output"]["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        train_loss, n = 0.0, 0
        for i, batch in enumerate(train_loader):
            features = batch["features"].to(device)
            labels   = batch["labels"].to(device)
            depths   = batch["depths"].to(device)
            mask     = batch["padding_mask"].to(device)

            loss = focal_loss(model(features, depths, padding_mask=mask), labels, gamma, mask)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            train_loss += loss.item(); n += 1
            if (i + 1) % log_every == 0:
                print(f"  [{epoch}/{config['training']['epochs']}] "
                      f"batch {i+1}/{len(train_loader)}  loss={train_loss/n:.4f}")

        scheduler.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                val_loss += focal_loss(
                    model(batch["features"].to(device),
                          batch["depths"].to(device),
                          padding_mask=batch["padding_mask"].to(device)),
                    batch["labels"].to(device),
                    gamma,
                    batch["padding_mask"].to(device),
                ).item()
        val_loss /= len(val_loader)

        print(f"Epoch {epoch:3d}  train={train_loss/n:.4f}  val={val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"epoch": epoch, "model": model.state_dict(),
                        "val_loss": val_loss, "feature_cols": feature_cols,
                        "val_wells": val_wells, "config": config},
                       ckpt_dir / "best.pt")
            print(f"  Saved best checkpoint (val={val_loss:.4f})")

    torch.save({"epoch": epoch, "model": model.state_dict(),
                "feature_cols": feature_cols, "val_wells": val_wells,
                "config": config}, ckpt_dir / "last.pt")
    print(f"\nDone. Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
