"""Lithology classifier training loop (Stage 1)."""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

from src.data.loader import load_all_wells, get_well_list
from src.data.dataset import WellLogDataset
from src.models.transformer import LithologyTransformer


def train(config: dict) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
