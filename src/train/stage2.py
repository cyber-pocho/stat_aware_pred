"""Grade prediction training loop with stratigraphic conditioning (Stage 2)."""

from __future__ import annotations

import argparse

import yaml

from src.models.fusion import CrossAttentionFusion
from src.models.heads import RegressionHead
from src.models.uncertainty import DeepEnsemble


def train(config: dict) -> None:
    raise NotImplementedError


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    train(cfg)
