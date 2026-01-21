"""
DANN (Domain Adversarial Neural Networks) training scripts.

This package provides training scripts for DANN models:
- Unified training script: train.py (supports all model types)
"""

from .train import main as train_dann_func
from .train_config import TrainerConfig

__all__ = [
    "TrainerConfig",
    "train_dann_func",  # Unified training function
]
