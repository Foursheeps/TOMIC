"""
ADDA (Adversarial Discriminative Domain Adaptation) training scripts.

This package provides training scripts for ADDA models:
- Unified training script: train.py (supports all model types)
"""

from .train import main as train_adda_func
from .train_config import TrainerConfig

__all__ = [
    "TrainerConfig",
    "train_adda_func",  # Unified training function
]
