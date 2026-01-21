"""
DSN (Domain Separation Network) training scripts.

This package provides training scripts for DSN models:
- Unified training script: train.py (supports all model types)
"""

from .train import main as train_dsn_func
from .train_config import TrainerConfig

__all__ = [
    "TrainerConfig",
    "train_dsn_func",  # Unified training function
]
