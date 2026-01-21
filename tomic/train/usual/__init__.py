"""
Training scripts for standard supervised learning models.

This package provides training scripts for models without domain adaptation:
- Unified training script: train.py (supports all model types)
"""

from .train import main as train_usual_func
from .train_config import TrainerConfig

__all__ = [
    "TrainerConfig",
    "train_usual_func",  # Unified training function
]
