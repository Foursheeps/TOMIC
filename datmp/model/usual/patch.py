"""
Lightning Module for standard supervised learning with Patch-based Transformer.

This module uses patch-based encoding where input vectors are split into patches
and processed by a Transformer encoder.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..encoder_decoder import PatchTransformerEncoder
from .base import BaseLightningModule


@dataclass
class PatchModelConfig:
    """Patch-based Transformer model architecture configuration."""

    def __init__(
        self,
        hidden_size: int = 64,
        patch_size: int = 40,
        num_heads: int = 4,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation


class LightningModule(BaseLightningModule):
    """
    Lightning Module for standard supervised learning with Patch-based Transformer.

    This model:
    - Splits input vectors into patches
    - Uses Transformer encoder for feature extraction
    - Performs standard classification
    """

    def __init__(
        self,
        seq_len: int = 512,
        hidden_size: int = 64,
        patch_size: int = 40,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_heads: int = 4,
        num_layers: int = 6,
        num_classes: int = None,
        lr: float = 1e-4,
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        num_epochs: int = 100,
        train_batch_size: int = 32,
    ):
        """
        Initialize Patch-based Lightning Module.

        Args:
            seq_len: Sequence length
            hidden_size: Hidden size for transformer
            patch_size: Size of each patch
            dropout: Dropout rate
            activation: Activation function name
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of classes for classification
            lr: Learning rate
            scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs
            train_batch_size: Training batch size
        """
        super().__init__(
            lr=lr,
            scheduler_type=scheduler_type,
            warmup_ratio=warmup_ratio,
            num_epochs=num_epochs,
            train_batch_size=train_batch_size,
            num_classes=num_classes,
        )

        self.encoder = PatchTransformerEncoder(
            seq_len=seq_len,
            hidden_size=hidden_size,
            patch_size=patch_size,
            dropout=dropout,
            activation=activation,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def _extract_data_from_batch(self, batch: dict) -> torch.Tensor:
        """
        Extract data from batch for Patch model.

        Args:
            batch: Batch dictionary with "expr" key

        Returns:
            Expression tensor
        """
        return batch["expr"]

    def _forward_encoder(self, encoder: nn.Module, data: dict | torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder for Patch model.

        Args:
            encoder: Patch Transformer encoder module
            data: Input tensor of shape (batch_size, seq_len)

        Returns:
            Encoded features of shape (batch_size, hidden_size)
            Note: We use task CLS token (index 1) for classification
        """
        # Patch encoder outputs (batch_size, num_patches + 2, hidden_size)
        # [domain_cls_token, task_cls_token, ...patches...]
        features = encoder(data)

        # Extract task CLS token for classification
        # Use index 1 (task CLS token)
        return features[:, 1, :]  # (batch_size, hidden_size)
