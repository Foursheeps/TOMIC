"""
Lightning Module for standard supervised learning with MLP encoder.

This module uses MLP (Multi-Layer Perceptron) for feature extraction
without domain adaptation.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..encoder_decoder import MLPEncoder
from .base import BaseLightningModule


@dataclass
class MLPModelConfig:
    """MLP-based model architecture configuration."""

    def __init__(
        self,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.activation = activation


class LightningModule(BaseLightningModule):
    """
    Lightning Module for standard supervised learning with MLP encoder.

    This model:
    - Uses MLP encoder for feature extraction
    - Performs standard classification
    """

    def __init__(
        self,
        seq_len: int = 1024,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
        activation: str = "gelu",
        num_classes: int = None,
        lr: float = 1e-4,
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        num_epochs: int = 100,
        train_batch_size: int = 32,
    ):
        """
        Initialize MLP-based Lightning Module.

        Args:
            seq_len: Sequence length
            hidden_dims: Hidden dimensions for encoder MLP layers
            dropout: Dropout rate
            activation: Activation function name
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

        # Initialize encoder
        self.encoder = MLPEncoder(
            input_dim=seq_len,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )

        # Initialize classifier
        self.classifier = nn.Linear(hidden_dims[-1], num_classes)

    def _extract_data_from_batch(self, batch: dict) -> torch.Tensor:
        """
        Extract data from batch for MLP model.

        Args:
            batch: Batch dictionary with "expr" key

        Returns:
            Expression tensor
        """
        return batch["expr"]

    def _forward_encoder(self, encoder: nn.Module, data: dict | torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder for MLP model.

        Args:
            encoder: MLP encoder module
            data: Input tensor of shape (batch_size, seq_len)

        Returns:
            Encoded features of shape (batch_size, hidden_dims[-1])
        """
        return encoder(data)
