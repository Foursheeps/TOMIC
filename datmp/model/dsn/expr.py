import copy
from dataclasses import dataclass

import torch
from torch import nn

from ..encoder_decoder import ExpressionDecoder, ExpressionTransformerEncoder
from .base import BaseLightningModule

"""
Lightning Module for Domain Separation Networks with Expression-based Transformer.

This module uses gene expression sequences for feature extraction.
"""


@dataclass
class ExprModelConfig:
    """Expression-based Transformer model architecture configuration."""

    # Transformer architecture parameters
    def __init__(
        self,
        hidden_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 6,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        self.activation = activation

    def __repr__(self) -> str:
        return f"""
        ExprModelConfig(
            hidden_size={self.hidden_size},
            num_heads={self.num_heads},
            num_layers={self.num_layers},
            dropout={self.dropout},
            activation={self.activation},
        )
        """


class LightningModule(BaseLightningModule):
    """
    Lightning Module for Domain Separation Networks with Expression-based Transformer.

    This model:
    - Uses gene expression sequences as input
    - Uses Transformer encoder for feature extraction
    - Reconstructs expression using ExpressionDecoder
    """

    def __init__(
        self,
        hidden_size: int = 64,
        seq_len: int = 512,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_heads: int = 4,
        num_layers: int = 6,
        num_classes: int = None,
        lr: float = 1e-4,
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.1,
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        num_epochs: int = 100,
        train_batch_size: int = 32,
        **kwargs,
    ) -> None:
        """
        Initialize Expression-based Lightning Module.

        Args:
            hidden_size: Hidden size for transformer
            seq_len: Sequence length
            dropout: Dropout rate
            activation: Activation function name
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of classes for classification
            lr: Learning rate
            alpha: Reconstruction loss weight
            beta: Difference loss weight
            gamma: DANN loss weight
            scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs
            train_batch_size: Training batch size
        """

        super().__init__(
            num_classes=num_classes,
            lr=lr,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            scheduler_type=scheduler_type,
            warmup_ratio=warmup_ratio,
            num_epochs=num_epochs,
            train_batch_size=train_batch_size,
        )

        # Initialize encoders
        encoder = ExpressionTransformerEncoder(
            seq_len=seq_len,
            hidden_size=hidden_size,
            dropout=dropout,
            activation=activation,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.private_source_encoder = copy.deepcopy(encoder)
        self.private_target_encoder = copy.deepcopy(encoder)
        self.shared_encoder = copy.deepcopy(encoder)

        # Initialize decoder
        self.reconstructor = ExpressionDecoder(
            hidden_size=hidden_size,
            seq_len=seq_len,
            dropout=dropout,
            activation=activation,
        )

        # Domain classifier and classifier initialized in base class
        self.classifier = nn.Linear(hidden_size, num_classes)
        self.domain_classifier = nn.Linear(hidden_size, 2)
        # Loss functions initialized in base class

    def _forward_encoder(self, encoder, data: dict | torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder for MLP model."""
        return encoder(data)

    def _extract_batch_data(self, batch: dict) -> tuple:
        """Extract data from batch for MLP model."""
        # Convert to torch tensor if needed
        source_expr = batch["s_expr"]
        target_expr = batch["t_expr"]
        source_labels = batch["s_label"]
        target_labels = batch["t_label"]
        source_original = source_expr
        target_original = target_expr
        return source_expr, source_labels, target_expr, target_labels, source_original, target_original

    def _combine_features(self, private: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        """Combine private and shared features for patch model."""
        # Transformer models output (batch_size, seq_len+2, hidden_size)
        # Skip CLS tokens (indices 0 and 1) and combine
        return private[:, 2:, :] + shared[:, 2:, :]

    def _extract_classification_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Extract features for classification for patch model."""
        # Transformer models output (batch_size, seq_len+2, hidden_size)
        # Use task CLS token at index 1
        return shared_features[:, 1, :]

    def _extract_dann_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        """
        Extract features for domain classification for patch model.

        Args:
            shared_features: Shared feature representations of shape (batch_size, seq_len+2, hidden_size).

        Returns:
            Features for domain classification.
        """
        # Transformer models output (batch_size, seq_len+2, hidden_size)
        # Use domain CLS token at index 0
        return shared_features[:, 0, :]

    def _gather_diff_features(self, private: torch.Tensor, shared: torch.Tensor) -> tuple:
        """Gather difference features for patch model."""
        # Transformer models output (batch_size, seq_len+2, hidden_size)
        # Use private and shared features at index 2
        return private[:, 1, :], shared[:, 1, :]
