from dataclasses import dataclass

import torch

from ..encoder_decoder import NameTransformerEncoder
from .base import BaseLightningModule

"""
Lightning Module for Domain Adversarial Neural Networks with Token-based Transformer.

This module uses token sequences (input_ids) for feature extraction.
"""


@dataclass
class NameModelConfig:
    """Name-based Transformer model architecture configuration."""

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


class LightningModule(BaseLightningModule):
    """
    Lightning Module for Domain Adversarial Neural Networks with Token-based Transformer.

    This model:
    - Uses token sequences (input_ids) as input
    - Uses Transformer encoder for feature extraction
    - Uses DANN adversarial loss for domain adaptation
    """

    def __init__(
        self,
        seq_len: int = 512,
        hidden_size: int = 64,
        dropout: float = 0.1,
        activation: str = "gelu",
        num_heads: int = 4,
        num_layers: int = 6,
        num_classes: int = None,
        lr: float = 1e-4,
        gamma: float = 0.1,
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        num_epochs: int = 100,
        train_batch_size: int = 32,
        **kwargs,
    ):
        """
        Initialize Token-based Lightning Module.

        Args:
            seq_len: Sequence length (used as vocab_size for embedding)
            hidden_size: Hidden size for transformer
            dropout: Dropout rate
            activation: Activation function name
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            num_classes: Number of classes for classification
            lr: Learning rate
            gamma: DANN loss weight
            scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs
            train_batch_size: Training batch size
        """
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            gamma=gamma,
            scheduler_type=scheduler_type,
            warmup_ratio=warmup_ratio,
            num_epochs=num_epochs,
            train_batch_size=train_batch_size,
        )

        # Initialize shared encoder
        # Note: NameTransformerEncoder uses seq_len parameter as vocab_size for embedding
        self.shared_encoder = NameTransformerEncoder(
            seq_len=seq_len,
            hidden_size=hidden_size,
            dropout=dropout,
            activation=activation,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        # Initialize classifier and domain classifier
        self.classifier = torch.nn.Linear(hidden_size, num_classes)
        self.domain_classifier = torch.nn.Linear(hidden_size, 2)
        # Loss functions initialized in base class

    def _forward_encoder(self, encoder, input_ids: torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder for name model."""
        # NameTransformerEncoder only takes input_ids, not attention_mask
        return encoder(input_ids)

    def _extract_batch_data(self, batch: dict) -> tuple:
        """Extract data from batch for name model."""
        source_input_ids = batch["s_gene_ids"]
        target_input_ids = batch["t_gene_ids"]
        source_labels = batch["s_label"]
        target_labels = batch["t_label"]
        source_original = source_input_ids
        target_original = target_input_ids
        return source_input_ids, source_labels, target_input_ids, target_labels, source_original, target_original

    def _extract_classification_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Extract features for classification for name model."""
        # Transformer models output (batch_size, seq_len+2, hidden_size)
        # Use task CLS token at index 1
        return shared_features[:, 1, :]

    def _extract_dann_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Extract features for domain classification for name model."""
        # Transformer models output (batch_size, seq_len+2, hidden_size)
        # Use domain CLS token at index 0
        return shared_features[:, 0, :]
