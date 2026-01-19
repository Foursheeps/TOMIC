"""
Lightning Module for standard supervised learning with Token-based Transformer.

This module uses token sequences (input_ids) for feature extraction.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..encoder_decoder import NameTransformerEncoder
from .base import BaseLightningModule


@dataclass
class NameModelConfig:
    """Name-based Transformer model architecture configuration."""

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
    Lightning Module for standard supervised learning with Token-based Transformer.

    This model:
    - Uses token sequences (input_ids) as input
    - Uses Transformer encoder for feature extraction
    - Performs standard classification
    """

    def __init__(
        self,
        seq_len: int,
        hidden_size: int = 64,
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
        **kwargs,
    ):
        """
        Initialize Token-based Lightning Module.

        Args:
            seq_len: Sequence length
            hidden_size: Hidden size for transformer
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

        # Note: NameTransformerEncoder uses seq_len parameter as vocab_size for embedding
        self.encoder = NameTransformerEncoder(
            seq_len=seq_len,  # This parameter is used as vocab_size for embedding
            hidden_size=hidden_size,
            dropout=dropout,
            activation=activation,
            num_heads=num_heads,
            num_layers=num_layers,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def _extract_data_from_batch(self, batch: dict) -> dict:
        """
        Extract data from batch for Name model.

        Args:
            batch: Batch dictionary with "gene_ids" key

        Returns:
            Dictionary with "input_ids" key
        """
        return batch["gene_ids"]

    def _forward_encoder(self, encoder: nn.Module, data: dict | torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder for Name model.

        Args:
            encoder: Token Transformer encoder module
            data: Input dictionary with keys:
                - "input_ids": Token IDs (batch_size, seq_len)
                - "attention_mask": Attention mask (batch_size, seq_len)

        Returns:
            Encoded features of shape (batch_size, hidden_size)
            Note: We use task CLS token (index 1) for classification
        """

        # Token encoder outputs (batch_size, seq_len + 2, hidden_size)
        # [domain_cls_token, task_cls_token, ...token_embeddings...]
        # NameTransformerEncoder only takes input_ids, not attention_mask
        features = encoder(data)

        # Extract task CLS token for classification
        # Use index 1 (task CLS token)
        return features[:, 1, :]  # (batch_size, hidden_size)
