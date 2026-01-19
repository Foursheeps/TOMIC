"""
Lightning Module for standard supervised learning with Dual Transformer.

This module uses dual cross-attention encoder that processes gene names and expressions
simultaneously, similar to Dual Transformer architecture.
"""

from dataclasses import dataclass

import torch
import torch.nn as nn

from ..encoder_decoder import DualTransformerEncoder
from .base import BaseLightningModule


@dataclass
class DualTransformerModelConfig:
    """Dual Transformer model architecture configuration."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_heads_cross_attn: int = 4,
        num_layers_cross_attn: int = 6,
        num_heads_encoder: int = 4,
        num_layers_encoder: int = 6,
        dropout: float = 0.1,
        activation: str = "gelu",
        **kwargs,
    ):
        self.hidden_size = hidden_size
        self.num_heads_cross_attn = num_heads_cross_attn
        self.num_heads_encoder = num_heads_encoder
        self.num_layers_cross_attn = num_layers_cross_attn
        self.num_layers_encoder = num_layers_encoder
        self.dropout = dropout
        self.activation = activation


class LightningModule(BaseLightningModule):
    """
    Lightning Module for standard supervised learning with Dual Transformer.

    This model:
    - Uses gene name token IDs and gene expression values as input
    - Uses DualTransformerEncoder (dual cross-attention) for feature extraction
    - Performs standard classification
    """

    def __init__(
        self,
        # Data parameters
        seq_len: int,
        hidden_size: int = 64,
        num_heads_cross_attn: int = 4,
        num_heads_encoder: int = 4,
        num_layers_cross_attn: int = 6,
        num_layers_encoder: int = 6,
        dropout: float = 0.1,
        activation: str = "gelu",
        binning: int = None,
        # Training parameters
        num_classes: int = None,
        lr: float = 1e-4,
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        num_epochs: int = 100,
        train_batch_size: int = 32,
        **kwargs,
    ):
        """
        Initialize Dual Transformer Lightning Module.

        Args:
            seq_len: Sequence length
            hidden_size: Hidden size for encoder
            num_heads_cross_attn: Number of attention heads for cross-attention
            num_heads_encoder: Number of attention heads for encoder
            num_layers_cross_attn: Number of layers for cross-attention
            num_layers_encoder: Number of layers for encoder
            binning: Number of bins for discrete expression values. If None, uses continuous values.
            lr: Learning rate
            scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs
            train_batch_size: Training batch size
        """
        super().__init__(
            num_classes=num_classes,
            lr=lr,
            scheduler_type=scheduler_type,
            warmup_ratio=warmup_ratio,
            num_epochs=num_epochs,
            train_batch_size=train_batch_size,
        )

        # Initialize encoder
        self.encoder = DualTransformerEncoder(
            seq_len=seq_len,
            hidden_size=hidden_size,
            num_heads_cross_attn=num_heads_cross_attn,
            num_heads_encoder=num_heads_encoder,
            num_layers_cross_attn=num_layers_cross_attn,
            num_layers_encoder=num_layers_encoder,
            dropout=dropout,
            activation=activation,
            binning=binning,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

        if binning is not None:
            self.batch_expr_key = "expr_ids"
        else:
            self.batch_expr_key = "expr"

    def _extract_data_from_batch(self, batch: dict) -> dict:
        """
        Extract data from batch for Dual Transformer model.

        Args:
            batch: Batch dictionary with "gene_ids" and "expr"/"expr_ids" keys

        Returns:
            Dictionary with "name" and "expr" keys
        """
        name = batch["gene_ids"]
        expr = batch[self.batch_expr_key]
        return {"name": name, "expr": expr}

    def _forward_encoder(self, encoder, data: dict | torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder for Dual Transformer model.

        Returns task CLS token for classification.
        """
        name = data["name"]
        expr = data["expr"]
        # DualTransformerEncoder returns (aligned, encoded_name, encoded_expr)
        aligned, encoded_name, encoded_expr = encoder(name, expr)
        # Use task CLS token (index 1) for classification
        return aligned[:, 1, :]  # [batch_size, hidden_size]
