from dataclasses import dataclass

import torch

from ..encoder_decoder.dual import DualTransformerEncoder
from .base import BaseLightningModule

"""
Lightning Module for Domain Adversarial Neural Networks with Dual Transformer.

This module uses dual cross-attention encoder that processes gene names and expressions
simultaneously.
"""


@dataclass
class DualTransformerModelConfig:
    """Dual Transformer model architecture configuration."""

    # Transformer architecture parameters
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
    Lightning Module for Domain Adversarial Neural Networks with Dual Transformer.

    This model:
    - Uses gene name token IDs and gene expression values as input
    - Uses DualTransformerEncoder (dual cross-attention) for feature extraction
    - Uses DANN adversarial loss for domain adaptation
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
        gamma: float = 0.1,
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
        self.shared_encoder = DualTransformerEncoder(
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

        # Initialize classifier and domain classifier
        self.classifier = torch.nn.Linear(hidden_size, num_classes)
        self.domain_classifier = torch.nn.Linear(hidden_size, 2)
        # Loss functions initialized in base class

        self.batch_expr_key = "_expr_ids" if binning is not None else "_expr"

    def _forward_encoder(self, encoder, data: dict | torch.Tensor) -> dict:
        """Forward pass through encoder for scGPT model.

        Returns a dict with 'aligned', 'name', and 'expr' keys for scGPT model.
        """
        name = data["name"]
        expr = data["expr"]
        # DualTransformerEncoder returns (aligned, encoded_name, encoded_expr)
        aligned, encoded_name, encoded_expr = encoder(name, expr)
        # Return dict with all outputs needed for feature extraction
        return {
            "aligned": aligned,
            "name": encoded_name,
            "expr": encoded_expr,
        }

    def _extract_batch_data(self, batch: dict) -> tuple:
        """Extract data from batch for scGPT model."""
        source_expr = batch["s" + self.batch_expr_key]
        target_expr = batch["t" + self.batch_expr_key]

        source_data = {
            "name": batch["s_gene_ids"],
            "expr": source_expr,
        }
        source_labels = batch["s_label"]
        target_data = {
            "name": batch["t_gene_ids"],
            "expr": target_expr,
        }
        target_labels = batch["t_label"]
        source_original = {
            "name": batch["s_gene_ids"],
            "expr": source_expr,
        }
        target_original = {
            "name": batch["t_gene_ids"],
            "expr": target_expr,
        }
        return source_data, source_labels, target_data, target_labels, source_original, target_original

    def _extract_classification_features(self, shared_features: dict) -> torch.Tensor:
        """Extract features for classification for scGPT model."""
        # shared_features is a dict with 'aligned', 'name', 'expr' keys
        # Use CLS token (first token) from concatenated output for classification
        shared_output = shared_features["aligned"]  # [batch_size, 1 + seq_len + seq_len, hidden_size]
        return shared_output[:, 1, :]  # [batch_size, hidden_size]

    def _extract_dann_features(self, shared_features: dict) -> torch.Tensor:
        """
        Extract features for domain classification for scGPT model.

        Args:
            shared_features: Dict with 'aligned', 'name', 'expr' keys.

        Returns:
            Features for domain classification of shape [batch_size, hidden_size].
        """
        # Use CLS token (first token) from aligned output for domain classification
        shared_output = shared_features["aligned"]  # [batch_size, 1 + seq_len + seq_len, hidden_size]
        return shared_output[:, 0, :]  # [batch_size, hidden_size]
