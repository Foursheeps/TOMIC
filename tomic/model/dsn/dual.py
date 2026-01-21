import copy
from dataclasses import dataclass

import torch
import torch.nn as nn

from ..encoder_decoder.dual import DualTransformerDecoder, DualTransformerEncoder
from .base import BaseLightningModule
from .loss import MSE, SIMSE, GradReverse

"""
Lightning Module for Domain Separation Networks with Dual Transformer.

This module uses dual cross-attention encoder that processes gene names and expressions
simultaneously, similar to Dual Transformer architecture.
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

    def __repr__(self) -> str:
        return f"""
        DualTransformerModelConfig(
            hidden_size={self.hidden_size},
            num_heads_cross_attn={self.num_heads_cross_attn},
            num_heads_encoder={self.num_heads_encoder},
            num_layers_cross_attn={self.num_layers_cross_attn},
            num_layers_encoder={self.num_layers_encoder},
            dropout={self.dropout},
            activation={self.activation},
        )
        """


class LightningModule(BaseLightningModule):
    """
    Lightning Module for Domain Separation Networks with Dual Transformer.

    This model:
    - Uses gene name token IDs and gene expression values as input
    - Uses DualTransformerEncoder (dual cross-attention) for feature extraction
    - Reconstructs name and expr sequences using DualTransformerDecoder
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
        alpha: float = 0.25,
        beta: float = 0.25,
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
        encoder = DualTransformerEncoder(
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

        self.private_source_encoder = copy.deepcopy(encoder)
        self.private_target_encoder = copy.deepcopy(encoder)
        self.shared_encoder = copy.deepcopy(encoder)

        # Initialize decoder
        self.reconstructor = DualTransformerDecoder(
            seq_len=seq_len,
            hidden_size=hidden_size,
            dropout=dropout,
            binning=binning,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)
        self.domain_classifier = nn.Linear(hidden_size, 2)

        if binning is not None:
            self._compute_scgpt_reconstruction_loss = self._compute_scgpt_reconstruction_loss_discrete
        else:
            self._compute_scgpt_reconstruction_loss = self._compute_scgpt_reconstruction_loss_continuous

        self.batch_expr_key = "_expr_ids" if binning is not None else "_expr"

    def _forward_encoder(self, encoder, data: dict | torch.Tensor) -> dict:
        """Forward pass through encoder for scGPT model.

        Returns a dict with 'aligned', 'name', and 'expr' keys for scGPT model.
        """
        name = data["name"]
        expr = data["expr"]
        # ScGPTEncoder returns (aligned, encoded_name, encoded_expr)
        aligned, encoded_name, encoded_expr = encoder(name, expr)
        # Return dict with all outputs needed for reconstruction
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

    def _combine_features(self, private: dict, shared: dict) -> dict:
        """Combine private and shared features for scGPT model."""
        # Both are dicts with 'output', 'name', 'expr' keys
        # Combine name and expr features separately
        combined_name = (private["name"] + shared["name"]) / 2
        combined_expr = (private["expr"] + shared["expr"]) / 2
        return {
            "name": combined_name,
            "expr": combined_expr,
        }

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

    def _gather_diff_features(self, private: dict, shared: dict) -> tuple:
        """Gather difference features for scGPT model."""
        # Both are dicts with 'aligned', 'name', 'expr' keys
        # Use CLS token from aligned output for difference loss
        private_output = private["aligned"]  # [batch_size, 1 + seq_len + seq_len, hidden_size]
        shared_output = shared["aligned"]  # [batch_size, 1 + seq_len + seq_len, hidden_size]
        private_cls = private_output[:, 0, :]  # [batch_size, hidden_size]
        shared_cls = shared_output[:, 0, :]  # [batch_size, hidden_size]
        return private_cls, shared_cls

    def compute_dann_loss(
        self,
        shared_features: dict,
        domain_labels: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """
        Compute DANN loss using Gradient Reversal Layer (GRL) for scGPT model.

        Args:
            shared_features: Dict with 'aligned', 'name', 'expr' keys
            domain_labels: Domain labels (0 for source, 1 for target)
            p: Gradient reversal scaling factor

        Returns:
            DANN loss
        """
        # Extract features for DANN
        dann_features = self._extract_dann_features(shared_features)  # [batch_size, hidden_size]
        reversed_features = GradReverse.apply(dann_features, p)
        domain_preds = self.domain_classifier(reversed_features)
        return self.loss_dann(domain_preds, domain_labels)

    @staticmethod
    def _compute_scgpt_reconstruction_loss_discrete(
        reconstructed_name: torch.Tensor,
        reconstructed_expr: torch.Tensor,
        original_name: torch.Tensor,
        original_expr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for scGPT model (name and expr sequences).

        Args:
            reconstructed_name: Predicted name token logits [batch_size, seq_len, vocab_size]
            reconstructed_expr: Predicted expression values [batch_size, seq_len] or [batch_size, seq_len, binning+1]
            original_name: Ground truth name token IDs [batch_size, seq_len]
            original_expr: Ground truth expression values [batch_size, seq_len]

        Returns:
            Reconstruction loss
        """
        # Name reconstruction loss (cross-entropy)
        name_loss = nn.CrossEntropyLoss()(
            reconstructed_name.reshape(-1, reconstructed_name.shape[-1]),
            original_name.reshape(-1).long(),
        )

        # Expression reconstruction loss
        # Discrete: cross-entropy loss
        expr_loss = nn.CrossEntropyLoss()(
            reconstructed_expr.reshape(-1, reconstructed_expr.shape[-1]),
            original_expr.reshape(-1).long(),
        )

        return name_loss + expr_loss

    @staticmethod
    def _compute_scgpt_reconstruction_loss_continuous(
        reconstructed_name: torch.Tensor,
        reconstructed_expr: torch.Tensor,
        original_name: torch.Tensor,
        original_expr: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for scGPT model (expression sequence).

        Args:
            reconstructed_name: Predicted name token logits [batch_size, seq_len, vocab_size]
            reconstructed_expr: Predicted expression values [batch_size, seq_len]
            original_name: Ground truth name token IDs [batch_size, seq_len]
            original_expr: Ground truth expression values [batch_size, seq_len]

        Returns:
            Reconstruction loss
        """

        # Name reconstruction loss (cross-entropy)
        name_loss = nn.CrossEntropyLoss()(
            reconstructed_name.reshape(-1, reconstructed_name.shape[-1]),
            original_name.reshape(-1).long(),
        )

        # Expression reconstruction loss
        # Continuous: MSE + SIMSE loss
        expr_loss = MSE()(reconstructed_expr, original_expr) + SIMSE()(reconstructed_expr, original_expr)

        return name_loss + expr_loss

    def _reconstruct(self, combined_features: dict) -> tuple:
        """
        Reconstruct input from combined features for scGPT model.

        Args:
            combined_features: Dict with 'name' and 'expr' keys

        Returns:
            Tuple of (reconstructed_name, reconstructed_expr)
        """
        combined_name = combined_features["name"]
        combined_expr = combined_features["expr"]
        # ScGPTDecoder takes encoded_name and encoded_expr
        reconstructed_name, reconstructed_expr = self.reconstructor(combined_name, combined_expr)
        return reconstructed_name, reconstructed_expr

    def loss_recon(
        self,
        reconstructed: tuple,
        original: dict,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for scGPT model.

        Args:
            reconstructed: Tuple of (reconstructed_name, reconstructed_expr)
            original: Dict with 'name' and 'expr' keys
            attention_mask: Not used for scGPT model

        Returns:
            Reconstruction loss
        """
        reconstructed_name, reconstructed_expr = reconstructed
        original_name = original["name"]
        original_expr = original["expr"]

        return self._compute_scgpt_reconstruction_loss(
            reconstructed_name, reconstructed_expr, original_name, original_expr
        )
