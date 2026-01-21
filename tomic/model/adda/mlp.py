import copy
from dataclasses import dataclass

import torch

from ..encoder_decoder import MLPEncoder
from .base import BaseLightningModule

"""
Lightning Module for Adversarial Discriminative Domain Adaptation with MLP encoder.

This module uses MLP (Multi-Layer Perceptron) for feature extraction.
"""


@dataclass
class MLPModelConfig:
    """MLP-based model architecture configuration."""

    # MLP architecture parameters
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
    Lightning Module for Adversarial Discriminative Domain Adaptation with MLP encoder.

    This model:
    - Uses MLP encoder for feature extraction
    - Uses ADDA two-stage training for domain adaptation
    """

    def __init__(
        self,
        seq_len: int = 1024,
        hidden_dims: list[int] = [512, 256, 128],
        dropout: float = 0.1,
        activation: str = "gelu",
        num_classes: int = None,
        lr: float = 1e-4,
        pretrain_epochs: int = 80,
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
            pretrain_epochs: Number of epochs for source domain pre-training
            scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs (for adversarial stage)
            train_batch_size: Training batch size
        """

        super().__init__(
            lr=lr,
            pretrain_epochs=pretrain_epochs,
            scheduler_type=scheduler_type,
            warmup_ratio=warmup_ratio,
            num_epochs=num_epochs,
            train_batch_size=train_batch_size,
            num_classes=num_classes,
        )

        # Initialize source encoder
        self.source_encoder = MLPEncoder(
            input_dim=seq_len,
            hidden_dims=hidden_dims,
            dropout=dropout,
            activation=activation,
        )

        # Initialize target encoder
        self.target_encoder = copy.deepcopy(self.source_encoder)

        # Initialize classifier and discriminator
        self.classifier = torch.nn.Linear(hidden_dims[-1], num_classes)
        self.discriminator = torch.nn.Sequential(
            torch.nn.Linear(hidden_dims[-1], 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),
            torch.nn.Linear(512, 1),
        )
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

    def _extract_classification_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract features for classification for MLP model."""
        # MLP outputs (batch_size, hidden_size), use entire feature vector
        return features

    def _extract_discriminator_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Extract features for discriminator for MLP model.

        Args:
            features: Feature representations of shape (batch_size, hidden_size).

        Returns:
            Features for discriminator.
        """
        # MLP outputs (batch_size, hidden_size), use entire feature vector
        return features
