"""
Base Lightning Module for standard training and testing.

This base class provides a simplified training pipeline without domain adaptation.
It focuses on standard supervised learning with encoder + classifier architecture.

Subclasses should implement:
- _create_encoder(): Create model-specific encoder (called lazily in forward() if not created in __init__)
- _forward_encoder(): Forward pass through encoder
"""

import logging
from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from ...utils import compute_accuracy, compute_metrics

torch.set_float32_matmul_precision("medium")

logger = logging.getLogger(__name__)


class BaseLightningModule(pl.LightningModule, ABC):
    """
    Base Lightning Module for standard supervised learning.

    This class provides shared functionality for all model types:
    - Standard training/validation/test steps
    - Classification loss computation
    - Optimizer and scheduler configuration
    - Logging and metrics

    Subclasses should implement:
    - _forward_encoder(): Forward pass through encoder
    """

    def __init__(
        self,
        lr: float = 1e-4,
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        num_epochs: int = 100,
        max_steps: int | None = None,
        warmup_steps: int | None = None,
        train_batch_size: int = 32,
        num_classes: int = None,
    ):
        """
        Initialize base Lightning Module.

        Args:
            lr: Learning rate
            scheduler_type: Learning rate scheduler type ("constant", "warmupcosine", "cosine")
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs
            max_steps: Maximum number of training steps (calculated automatically if None)
            warmup_steps: Number of warmup steps (calculated automatically if None)
            train_batch_size: Training batch size (for calculating steps)
            num_classes: Number of classes for classification
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = True

        # Initialize encoder (to be created by subclass)
        # Note: encoder will be created lazily in forward() if not created in subclass __init__
        self.encoder = nn.Identity()
        # Classifier should be initialized in subclass with nn.Linear(hidden_size, num_classes)
        # or nn.Linear(hidden_dims[-1], num_classes) for MLP models
        self.classifier = None  # Will be set by subclass
        self.loss_fn = nn.CrossEntropyLoss()

    @abstractmethod
    def _forward_encoder(self, encoder: nn.Module, data: dict | torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder.

        Args:
            encoder: Encoder module
            data: Input data (can be tensor or dict for token-based models)

        Returns:
            Encoded features
        """
        raise NotImplementedError("Subclass must implement _forward_encoder")

    def forward(self, data: dict | torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            data: Input data

        Returns:
            Classification logits
        """
        # Encode input
        features = self._forward_encoder(self.encoder, data)

        # Classify
        logits = self.classifier(features)

        return logits

    def _extract_data_from_batch(self, batch: dict) -> dict | torch.Tensor:
        """
        Extract and format data from batch for model forward pass.

        This method handles the conversion from Datasetcommon batch format to
        model-specific input format. Subclasses can override this if needed.

        Args:
            batch: Batch dictionary from Datasetcommon with keys:
                - "gene_ids": Gene ID sequences (for name/dual models)
                - "expr_ids": Binned expression values (for dual model with binning)
                - "expr": Continuous expression values (for patch/mlp/expr/dual models)
                - "label": Class labels

        Returns:
            Formatted data for model forward pass (dict or tensor)
        """
        # Default implementation: try to extract data based on batch keys
        # This will be overridden by subclasses if needed
        if "gene_ids" in batch and "expr" in batch:
            # Dual model format
            return {"name": batch["gene_ids"], "expr": batch["expr"]}
        elif "gene_ids" in batch:
            # Name model format
            return {"input_ids": batch["gene_ids"]}
        elif "expr" in batch:
            # Patch/MLP/Expr model format
            return batch["expr"]
        else:
            raise ValueError(f"Unknown batch format. Batch keys: {list(batch.keys())}")

    def _shared_step(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Shared step for training, validation, and testing.

        Args:
            batch: Batch dictionary from Datasetcommon

        Returns:
            Tuple of (pred, labels, loss)
        """
        # Extract data from batch (handles Datasetcommon format)
        data = self._extract_data_from_batch(batch)
        labels = batch["label"]

        # Forward pass
        pred = self.forward(data)

        # Compute loss
        loss = self.loss_fn(pred, labels)

        return pred, labels, loss

    def _build_log_dict(
        self,
        stage: str,
        pred: torch.Tensor,
        labels: torch.Tensor,
        loss: torch.Tensor,
        dataloader_idx: int | None = None,
    ) -> dict:
        """
        Build log dictionary for logging metrics and losses.

        Args:
            stage: Stage name ('train', 'val', 'test')
            pred: Predicted logits
            labels: Ground truth labels
            loss: Loss value
            dataloader_idx: Optional dataloader index for test stage (0=source, 1=target)

        Returns:
            Log dictionary with all metrics and losses
        """
        # Determine prefix based on stage and dataloader_idx
        if stage == "test" and dataloader_idx is not None:
            prefix = "test_source" if dataloader_idx == 0 else "test_target"
        else:
            prefix = stage

        log_dict = {f"{prefix}/loss": loss}

        if stage in ["train", "val"]:
            accuracy = compute_accuracy(pred, labels)
            log_dict[f"{prefix}/accuracy"] = accuracy
        elif stage == "test":
            metrics = compute_metrics(pred, labels)
            for metric_name, metric_value in metrics.items():
                log_dict[f"{prefix}/{metric_name}"] = metric_value

        return log_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Training step for a single batch.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        pred, labels, loss = self._shared_step(batch)

        # Log metrics
        log_dict = self._build_log_dict(stage="train", pred=pred, labels=labels, loss=loss)
        self.log_dict(log_dict, on_step=True, on_epoch=True)

        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """
        Validation step for a single batch.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index

        Returns:
            Loss tensor
        """
        pred, labels, loss = self._shared_step(batch)

        # Log metrics
        log_dict = self._build_log_dict(stage="val", pred=pred, labels=labels, loss=loss)
        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return loss

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx: int = 0) -> torch.Tensor:
        """
        Test step for a single batch with comprehensive metrics.

        Args:
            batch: Batch dictionary
            batch_idx: Batch index
            dataloader_idx: Dataloader index (0=source, 1=target for multiple test dataloaders)

        Returns:
            Loss tensor
        """
        pred, labels, loss = self._shared_step(batch)

        # Log metrics with dataloader prefix
        log_dict = self._build_log_dict(
            stage="test", pred=pred, labels=labels, loss=loss, dataloader_idx=dataloader_idx
        )
        self.log_dict(log_dict, on_step=False, on_epoch=True)

        return loss

    def configure_optimizers(self):
        """
        Configure the optimizer and learning rate scheduler.

        Returns:
            Optimizer (and optionally scheduler)
        """
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        self._calculate_scheduler_steps()

        if self.hparams.scheduler_type == "constant":
            return optimizer
        elif self.hparams.scheduler_type == "warmupcosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.max_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        elif self.hparams.scheduler_type == "cosine":
            scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams.max_steps,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "interval": "step",
                    "frequency": 1,
                },
            }
        else:
            raise ValueError(f"Invalid scheduler type: {self.hparams.scheduler_type}")

    def _calculate_scheduler_steps(self):
        """Calculate max_steps and warmup_steps from dataloader if available."""
        train_dataloader = None
        if self.trainer:
            if hasattr(self.trainer, "datamodule") and self.trainer.datamodule:
                try:
                    train_dataloader = self.trainer.datamodule.train_dataloader()
                except Exception:
                    pass
            elif hasattr(self.trainer, "train_dataloader"):
                try:
                    train_dataloader = self.trainer.train_dataloader
                    if callable(train_dataloader):
                        train_dataloader = train_dataloader()
                except Exception:
                    pass

        if train_dataloader is not None:
            self.hparams.max_steps = len(train_dataloader) * self.hparams.num_epochs
            self.hparams.warmup_steps = int(self.hparams.warmup_ratio * self.hparams.max_steps)

    def on_train_start(self):
        """Called when training starts. Recalculate scheduler steps if needed."""
        super().on_train_start()
        if self.hparams.max_steps is None:
            self._calculate_scheduler_steps()
            if self.hparams.max_steps is not None:
                # Reconfigure optimizer with scheduler if needed
                optimizer = self.optimizers()
                if isinstance(optimizer, torch.optim.Optimizer):
                    # If scheduler wasn't configured, configure it now
                    if self.hparams.scheduler_type == "warmupcosine":
                        scheduler = get_cosine_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=self.hparams.warmup_steps,
                            num_training_steps=self.hparams.max_steps,
                        )
                        self.lr_schedulers = [scheduler]
                    elif self.hparams.scheduler_type == "cosine":
                        scheduler = get_cosine_schedule_with_warmup(
                            optimizer,
                            num_warmup_steps=0,
                            num_training_steps=self.hparams.max_steps,
                        )
                        self.lr_schedulers = [scheduler]

    @torch.no_grad()
    def get_embeddings(self, data: dict | torch.Tensor) -> dict:
        """
        Get embeddings using encoder.

        Args:
            data: Input data (can be tensor or dict for token-based models)

        Returns:
            Dictionary containing:
                - features: Classification features (batch_size, hidden_size)
                - prob: Class probabilities (batch_size, num_classes)
                - pred: Class predictions (batch_size,)
        """
        # Forward through encoder to get classification features
        # Note: _forward_encoder already returns features suitable for classification
        # (e.g., CLS token for Transformer models, or direct output for MLP)
        cls_features = self._forward_encoder(self.encoder, data)

        # Get logits from classifier
        logits = self.classifier(cls_features)

        # Apply softmax to get probabilities
        prob = nn.functional.softmax(logits, dim=1)

        # Get predictions
        pred = torch.argmax(logits, dim=1)

        return {
            "features": cls_features,
            "prob": prob,
            "pred": pred,
        }


# ============================================================================
# Dataset and DataModule moved to base_dataset.py
# ============================================================================
