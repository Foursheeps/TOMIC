from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from ...utils import compute_accuracy, compute_metrics
from ..dsn.loss import GradReverse

"""
Base Lightning Module for Domain Adversarial Neural Networks (DANN).

DANN uses a single shared encoder with adversarial domain classification
via gradient reversal layer (GRL) to learn domain-invariant features.
"""

torch.set_float32_matmul_precision("medium")


class BaseLightningModule(pl.LightningModule, ABC):
    """
    Base Lightning Module for Domain Adversarial Neural Networks.

    This class provides shared functionality for all model types:
    - Loss computation
    - Training/validation/test steps
    - Optimizer and scheduler configuration
    - Domain-specific forward passes
    - Logging and metrics

    Subclasses should implement:
    - _forward_encoder(): Forward pass through encoder
    - _extract_batch_data(): Extract data from batch
    - _extract_classification_features(): Extract features for classification
    - _extract_dann_features(): Extract features for domain classification
    """

    def __init__(
        self,
        lr: float = 1e-4,
        gamma: float = 0.1,
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
            gamma: DANN loss weight
            scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs
            max_steps: Maximum number of training steps
            warmup_steps: Number of warmup steps
            train_batch_size: Training batch size
            num_classes: Number of classes for classification
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Initialize shared encoder (no private encoders for DANN)
        self.shared_encoder = nn.Identity()

        # Initialize classifier
        # Subclasses should initialize self.classifier = nn.Linear(feature_dim, num_classes)
        self.classifier = None

        # Initialize domain classifier
        # Subclasses should initialize self.domain_classifier = nn.Linear(feature_dim, 2)
        self.domain_classifier = None

        # Initialize loss functions
        self.loss_classification_fn = nn.CrossEntropyLoss()
        self.loss_dann = nn.CrossEntropyLoss()

    @abstractmethod
    def _forward_encoder(self, encoder, data: dict | torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _forward_encoder")

    @abstractmethod
    def _extract_batch_data(self, batch: dict) -> tuple:
        """Extract data from batch. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_batch_data")

    @abstractmethod
    def _extract_classification_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Extract features for classification. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_classification_features")

    @abstractmethod
    def _extract_dann_features(self, shared_features: torch.Tensor) -> torch.Tensor:
        """Extract features for domain classification. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_dann_features")

    def _compute_gradient_reversal_factor(self, batch_idx: int, stage: str) -> float:
        """
        Compute gradient reversal scaling factor p.

        Args:
            batch_idx (int): Batch index.
            stage (str): Stage name ('train', 'val', 'test').

        Returns:
            float: Gradient reversal scaling factor.
        """
        if stage == "train":
            p = float(batch_idx + self.current_epoch * len(self.trainer.datamodule.train_dataloader())) / (
                self.trainer.max_epochs * len(self.trainer.datamodule.train_dataloader())
            )
            p = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        elif stage == "val":
            p = float(batch_idx + self.current_epoch * len(self.trainer.datamodule.val_dataloader())) / (
                self.trainer.max_epochs * len(self.trainer.datamodule.val_dataloader())
            )
            p = 2.0 / (1.0 + np.exp(-10 * p)) - 1.0
        else:  # test
            p = 1.0
        return p

    def _source_domain_step(
        self,
        source_data: dict | torch.Tensor,
        source_labels: torch.Tensor,
        batch_idx: int,
        stage: str,
    ) -> dict:
        """
        Process source domain: forward pass and compute losses.

        Args:
            source_data: Source domain input data.
            source_labels: Source domain labels.
            batch_idx: Batch index.
            stage: Stage name ('train', 'val', 'test').

        Returns:
            dict: Dictionary containing source domain results and losses.
        """
        # Domain labels: 0 for source
        source_domain_labels = torch.zeros(source_labels.size(0), dtype=torch.long, device=self.device)

        # Gradient reversal scaling factor
        p = self._compute_gradient_reversal_factor(batch_idx, stage)

        # Forward pass for source domain
        s_shared = self._forward_encoder(self.shared_encoder, source_data)
        s_pred = self.classifier(self._extract_classification_features(s_shared))

        # Compute source domain losses
        source_losses = self.compute_losses(
            shared=s_shared,
            pred=s_pred,
            labels=source_labels,
            domain_labels=source_domain_labels,
            p=p,
        )

        return {
            "shared": s_shared,
            "pred": s_pred,
        } | source_losses

    def _target_domain_step(
        self,
        target_data: dict | torch.Tensor,
        target_labels: torch.Tensor,
        batch_idx: int,
        stage: str,
    ) -> dict:
        """
        Process target domain: forward pass and compute losses.

        Args:
            target_data: Target domain input data.
            target_labels: Target domain labels.
            batch_idx: Batch index.
            stage: Stage name ('train', 'val', 'test').

        Returns:
            dict: Dictionary containing target domain results and losses.
        """
        # Domain labels: 1 for target
        target_domain_labels = torch.ones(target_labels.size(0), dtype=torch.long, device=self.device)

        # Gradient reversal scaling factor
        p = self._compute_gradient_reversal_factor(batch_idx, stage)

        # Forward pass for target domain
        t_shared = self._forward_encoder(self.shared_encoder, target_data)

        if stage == "test":
            t_pred = self.classifier(self._extract_classification_features(t_shared))
        else:
            t_pred = None

        # Compute target domain losses
        target_losses = self.compute_losses(
            shared=t_shared,
            pred=t_pred,
            labels=target_labels,
            domain_labels=target_domain_labels,
            p=p,
        )

        return {
            "shared": t_shared,
            "pred": t_pred,
        } | target_losses

    def compute_dann_loss(
        self,
        shared_features: dict | torch.Tensor,
        domain_labels: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """
        Compute DANN loss using Gradient Reversal Layer (GRL).

        Args:
            shared_features: Shared feature representations (dict for scGPT model, torch.Tensor for other models).
            domain_labels (torch.Tensor): Domain labels (0 for source, 1 for target).
            p (float): Gradient reversal scaling factor.

        Returns:
            torch.Tensor: DANN loss.
        """
        # Extract features for DANN (handles both dict and tensor inputs)
        dann_features = self._extract_dann_features(shared_features)
        reversed_features = GradReverse.apply(dann_features, p)
        domain_preds = self.domain_classifier(reversed_features)
        return self.loss_dann(domain_preds, domain_labels)

    def loss_classification(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss for base model.
        """
        return self.loss_classification_fn(pred, labels)

    def compute_losses(
        self,
        shared: torch.Tensor,
        pred: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        p: float = 1.0,
    ) -> dict:
        """
        Compute all losses for DANN model.

        Uses classification loss and DANN adversarial loss.
        """
        # Classification loss (only for source domain during training)
        if pred is not None and labels is not None:
            class_loss = self.loss_classification_fn(pred, labels)
        else:
            class_loss = torch.tensor(0.0, device=self.device)

        # DANN loss
        dann_loss = self.compute_dann_loss(shared, domain_labels, p)

        # Total loss
        total_loss = class_loss + self.hparams.gamma * dann_loss

        return {
            "class_loss": class_loss,
            "dann_loss": dann_loss,
            "total_loss": total_loss,
        }

    def _build_log_dict(
        self,
        stage: str,
        source_results: dict,
        target_results: dict,
        source_labels: torch.Tensor,
        target_labels: torch.Tensor,
    ) -> dict:
        """
        Build log dictionary for logging metrics and losses.

        Args:
            stage: Stage name ('train', 'val', 'test').
            source_results: Source domain results from _source_domain_step.
            target_results: Target domain results from _target_domain_step.
            source_labels: Source domain labels.
            target_labels: Target domain labels.

        Returns:
            dict: Log dictionary with all metrics and losses.
        """
        log_dict = {}

        if "train" == stage or "val" == stage:
            log_dict[f"{stage}/source_class_loss"] = source_results["class_loss"]
            log_dict[f"{stage}/source_dann_loss"] = source_results["dann_loss"]
            log_dict[f"{stage}/target_dann_loss"] = target_results["dann_loss"]

            total_loss = source_results["total_loss"] + target_results["total_loss"]
            log_dict[f"{stage}/total_loss"] = total_loss
            log_dict[f"{stage}/source_accuracy"] = compute_accuracy(source_results["pred"], source_labels)
            log_dict[f"{stage}/target_accuracy"] = torch.tensor(0.0, device=self.device)
        elif stage == "test":
            s_metrics = compute_metrics(source_results["pred"], source_labels)
            if target_results["pred"] is not None:
                t_metrics = compute_metrics(target_results["pred"], target_labels)
            else:
                t_metrics = {}
            for metric in ["accuracy", "auc", "auc_macro", "auc_weighted", "f1_macro", "f1_micro", "f1_weighted"]:
                for domain in ["source", "target"]:
                    metrics = s_metrics if domain == "source" else t_metrics
                    if metric in metrics:
                        log_dict[f"{domain}_{metric}"] = metrics[metric]

        return log_dict

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Training step for a single batch."""
        opt = self.optimizers()

        (
            source_data,
            source_labels,
            target_data,
            target_labels,
            _,
            _,
        ) = self._extract_batch_data(batch)

        # Source Domain
        source_results = self._source_domain_step(
            source_data,
            source_labels,
            batch_idx,
            stage="train",
        )
        opt.zero_grad()
        self.manual_backward(source_results["total_loss"])
        self.opt_step(opt)

        # Target Domain
        target_results = self._target_domain_step(
            target_data,
            target_labels,
            batch_idx,
            stage="train",
        )
        opt.zero_grad()
        self.manual_backward(target_results["total_loss"])
        self.opt_step(opt)

        # Log losses
        log_dict = self._build_log_dict(
            stage="train",
            source_results=source_results,
            target_results=target_results,
            source_labels=source_labels,
            target_labels=target_labels,
        )

        self.log_dict(log_dict, on_step=True, on_epoch=True)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch."""
        (
            source_data,
            source_labels,
            target_data,
            target_labels,
            _,
            _,
        ) = self._extract_batch_data(batch)

        source_results = self._source_domain_step(source_data, source_labels, batch_idx, stage="val")

        target_results = self._target_domain_step(target_data, target_labels, batch_idx, stage="val", compute_pred=True)

        log_dict = self._build_log_dict(
            stage="val",
            source_results=source_results,
            target_results=target_results,
            source_labels=source_labels,
            target_labels=target_labels,
        )

        self.log_dict(log_dict, on_step=False, on_epoch=True)

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx=0) -> torch.Tensor:
        """Test step for a single batch with comprehensive metrics."""
        (
            source_data,
            source_labels,
            target_data,
            target_labels,
            _,
            _,
        ) = self._extract_batch_data(batch)

        source_results = self._source_domain_step(source_data, source_labels, batch_idx, stage="test")

        target_results = self._target_domain_step(
            target_data, target_labels, batch_idx, stage="test", compute_pred=True
        )

        log_dict = self._build_log_dict(
            stage="test",
            source_results=source_results,
            target_results=target_results,
            source_labels=source_labels,
            target_labels=target_labels,
        )

        self.log_dict(log_dict, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        self._calculate_scheduler_steps()

        if self.hparams.scheduler_type == "constant":
            self.lr_scheduler = None
        elif self.hparams.scheduler_type == "warmupcosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.hparams.max_steps,
            )
        elif self.hparams.scheduler_type == "cosine":
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=self.hparams.max_steps,
            )
        else:
            raise ValueError(f"Invalid scheduler type: {self.hparams.scheduler_type}")

        return optimizer

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
            self.hparams.max_steps = (
                len(train_dataloader) * self.hparams.num_epochs * 2
            )  # because optimizer step is called twice
            self.hparams.warmup_steps = (
                int(self.hparams.warmup_ratio * self.hparams.max_steps) * 2
            )  # because optimizer step is called twice

    def on_train_start(self):
        """Called when training starts. Recalculate scheduler steps if needed."""
        super().on_train_start()
        if self.hparams.max_steps is None:
            self._calculate_scheduler_steps()
            if self.hparams.max_steps is not None and self.lr_scheduler is None:
                optimizer = self.optimizers()
                if self.hparams.scheduler_type == "warmupcosine":
                    self.lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=self.hparams.warmup_steps,
                        num_training_steps=self.hparams.max_steps,
                    )
                elif self.hparams.scheduler_type == "cosine":
                    self.lr_scheduler = get_cosine_schedule_with_warmup(
                        optimizer,
                        num_warmup_steps=0,
                        num_training_steps=self.hparams.max_steps,
                    )

    def opt_step(self, opt: torch.optim.Optimizer):
        """Step the optimizer and learning rate scheduler."""
        opt.step()
        if hasattr(self, "lr_scheduler") and self.lr_scheduler is not None:
            self.lr_scheduler.step()
        self.log_dict(
            {
                "learning_rate": opt.param_groups[0]["lr"],
            },
            on_step=True,
            on_epoch=False,
        )
