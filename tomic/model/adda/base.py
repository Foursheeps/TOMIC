from abc import ABC, abstractmethod

import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from ...utils import compute_accuracy, compute_metrics
from ..dsn.loss import GradReverse

"""
Base Lightning Module for Adversarial Discriminative Domain Adaptation (ADDA).

ADDA uses a two-stage training approach:
1. Pre-train source encoder + classifier on source domain
2. Adversarially train target encoder against discriminator (source encoder frozen)
"""

torch.set_float32_matmul_precision("medium")


class BaseLightningModule(pl.LightningModule, ABC):
    """
    Base Lightning Module for Adversarial Discriminative Domain Adaptation.

    This class provides shared functionality for all model types:
    - Two-stage training (pre-train and adversarial)
    - Loss computation
    - Training/validation/test steps
    - Optimizer and scheduler configuration
    - Domain-specific forward passes
    - Logging and metrics

    Subclasses should implement:
    - _forward_encoder(): Forward pass through encoder
    - _extract_batch_data(): Extract data from batch
    - _extract_classification_features(): Extract features for classification
    - _extract_discriminator_features(): Extract features for discriminator
    """

    def __init__(
        self,
        lr: float = 1e-4,
        pretrain_epochs: int = 80,
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
            pretrain_epochs: Number of epochs for source domain pre-training
            scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs (for adversarial stage)
            max_steps: Maximum number of training steps
            warmup_steps: Number of warmup steps
            train_batch_size: Training batch size
            num_classes: Number of classes for classification
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Initialize source encoder (frozen after pre-training)
        self.source_encoder = nn.Identity()

        # Initialize target encoder (trained adversarially)
        self.target_encoder = nn.Identity()

        # Initialize classifier (trained with source encoder)
        # Subclasses should initialize self.classifier = nn.Linear(feature_dim, num_classes)
        self.classifier = None

        # Initialize discriminator (distinguishes source vs target features)
        # Subclasses should initialize self.discriminator with appropriate input dimension
        self.discriminator = None

        # Initialize loss functions
        self.loss_classification = nn.CrossEntropyLoss()
        self.loss_discriminator = nn.BCEWithLogitsLoss()

        # Track training stage
        self.pretrain_stage = True

    @abstractmethod
    def _forward_encoder(self, encoder, data: dict | torch.Tensor) -> torch.Tensor:
        """Forward pass through encoder. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _forward_encoder")

    @abstractmethod
    def _extract_batch_data(self, batch: dict) -> tuple:
        """Extract data from batch. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_batch_data")

    @abstractmethod
    def _extract_classification_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract features for classification. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_classification_features")

    @abstractmethod
    def _extract_discriminator_features(self, features: torch.Tensor) -> torch.Tensor:
        """Extract features for discriminator. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_discriminator_features")

    def _pretrain_source_step(
        self,
        source_data: dict | torch.Tensor,
        source_labels: torch.Tensor,
    ) -> dict:
        """
        Pre-train source encoder and classifier on source domain.

        Args:
            source_data: Source domain input data.
            source_labels: Source domain labels.

        Returns:
            dict: Dictionary containing source domain results and losses.
        """
        # Forward pass for source domain
        s_features = self._forward_encoder(self.source_encoder, source_data)
        s_pred = self.classifier(self._extract_classification_features(s_features))

        # Classification loss
        class_loss = self.loss_classification(s_pred, source_labels)

        return {
            "features": s_features,
            "pred": s_pred,
            "class_loss": class_loss,
            "total_loss": class_loss,
        }

    def _adversarial_step(
        self,
        source_data: dict | torch.Tensor,
        target_data: dict | torch.Tensor,
        batch_idx: int,
        stage: str,
    ) -> dict:
        """
        Adversarial training step: train target encoder and discriminator.

        Args:
            source_data: Source domain input data.
            target_data: Target domain input data.
            batch_idx: Batch index.
            stage: Stage name ('train', 'val', 'test').

        Returns:
            dict: Dictionary containing results and losses.
        """
        # Get batch size
        if isinstance(source_data, torch.Tensor):
            batch_size = source_data.size(0)
        elif isinstance(source_data, dict) and "input_ids" in source_data:
            batch_size = source_data["input_ids"].size(0)
        else:
            # Fallback: try to get from first tensor in dict
            batch_size = next(iter(source_data.values())).size(0) if isinstance(source_data, dict) else 1

        # Forward pass: source features (frozen)
        with torch.no_grad():
            s_features = self._forward_encoder(self.source_encoder, source_data)
        s_features_detached = s_features.detach()

        # Forward pass: target features (trainable)
        t_features = self._forward_encoder(self.target_encoder, target_data)

        # Discriminator features
        s_disc_features = self._extract_discriminator_features(s_features_detached)
        t_disc_features = self._extract_discriminator_features(t_features)

        # Discriminator predictions (no gradient reversal for discriminator training)
        # Source: label 1, Target: label 0
        s_disc_pred = self.discriminator(s_disc_features.detach())
        t_disc_pred = self.discriminator(t_disc_features.detach())

        # Discriminator loss (should distinguish source from target)
        s_labels = torch.ones(batch_size, 1, device=self.device)
        t_labels = torch.zeros(batch_size, 1, device=self.device)
        disc_loss = self.loss_discriminator(s_disc_pred, s_labels) + self.loss_discriminator(t_disc_pred, t_labels)

        # Target encoder loss (should fool discriminator - make target look like source)
        # Use gradient reversal to reverse gradients for target encoder
        t_disc_features_reversed = GradReverse.apply(t_disc_features, 1.0)
        t_disc_pred_fool = self.discriminator(t_disc_features_reversed)
        target_encoder_loss = self.loss_discriminator(t_disc_pred_fool, s_labels)

        return {
            "s_features": s_features_detached,
            "t_features": t_features,
            "disc_loss": disc_loss,
            "target_encoder_loss": target_encoder_loss,
            "total_loss": disc_loss + target_encoder_loss,
        }

    def _build_log_dict(
        self,
        stage: str,
        source_results: dict | None = None,
        adversarial_results: dict | None = None,
        source_labels: torch.Tensor | None = None,
    ) -> dict:
        """
        Build log dictionary for logging metrics and losses.

        Args:
            stage: Stage name ('train', 'val', 'test').
            source_results: Source domain results from _pretrain_source_step.
            adversarial_results: Adversarial training results from _adversarial_step.
            source_labels: Source domain labels.

        Returns:
            dict: Log dictionary with all metrics and losses.
        """
        log_dict = {}

        if self.pretrain_stage:
            if source_results is not None:
                log_dict[f"{stage}/pretrain_class_loss"] = source_results["class_loss"]
                log_dict[f"{stage}/pretrain_total_loss"] = source_results["total_loss"]
                if source_labels is not None:
                    log_dict[f"{stage}/pretrain_accuracy"] = compute_accuracy(source_results["pred"], source_labels)
        else:
            if adversarial_results is not None:
                log_dict[f"{stage}/disc_loss"] = adversarial_results["disc_loss"]
                log_dict[f"{stage}/target_encoder_loss"] = adversarial_results["target_encoder_loss"]
                log_dict[f"{stage}/adversarial_total_loss"] = adversarial_results["total_loss"]

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

        if self.pretrain_stage:
            # Pre-train stage: train source encoder + classifier
            source_results = self._pretrain_source_step(source_data, source_labels)
            opt.zero_grad()
            self.manual_backward(source_results["total_loss"])
            self.opt_step(opt)

            # Check if we should switch to adversarial stage
            if self.current_epoch >= self.hparams.pretrain_epochs:
                self.pretrain_stage = False
                # Freeze source encoder
                for param in self.source_encoder.parameters():
                    param.requires_grad = False
                # Create separate optimizers for target encoder and discriminator
                self._setup_adversarial_optimizers()

            log_dict = self._build_log_dict(
                stage="train",
                source_results=source_results,
                source_labels=source_labels,
            )
        else:
            # Adversarial stage: train target encoder and discriminator
            adversarial_results = self._adversarial_step(source_data, target_data, batch_idx, stage="train")

            # Update target encoder (fool discriminator)
            opt_target = self.target_optimizer
            opt_target.zero_grad()
            self.manual_backward(adversarial_results["target_encoder_loss"])
            opt_target.step()

            # Update discriminator (distinguish source from target)
            opt_disc = self.discriminator_optimizer
            opt_disc.zero_grad()
            self.manual_backward(adversarial_results["disc_loss"])
            opt_disc.step()

            log_dict = self._build_log_dict(
                stage="train",
                adversarial_results=adversarial_results,
            )

        self.log_dict(log_dict, on_step=True, on_epoch=True)

    def _setup_adversarial_optimizers(self):
        """Setup separate optimizers for target encoder and discriminator."""
        self.target_optimizer = torch.optim.Adam(self.target_encoder.parameters(), lr=self.hparams.lr)
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters(), lr=self.hparams.lr)

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

        if self.pretrain_stage:
            source_results = self._pretrain_source_step(source_data, source_labels)
            log_dict = self._build_log_dict(
                stage="val",
                source_results=source_results,
                source_labels=source_labels,
            )
        else:
            adversarial_results = self._adversarial_step(source_data, target_data, batch_idx, stage="val")

            # Compute predictions for evaluation
            with torch.no_grad():
                s_features = self._forward_encoder(self.source_encoder, source_data)
                t_features = self._forward_encoder(self.target_encoder, target_data)
                s_pred = self.classifier(self._extract_classification_features(s_features))
                t_pred = self.classifier(self._extract_classification_features(t_features))

            adversarial_results["s_pred"] = s_pred
            adversarial_results["t_pred"] = t_pred

            log_dict = self._build_log_dict(
                stage="val",
                adversarial_results=adversarial_results,
            )
            log_dict["val/source_accuracy"] = compute_accuracy(s_pred, source_labels)
            log_dict["val/target_accuracy"] = compute_accuracy(t_pred, target_labels)

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

        # Compute predictions
        with torch.no_grad():
            s_features = self._forward_encoder(self.source_encoder, source_data)
            t_features = self._forward_encoder(self.target_encoder, target_data)
            s_pred = self.classifier(self._extract_classification_features(s_features))
            t_pred = self.classifier(self._extract_classification_features(t_features))

        s_metrics = compute_metrics(s_pred, source_labels)
        t_metrics = compute_metrics(t_pred, target_labels)

        log_dict = {}
        for metric in ["accuracy", "auc", "auc_macro", "auc_weighted", "f1_macro", "f1_micro", "f1_weighted"]:
            for domain in ["source", "target"]:
                metrics = s_metrics if domain == "source" else t_metrics
                if metric in metrics:
                    log_dict[f"{domain}_{metric}"] = metrics[metric]

        self.log_dict(log_dict, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        # During pre-train stage, optimize source encoder + classifier
        optimizer = torch.optim.Adam(
            list(self.source_encoder.parameters()) + list(self.classifier.parameters()), lr=self.hparams.lr
        )

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
            # For pre-train stage
            self.hparams.max_steps = len(train_dataloader) * self.hparams.pretrain_epochs
            self.hparams.warmup_steps = int(self.hparams.warmup_ratio * self.hparams.max_steps)

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
