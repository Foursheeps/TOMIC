from abc import ABC, abstractmethod

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from transformers import get_cosine_schedule_with_warmup

from ...utils import compute_accuracy, compute_metrics
from .loss import MSE, SIMSE, DiffLoss, GradReverse

"""
Base Lightning Module for Domain Separation Networks.

This base class contains all shared functionality across different model types.
Individual model implementations should inherit from this class and override
model-specific methods.
"""

torch.set_float32_matmul_precision("medium")


class BaseLightningModule(pl.LightningModule, ABC):
    """
    Base Lightning Module for Domain Separation Networks.

    This class provides shared functionality for all model types:
    - Loss computation
    - Training/validation/test steps
    - Optimizer and scheduler configuration
    - Domain-specific forward passes
    - Logging and metrics

    Subclasses should implement:
    - _create_encoder(): Create model-specific encoder
    - _create_decoder(): Create model-specific decoder
    - _forward_encoder(): Forward pass through encoder
    - _extract_batch_data(): Extract data from batch
    - _combine_features(): Combine private and shared features
    - _extract_classification_features(): Extract features for classification
    - compute_losses(): Compute losses (can override for model-specific loss computation)
    """

    def __init__(
        self,
        lr: float = 1e-4,
        alpha: float = 0.25,
        beta: float = 0.25,
        gamma: float = 0.1,
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        num_epochs: int = 100,
        max_steps: int | None = None,
        warmup_steps: int | None = None,
        train_batch_size: int = 32,
        num_classes: int = None,
        **kwargs,
    ) -> None:
        """
        Initialize base Lightning Module.

        Args:
            lr: Learning rate
            alpha: Reconstruction loss weight
            beta: Difference loss weight
            gamma: DANN loss weight
            scheduler_type: Learning rate scheduler type
            warmup_ratio: Warmup ratio for scheduler
            num_epochs: Number of training epochs
            max_steps: Maximum number of training steps
            warmup_steps: Number of warmup steps
            train_batch_size: Training batch size
            num_classes: Number of classes for classification
            **kwargs: Additional model-specific parameters (will be passed to subclasses)
        """
        super().__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        # Initialize Encoders
        self.private_source_encoder = nn.Identity()
        self.private_target_encoder = nn.Identity()
        self.shared_encoder = nn.Identity()
        self.reconstructor = nn.Identity()

        # Initialize loss functions
        self.loss_dann = nn.CrossEntropyLoss()

    @abstractmethod
    def _forward_encoder(self, encoder, data: dict | torch.Tensor) -> torch.Tensor | dict:
        """Forward pass through encoder. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _forward_encoder")

    @abstractmethod
    def _extract_batch_data(self, batch: dict) -> tuple:
        """Extract data from batch. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_batch_data")

    @abstractmethod
    def _combine_features(self, private: torch.Tensor | dict, shared: torch.Tensor | dict) -> torch.Tensor | dict:
        """Combine private and shared features. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _combine_features")

    @abstractmethod
    def _extract_classification_features(self, shared_features: torch.Tensor | dict) -> torch.Tensor:
        """Extract features for classification. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_classification_features")

    @abstractmethod
    def _extract_dann_features(self, shared_features: torch.Tensor | dict) -> torch.Tensor:
        """Extract features for domain classification. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _extract_dann_features")

    @abstractmethod
    def _gather_diff_features(self, private: torch.Tensor | dict, shared: torch.Tensor | dict) -> tuple:
        """Gather difference features. Must be implemented by subclass."""
        raise NotImplementedError("Subclass must implement _gather_diff_features")

    # @torch.no_grad()
    # def get_embeddings(self, batch: dict) -> dict:
    #     """
    #     Get embeddings using shared encoder.

    #     Args:
    #         batch: Batch dictionary from DatasetTomic containing:
    #             - s_cell_id, t_cell_id: Cell identifiers
    #             - s_gene_ids, t_gene_ids: Gene ID sequences
    #             - s_expr_ids, t_expr_ids: Expression IDs (if binned)
    #             - s_expr, t_expr: Expression values (if not binned)
    #             - s_label, t_label: Labels

    #     Returns:
    #         Dictionary containing:
    #             - source_cls_features: Source domain classification features
    #             - target_cls_features: Target domain classification features
    #             - source_prob: Source domain class probabilities
    #             - target_prob: Target domain class probabilities
    #             - source_pred: Source domain predictions
    #             - target_pred: Target domain predictions
    #     """
    #     source_expr, _, target_expr, _, _, _ = self._extract_batch_data(batch)

    #     # Forward through shared encoder
    #     source_shared_features = self._forward_encoder(self.shared_encoder, source_expr)
    #     target_shared_features = self._forward_encoder(self.shared_encoder, target_expr)
    #     source_cls_features = self._extract_classification_features(source_shared_features)
    #     target_cls_features = self._extract_classification_features(target_shared_features)

    #     # Get logits from classifier
    #     source_logits = self.classifier(source_cls_features)
    #     target_logits = self.classifier(target_cls_features)

    #     # Apply softmax to get probabilities
    #     source_prob = nn.functional.softmax(source_logits, dim=1)
    #     target_prob = nn.functional.softmax(target_logits, dim=1)

    #     # Get predictions
    #     source_pred = torch.argmax(source_logits, dim=1)
    #     target_pred = torch.argmax(target_logits, dim=1)

    #     return {
    #         "source_cls_features": source_cls_features.cpu(),
    #         "target_cls_features": target_cls_features.cpu(),
    #         "source_prob": source_prob.cpu(),
    #         "target_prob": target_prob.cpu(),
    #         "source_pred": source_pred.cpu(),
    #         "target_pred": target_pred.cpu(),
    #     }
    @torch.no_grad()
    def get_embeddings(self, batch: torch.Tensor) -> dict:
        """
        Get embeddings using shared encoder.
        """
        shared = self._forward_encoder(self.shared_encoder, batch)
        cls_features = self._extract_classification_features(shared)
        prob = self.classifier(cls_features)
        prob = nn.functional.softmax(prob, dim=1)
        pred = torch.argmax(prob, dim=1)
        return {
            "features": cls_features,
            "prob": prob,
            "pred": pred,
        }

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
        source_original: torch.Tensor | dict,
        batch_idx: int,
        stage: str,
    ) -> dict:
        """
        Process source domain: forward pass and compute losses.

        Args:
            source_data: Source domain input data.
            source_labels: Source domain labels.
            source_original: Original source data for reconstruction loss.
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
        s_private = self._forward_encoder(self.private_source_encoder, source_data)
        s_shared = self._forward_encoder(self.shared_encoder, source_data)
        s_combined = self._combine_features(s_private, s_shared)
        s_reconstructed = self._reconstruct(s_combined)
        s_pred = self.classifier(self._extract_classification_features(s_shared))

        # Compute source domain losses
        source_losses = self.compute_losses(
            private=s_private,
            shared=s_shared,
            reconstructed=s_reconstructed,
            pred=s_pred,
            original=source_original,
            labels=source_labels,
            domain_labels=source_domain_labels,
            p=p,
        )

        return {
            "private": s_private,
            "shared": s_shared,
            "reconstructed": s_reconstructed,
            "pred": s_pred,
        } | source_losses

    def _target_domain_step(
        self,
        target_data: dict | torch.Tensor,
        target_labels: torch.Tensor,
        target_original: torch.Tensor | dict,
        batch_idx: int,
        stage: str,
    ) -> dict:
        """
        Process target domain: forward pass and compute losses.

        Args:
            target_data: Target domain input data.
            target_labels: Target domain labels.
            target_original: Original target data for reconstruction loss.
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
        t_private = self._forward_encoder(self.private_target_encoder, target_data)
        t_shared = self._forward_encoder(self.shared_encoder, target_data)
        t_combined = self._combine_features(t_private, t_shared)
        t_reconstructed = self._reconstruct(t_combined)
        if stage == "test":
            t_pred = self.classifier(self._extract_classification_features(t_shared))
        else:
            t_pred = None

        # Compute target domain losses
        target_losses = self.compute_losses(
            private=t_private,
            shared=t_shared,
            reconstructed=t_reconstructed,
            pred=t_pred,
            original=target_original,
            labels=target_labels,
            domain_labels=target_domain_labels,
            p=p,
        )

        return {
            "private": t_private,
            "shared": t_shared,
            "reconstructed": t_reconstructed,
            "pred": t_pred,
        } | target_losses

    def _reconstruct(self, combined_features: torch.Tensor) -> torch.Tensor | tuple:
        """
        Reconstruct input from combined features.

        Args:
            combined_features: Combined private and shared features.

        Returns:
            Reconstructed input (or tuple for fusion model).
        """
        return self.reconstructor(combined_features)

    def compute_dann_loss(
        self,
        shared_features: torch.Tensor,
        domain_labels: torch.Tensor,
        p: float,
    ) -> torch.Tensor:
        """
        Compute DANN loss using Gradient Reversal Layer (GRL) for patch model.

        Args:
            shared_features (torch.Tensor): Shared feature representations.
            domain_labels (torch.Tensor): Domain labels (0 for source, 1 for target).
            p (float): Gradient reversal scaling factor.

        Returns:
            torch.Tensor: DANN loss.
        """
        reversed_features = GradReverse.apply(shared_features, p)
        domain_features = self._extract_dann_features(reversed_features)
        domain_preds = self.domain_classifier(domain_features)
        return self.loss_dann(domain_preds, domain_labels)

    def loss_diff(self, private: torch.Tensor, shared: torch.Tensor) -> torch.Tensor:
        """
        Compute difference loss for base model.
        """
        s_diff_private, s_diff_shared = self._gather_diff_features(private, shared)
        return DiffLoss()(s_diff_private, s_diff_shared)

    def loss_recon(
        self,
        reconstructed: torch.Tensor,
        original: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute reconstruction loss for base model.
        """
        return MSE()(reconstructed, original) + SIMSE()(reconstructed, original)

    def loss_classification(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Compute classification loss for base model.
        """
        return nn.CrossEntropyLoss()(pred, labels)

    def compute_losses(
        self,
        private: torch.Tensor,
        shared: torch.Tensor,
        reconstructed: torch.Tensor,
        pred: torch.Tensor | None = None,
        original: torch.Tensor | dict | None = None,
        labels: torch.Tensor | None = None,
        domain_labels: torch.Tensor | None = None,
        p: float = 1.0,
    ) -> dict:
        """
        Compute all losses for patch model.

        Uses standard MSE + SIMSE reconstruction loss.
        """
        # Classification loss
        if pred is not None and labels is not None:
            class_loss = self.loss_classification(pred, labels)
        else:
            class_loss = torch.tensor(0.0, device=self.device)

        # Reconstruction loss (MSE + SIMSE)
        recon_loss = self.loss_recon(reconstructed, original)

        # Difference loss
        diff_loss = self.loss_diff(private, shared)

        # DANN loss
        dann_loss = self.compute_dann_loss(shared, domain_labels, p)

        # Total loss
        total_loss = (
            class_loss
            + self.hparams.alpha * recon_loss
            + self.hparams.beta * diff_loss
            + self.hparams.gamma * dann_loss
        )

        return {
            "class_loss": class_loss,
            "recon_loss": recon_loss,
            "diff_loss": diff_loss,
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
            log_dict[f"{stage}/source_recon_loss"] = source_results["recon_loss"]
            log_dict[f"{stage}/source_diff_loss"] = source_results["diff_loss"]
            log_dict[f"{stage}/source_dann_loss"] = source_results["dann_loss"]
            log_dict[f"{stage}/target_recon_loss"] = target_results["recon_loss"]
            log_dict[f"{stage}/target_diff_loss"] = target_results["diff_loss"]
            log_dict[f"{stage}/target_dann_loss"] = target_results["dann_loss"]

            total_loss = source_results["total_loss"] + target_results["total_loss"]
            log_dict[f"{stage}/total_loss"] = total_loss
            log_dict[f"{stage}/source_accuracy"] = compute_accuracy(source_results["pred"], source_labels)

            log_dict[f"{stage}/target_accuracy"] = torch.tensor(0.0, device=self.device)
        elif stage == "test":
            s_metrics = compute_metrics(source_results["pred"], source_labels)
            t_metrics = compute_metrics(target_results["pred"], target_labels)
            for metric in ["accuracy", "auc", "auc_macro", "auc_weighted", "f1_macro", "f1_micro", "f1_weighted"]:
                for domain in ["source", "target"]:
                    metrics = s_metrics if domain == "source" else t_metrics
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
            source_original,
            target_original,
        ) = self._extract_batch_data(batch)

        # Source Domain
        source_results = self._source_domain_step(
            source_data,
            source_labels,
            source_original,
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
            target_original,
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

        self.log_dict(log_dict, on_step=True, on_epoch=True, sync_dist=True)

    def validation_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        """Validation step for a single batch."""
        (
            source_data,
            source_labels,
            target_data,
            target_labels,
            source_original,
            target_original,
        ) = self._extract_batch_data(batch)

        source_results = self._source_domain_step(source_data, source_labels, source_original, batch_idx, stage="val")

        target_results = self._target_domain_step(target_data, target_labels, target_original, batch_idx, stage="val")

        log_dict = self._build_log_dict(
            stage="val",
            source_results=source_results,
            target_results=target_results,
            source_labels=source_labels,
            target_labels=target_labels,
        )

        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)

    def test_step(self, batch: dict, batch_idx: int, dataloader_idx=0) -> torch.Tensor:
        """Test step for a single batch with comprehensive metrics."""
        (
            source_data,
            source_labels,
            target_data,
            target_labels,
            source_original,
            target_original,
        ) = self._extract_batch_data(batch)

        source_results = self._source_domain_step(source_data, source_labels, source_original, batch_idx, stage="test")

        target_results = self._target_domain_step(target_data, target_labels, target_original, batch_idx, stage="test")

        log_dict = self._build_log_dict(
            stage="test",
            source_results=source_results,
            target_results=target_results,
            source_labels=source_labels,
            target_labels=target_labels,
        )

        self.log_dict(log_dict, on_step=False, on_epoch=True, sync_dist=True)

    def configure_optimizers(self):
        """Configure the optimizer and learning rate scheduler."""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

        self._calculate_scheduler_steps()

        if self.hparams.scheduler_type == "constant":
            self.lr_scheduler = None
        elif self.hparams.scheduler_type == "warmupcosine":
            # Ensure warmup_steps and max_steps are not None
            # Use a default value if max_steps is not calculated yet
            warmup_steps = self.hparams.warmup_steps if self.hparams.warmup_steps is not None else 0
            max_steps = self.hparams.max_steps if self.hparams.max_steps is not None else 1000
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=warmup_steps,
                num_training_steps=max_steps,
            )
        elif self.hparams.scheduler_type == "cosine":
            max_steps = self.hparams.max_steps if self.hparams.max_steps is not None else 1000
            self.lr_scheduler = get_cosine_schedule_with_warmup(
                optimizer,
                num_warmup_steps=0,
                num_training_steps=max_steps,
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
