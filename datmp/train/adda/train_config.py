"""
Training configuration classes for ADDA training scripts.

This module provides configuration dataclasses for training hyperparameters.
ADDA uses a two-stage training approach:
1. Pre-train source encoder + classifier on source domain
2. Adversarially train target encoder against discriminator (source encoder frozen)
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainerConfig:
    """Training and trainer configuration parameters for ADDA."""

    def __init__(
        self,
        # Basic training hyperparameters
        lr: float = 1e-4,
        max_epochs: int = 200,
        pretrain_epochs: int = 80,  # Number of epochs for source domain pre-training
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        # Data loading
        train_batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 4,
        # PyTorch Lightning Trainer configuration
        seed: int = 2025,
        devices: int = 1,
        precision: str = "16-mixed",
        log_every_n_steps: int = 50,
        val_check_interval: int = 200,
        check_val_every_n_epoch: int | None = None,
        # General callback configuration
        patience: int = 10,
        monitor_metric: str = "val/target_accuracy",
        mode: str = "max",
        save_top_k: int = 2,
        # Pretrain stage early stopping configuration
        pretrain_patience: int | None = None,
        pretrain_monitor_metric: str = "val/pretrain_accuracy",
        pretrain_mode: str = "max",
        # Adversarial stage early stopping configuration
        adversarial_patience: int | None = None,
        adversarial_monitor_metric: str | None = None,
        adversarial_mode: str | None = None,
        # Output directory configuration
        default_root_dir: Path | str | None = None,
        # Control flags
        run_training: bool = True,
        run_testing: bool = True,
        checkpoint_path: str | None = None,
        **kwargs,
    ):
        """
        Initialize TrainerConfig for ADDA.

        Args:
            lr: Learning rate
            max_epochs: Total number of training epochs (pretrain_epochs + adversarial epochs)
            pretrain_epochs: Number of epochs for source domain pre-training
            scheduler_type: Learning rate scheduler type ("warmupcosine", "cosine", or None)
            warmup_ratio: Warmup ratio for scheduler
            train_batch_size: Training batch size
            test_batch_size: Test batch size
            num_workers: Number of data loading workers
            seed: Random seed
            devices: Number of devices
            precision: Training precision ("16-mixed", "32", "bf16-mixed")
            log_every_n_steps: Log every N steps
            val_check_interval: Validation check interval
            check_val_every_n_epoch: Check validation every N epochs
            patience: Default patience for early stopping
            monitor_metric: Default metric to monitor
            mode: Default mode for monitoring ("max" or "min")
            save_top_k: Number of top checkpoints to save
            pretrain_patience: Patience for pretrain stage (defaults to patience)
            pretrain_monitor_metric: Metric to monitor during pretrain stage
            pretrain_mode: Mode for pretrain stage monitoring ("max" or "min")
            adversarial_patience: Patience for adversarial stage (defaults to patience)
            adversarial_monitor_metric: Metric to monitor during adversarial stage (defaults to monitor_metric)
            adversarial_mode: Mode for adversarial stage monitoring (defaults to mode)
            default_root_dir: Root directory for checkpoints and logs
            run_training: Whether to run training
            run_testing: Whether to run testing
            checkpoint_path: Path to checkpoint for testing
        """
        # Basic training hyperparameters
        self.lr = lr
        self.max_epochs = max_epochs
        self.pretrain_epochs = pretrain_epochs

        # Learning rate scheduler
        self.scheduler_type = scheduler_type
        self.warmup_ratio = warmup_ratio

        # Data loading
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        # PyTorch Lightning Trainer configuration
        self.seed = seed
        self.devices = devices
        self.precision = precision
        self.log_every_n_steps = log_every_n_steps
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch

        # General callback configuration
        self.patience = patience
        self.monitor_metric = monitor_metric
        self.mode = mode
        self.save_top_k = save_top_k

        # Pretrain stage early stopping configuration
        self.pretrain_patience = pretrain_patience if pretrain_patience is not None else patience
        self.pretrain_monitor_metric = pretrain_monitor_metric
        self.pretrain_mode = pretrain_mode

        # Adversarial stage early stopping configuration
        self.adversarial_patience = adversarial_patience if adversarial_patience is not None else patience
        self.adversarial_monitor_metric = (
            adversarial_monitor_metric if adversarial_monitor_metric is not None else monitor_metric
        )
        self.adversarial_mode = adversarial_mode if adversarial_mode is not None else mode

        # Output directory configuration
        self.default_root_dir = default_root_dir

        # Control flags
        self.run_training = run_training
        self.run_testing = run_testing
        self.checkpoint_path = checkpoint_path
