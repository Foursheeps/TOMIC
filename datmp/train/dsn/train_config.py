"""
Training configuration classes for DSN training scripts.

This module provides configuration dataclasses for training hyperparameters.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainerConfig:
    """Training and trainer configuration parameters."""

    def __init__(
        self,
        lr: float = 1e-4,
        max_epochs: int = 200,
        scheduler_type: str = "warmupcosine",
        warmup_ratio: float = 0.1,
        train_batch_size: int = 128,
        test_batch_size: int = 128,
        num_workers: int = 4,
        seed: int = 2025,
        devices: int = 1,
        precision: str = "16-mixed",
        log_every_n_steps: int = 50,
        val_check_interval: int = 200,
        check_val_every_n_epoch: int | None = None,
        patience: int = 10,
        monitor_metric: str = "val/target_accuracy",
        mode: str = "max",
        save_top_k: int = 2,
        alpha: float = 4.0,
        beta: float = 0.25,
        gamma: float = 0.1,
        default_root_dir: Path | str | None = None,
        run_training: bool = True,
        run_testing: bool = True,
        checkpoint_path: str | None = None,
        **kwargs,
    ):
        # Training hyperparameters
        self.lr = lr
        self.max_epochs = max_epochs

        # Learning rate scheduler
        self.scheduler_type = scheduler_type  # "warmupcosine", "cosine", or None
        self.warmup_ratio = warmup_ratio

        # Batch sizes
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers

        # PyTorch Lightning Trainer configuration
        self.seed = seed
        self.devices = devices
        self.precision = precision  # "16-mixed", "32", "bf16-mixed"
        self.log_every_n_steps = log_every_n_steps
        self.val_check_interval = val_check_interval
        self.check_val_every_n_epoch = check_val_every_n_epoch

        # Callback configuration
        self.patience = patience
        self.monitor_metric = monitor_metric
        self.mode = mode  # "max" or "min"
        self.save_top_k = save_top_k

        # Loss function weights (Domain Separation Network)
        self.alpha = alpha  # Reconstruction loss weight
        self.beta = beta  # Difference loss weight (private vs shared features)
        self.gamma = gamma  # DANN loss weight (domain adversarial loss)

        # Output directory configuration
        self.default_root_dir = default_root_dir  # Root directory for checkpoints and logs

        # Control flags
        self.run_training = run_training
        self.run_testing = run_testing
        self.checkpoint_path = checkpoint_path

    def __repr__(self) -> str:
        return f"""
        TrainerConfig(
            lr={self.lr},
            max_epochs={self.max_epochs},
            scheduler_type={self.scheduler_type},
            warmup_ratio={self.warmup_ratio},
            train_batch_size={self.train_batch_size},
            test_batch_size={self.test_batch_size},
            num_workers={self.num_workers},
            seed={self.seed},
            devices={self.devices},
            precision={self.precision},
            log_every_n_steps={self.log_every_n_steps},
            val_check_interval={self.val_check_interval},
            check_val_every_n_epoch={self.check_val_every_n_epoch},
            patience={self.patience},
            monitor_metric={self.monitor_metric},
            mode={self.mode},
            save_top_k={self.save_top_k},
            alpha={self.alpha},
            beta={self.beta},
            gamma={self.gamma},
            default_root_dir={self.default_root_dir},
            run_training={self.run_training},
            run_testing={self.run_testing},
            checkpoint_path={self.checkpoint_path},
        )
        """
