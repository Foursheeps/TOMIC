"""
Unified training script for DSN models.

This script trains Domain Separation Networks supporting multiple model types:
- name: Name-based Transformer
- patch: Patch-based Transformer
- mlp: MLP encoder
- expr: Expression-based Transformer
- scgpt: scGPT-based Transformer

Usage:
    python -m tomic.train.dsn.train --model_type name --lr 1e-3
    python -m tomic.train.dsn.train config.json
"""

import json
import re
import sys
from pathlib import Path
from typing import Literal

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from transformers.hf_argparser import HfArgumentParser

from ...dataset.dataconfig import TomicDataConfig
from ...dataset.dataset4da import DomainDataModuleTomic
from ...logger import get_logger
from ...model.dsn import (
    DualTransformerModel4DSN,
    DualTransformerModelConfig4DSN,
    ExprTransformerModel4DSN,
    ExprTransformerModelConfig4DSN,
    MLPModel4DSN,
    MLPModelConfig4DSN,
    NameTransformerModel4DSN,
    NameTransformerModelConfig4DSN,
    PatchTransformerModel4DSN,
    PatchTransformerModelConfig4DSN,
)
from .train_config import TrainerConfig

# Get logger
logger = get_logger("train_dsn")


# Model type to config class mapping
MODEL_CONFIG_MAP = {
    "name": NameTransformerModelConfig4DSN,
    "patch": PatchTransformerModelConfig4DSN,
    "mlp": MLPModelConfig4DSN,
    "expr": ExprTransformerModelConfig4DSN,
    "dual": DualTransformerModelConfig4DSN,
}

# Model type to Lightning module class mapping
MODEL_TYPE_MAP = {
    "name": NameTransformerModel4DSN,
    "patch": PatchTransformerModel4DSN,
    "mlp": MLPModel4DSN,
    "expr": ExprTransformerModel4DSN,
    "dual": DualTransformerModel4DSN,
}
# Model type descriptions
MODEL_DESCRIPTIONS = {
    "name": "Name-based Transformer DSN",
    "patch": "Patch-based Transformer DSN",
    "mlp": "MLP-based DSN",
    "expr": "Expression-based Transformer DSN",
    "dual": "Dual-based Transformer DSN",
}


# Type alias for model config classes
_ConfigT = (
    NameTransformerModelConfig4DSN
    | PatchTransformerModelConfig4DSN
    | MLPModelConfig4DSN
    | ExprTransformerModelConfig4DSN
    | DualTransformerModelConfig4DSN
)
_ModelT = (
    NameTransformerModel4DSN
    | PatchTransformerModel4DSN
    | MLPModel4DSN
    | ExprTransformerModel4DSN
    | DualTransformerModel4DSN
)


# ============================================================================
# Utility Functions
# ============================================================================


def find_checkpoint(save_dir: Path, checkpoint_path: str | None = None) -> Path | None:
    """Find checkpoint file.

    Args:
        save_dir: Save directory
        checkpoint_path: Manually specified checkpoint path

    Returns:
        Checkpoint path if found, None otherwise
    """

    def extract_val_tar_acc_from_filename(filename: str) -> float:
        """Extract val_tar_acc value from checkpoint filename."""
        pattern = r"val_tar_acc=([\d.]+)"
        match = re.search(pattern, filename)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return -1.0
        return -1.0

    if checkpoint_path:
        checkpoint = Path(checkpoint_path)
        if checkpoint.exists():
            return checkpoint
        logger.warning(f"Specified checkpoint not found: {checkpoint_path}")

    # Try to find the best checkpoint in the checkpoint directory
    checkpoint_dir = save_dir  # / "checkpoints"
    if checkpoint_dir.exists():
        best_checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
        if best_checkpoints:
            # Sort by val_tar_acc value in filename (descending)
            best_checkpoints.sort(key=lambda x: extract_val_tar_acc_from_filename(x.name), reverse=True)
            best_checkpoint = best_checkpoints[0]
            best_acc = extract_val_tar_acc_from_filename(best_checkpoint.name)
            logger.info(
                f"Found {len(best_checkpoints)} checkpoint(s). "
                f"Best: {best_checkpoint.name} (val_tar_acc={best_acc:.4f})"
            )
            return best_checkpoint
        else:
            logger.warning(f"No checkpoint found in {checkpoint_dir}.")
            return None
    else:
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None


def create_callbacks(training_args: TrainerConfig) -> tuple[EarlyStopping, ModelCheckpoint]:
    """Create training callbacks.

    Args:
        training_args: Trainer configuration

    Returns:
        Tuple of (early_stop_callback, checkpoint_callback)
    """
    early_stop_callback = EarlyStopping(
        monitor=training_args.monitor_metric,
        mode=training_args.mode,
        patience=training_args.patience,
        verbose=True,
        min_delta=0.0001,
    )

    checkpoint_callback = ModelCheckpoint(
        filename="epoch={epoch:03d}-step={step:09d}-val_tar_acc={val/target_accuracy:.4f}",
        auto_insert_metric_name=False,
        monitor=training_args.monitor_metric,
        mode=training_args.mode,
        save_top_k=training_args.save_top_k,
        save_last=True,
        verbose=True,
    )

    return early_stop_callback, checkpoint_callback


def create_trainer(
    training_args: TrainerConfig,
    callbacks: list,
) -> pl.Trainer:
    """Create PyTorch Lightning trainer.

    Args:
        training_args: Trainer configuration
        callbacks: List of callbacks

    Returns:
        PyTorch Lightning Trainer instance
    """
    # Configure DDP strategy with find_unused_parameters=True
    # This is needed because in DSN training, source step doesn't use target encoder
    # and target step doesn't use source encoder
    strategy = "auto"
    if training_args.devices > 1:
        strategy = DDPStrategy(find_unused_parameters=True)

    trainer = pl.Trainer(
        max_epochs=training_args.max_epochs,
        devices=training_args.devices,
        precision=training_args.precision,
        log_every_n_steps=training_args.log_every_n_steps,
        val_check_interval=training_args.val_check_interval,
        check_val_every_n_epoch=training_args.check_val_every_n_epoch,
        callbacks=callbacks,
        default_root_dir=training_args.default_root_dir,
        strategy=strategy,
    )
    return trainer


def create_test_trainer(training_args: TrainerConfig) -> pl.Trainer:
    """Create PyTorch Lightning trainer for testing.

    Args:
        training_args: Trainer configuration

    Returns:
        PyTorch Lightning Trainer instance for testing
    """
    test_trainer = pl.Trainer(
        devices=training_args.devices,
        precision=training_args.precision,
        logger=False,
    )
    return test_trainer


def extract_test_results(all_results: list) -> tuple[dict, dict]:
    """Extract train and test results from test output.

    Args:
        all_results: List of test results dictionaries

    Returns:
        Tuple of (train_results, test_results)
    """
    train_results = {k.split("/")[0]: v for k, v in all_results[0].items()} if len(all_results) > 0 else {}
    test_results = {k.split("/")[0]: v for k, v in all_results[1].items()} if len(all_results) > 1 else {}
    return train_results, test_results


def log_training_completion(
    checkpoint_callback: ModelCheckpoint,
    training_args: TrainerConfig,
    model_name: str,
) -> Path | None:
    """Log training completion and return best checkpoint path.

    Args:
        checkpoint_callback: Model checkpoint callback
        training_args: Trainer configuration
        model_name: Name of the model being trained

    Returns:
        Best checkpoint path if available, None otherwise
    """
    best_model_path = checkpoint_callback.best_model_path
    if best_model_path:
        logger.info(f"Training completed. Best checkpoint: {best_model_path}")
        if checkpoint_callback.best_model_score is not None:
            logger.info(f"Best {training_args.monitor_metric}: {checkpoint_callback.best_model_score.item():.4f}")
    else:
        logger.warning("No checkpoint saved during training.")
        best_model_path = None

    return best_model_path


# ============================================================================
# Argument Parsing
# ============================================================================


def parse_args(
    model_type: Literal["name", "patch", "mlp", "expr", "dual"],
) -> tuple[
    TomicDataConfig,
    _ConfigT,
    TrainerConfig,
]:
    # Parse arguments
    config_class = MODEL_CONFIG_MAP[model_type]

    parser = HfArgumentParser((TomicDataConfig, config_class, TrainerConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Parse from JSON file
        data_args, model_args, training_args = parser.parse_json_file(json_file=Path(sys.argv[1]).absolute())
    else:
        # Parse from command line
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()

    return data_args, model_args, training_args


# ============================================================================
# Data Module Creation
# ============================================================================


def create_data_module(
    data_args: TomicDataConfig,
    training_args: TrainerConfig,
) -> DomainDataModuleTomic:
    """Create data module based on model type.

    Args:
        data_args: Data configuration
        training_args: Trainer configuration
        model_args: Model configuration
        model_type: Model type string

    Returns:
        Data module instance
    """

    data_module = DomainDataModuleTomic(
        data_config=data_args,
        train_batch_size=training_args.train_batch_size,
        test_batch_size=training_args.test_batch_size,
        num_workers=training_args.num_workers,
    )

    return data_module


# ============================================================================
# Model Creation
# ============================================================================


def create_model(
    data_args: TomicDataConfig,
    model_args: _ConfigT,
    training_args: TrainerConfig,
    model_type: Literal["name", "patch", "mlp", "expr", "dual"] = "name",
) -> _ModelT:
    """Create model instance based on model type.

    Args:
        data_args: Data configuration
        model_args: Model architecture configuration
        training_args: Trainer configuration
        model_type: Model type string

    Returns:
        Lightning module instance
    """

    # Model-specific parameters
    if model_type == "name":
        model = NameTransformerModel4DSN(
            seq_len=data_args.seq_len,
            hidden_size=model_args.hidden_size,
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            alpha=training_args.alpha,
            beta=training_args.beta,
            gamma=training_args.gamma,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    elif model_type == "patch":
        model = PatchTransformerModel4DSN(
            seq_len=data_args.seq_len,
            hidden_size=model_args.hidden_size,
            patch_size=model_args.patch_size,
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            alpha=training_args.alpha,
            beta=training_args.beta,
            gamma=training_args.gamma,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    elif model_type == "mlp":
        model = MLPModel4DSN(
            seq_len=data_args.seq_len,
            hidden_dims=model_args.hidden_dims,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            alpha=training_args.alpha,
            beta=training_args.beta,
            gamma=training_args.gamma,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    elif model_type == "expr":
        model = ExprTransformerModel4DSN(
            seq_len=data_args.seq_len,
            hidden_size=model_args.hidden_size,
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            alpha=training_args.alpha,
            beta=training_args.beta,
            gamma=training_args.gamma,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    elif model_type == "dual":
        model = DualTransformerModel4DSN(
            seq_len=data_args.seq_len,
            binning=data_args.binning,
            hidden_size=model_args.hidden_size,
            num_heads_cross_attn=model_args.num_heads_cross_attn,
            num_heads_encoder=model_args.num_heads_encoder,
            num_layers_cross_attn=model_args.num_layers_cross_attn,
            num_layers_encoder=model_args.num_layers_encoder,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            alpha=training_args.alpha,
            beta=training_args.beta,
            gamma=training_args.gamma,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model


# ============================================================================
# Training and Testing Functions
# ============================================================================


def train(
    data_args: TomicDataConfig,
    model_args: _ConfigT,
    training_args: TrainerConfig,
    model_type: Literal["patch", "mlp", "name", "expr", "dual"],
) -> Path | None:
    """Train the model.

    Args:
        data_args: Data configuration
        model_args: Model architecture configuration
        training_args: Trainer configuration
        model_type: Model type string

    Returns:
        Best checkpoint path if training succeeds, None otherwise
    """
    pl.seed_everything(training_args.seed)

    # Create data module
    data_module = create_data_module(data_args, training_args)
    # Create model
    model = create_model(data_args, model_args, training_args, model_type)
    # Create callbacks
    early_stop_callback, checkpoint_callback = create_callbacks(training_args)
    # Create trainer
    trainer = create_trainer(training_args, [early_stop_callback, checkpoint_callback])

    # Train the model
    logger.info("=" * 80)
    logger.info(f"Starting Training: {MODEL_DESCRIPTIONS[model_type]}")
    logger.info("=" * 80)
    trainer.fit(model, datamodule=data_module)

    # Log training completion
    return log_training_completion(checkpoint_callback, training_args, MODEL_DESCRIPTIONS[model_type])


def test(
    data_args: TomicDataConfig,
    model_args: _ConfigT,
    training_args: TrainerConfig,
    model_type: Literal["name", "patch", "mlp", "expr", "dual"] = "name",
) -> dict:
    """Test the model.

    Args:
        data_args: Data configuration
        model_args: Model architecture configuration
        training_args: Trainer configuration
        model_type: Model type string

    Returns:
        Test results dictionary
    """
    pl.seed_everything(training_args.seed)

    # Create data module
    data_module = create_data_module(data_args, training_args)

    # Setup data module
    data_module.setup()

    # Find checkpoint
    best_model_path = find_checkpoint(Path(training_args.default_root_dir), training_args.checkpoint_path)
    if best_model_path is None:
        raise FileNotFoundError("No checkpoint found. Please provide checkpoint_path or train the model first.")

    # Load model from checkpoint
    if model_type == "dual":
        LightningModuleClass = DualTransformerModel4DSN
    elif model_type == "name":
        LightningModuleClass = NameTransformerModel4DSN
    elif model_type == "patch":
        LightningModuleClass = PatchTransformerModel4DSN
    elif model_type == "mlp":
        LightningModuleClass = MLPModel4DSN
    elif model_type == "expr":
        LightningModuleClass = ExprTransformerModel4DSN
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    logger.info(f"Loading model from checkpoint: {best_model_path}")
    best_model = LightningModuleClass.load_from_checkpoint(best_model_path)

    # Create trainer for testing
    test_trainer = create_test_trainer(training_args)

    # Evaluate on both train and test sets
    logger.info("=" * 80)
    logger.info("Evaluating on TRAIN and TEST sets:")
    logger.info("=" * 80)
    all_results = test_trainer.test(best_model, datamodule=data_module, ckpt_path=best_model_path)

    # Extract results for train and test sets
    train_results, test_results = extract_test_results(all_results)

    results = {
        "model_type": model_type,
        "best_checkpoint": str(best_model_path) if best_model_path else None,
        "train_results": train_results,
        "test_results": test_results,
        "hyperparameters": {
            **training_args.__dict__,
            **model_args.__dict__,
            **data_args.__dict__,
        },
    }

    # Save results to checkpoint parent directory (lightning_logs/version_X/)
    results_save_path = best_model_path.parent / "results.json"
    results_save_path.write_text(json.dumps(results, indent=2, default=str))

    # Log summary
    logger.info("=" * 80)
    logger.info("Summary:")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {best_model_path}")
    logger.info(f"Results saved to: {results_save_path}")

    return results


# ============================================================================
# Main Entry Point
# ============================================================================


def main(
    data_args: TomicDataConfig = None,
    model_args: _ConfigT = None,
    training_args: TrainerConfig = None,
    model_type: Literal["name", "patch", "mlp", "expr", "dual"] = "name",
):
    """Main function."""
    # Parse arguments from command line or JSON config
    if data_args is None or model_args is None or training_args is None or model_type is None:
        data_args, model_args, training_args = parse_args(model_type)

    if training_args.run_training:
        best_model_path = train(data_args, model_args, training_args, model_type)
        if best_model_path:
            training_args.checkpoint_path = str(best_model_path)

    if training_args.run_testing:
        test(data_args, model_args, training_args, model_type)


if __name__ == "__main__":
    main()
