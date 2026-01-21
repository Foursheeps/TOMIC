"""
Unified training script for standard supervised learning models.

This script trains standard supervised learning models supporting multiple model types:
- name: Name-based Transformer
- patch: Patch-based Transformer
- mlp: MLP encoder
- expr: Expression-based Transformer
- dual: Dual Transformer

Usage:
    python -m tomic.train.usual.train --model_type name --lr 1e-3
    python -m tomic.train.usual.train config.json
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
from ...dataset.dataset4common import DomainDataModuleCommon
from ...logger import get_logger
from ...model.usual import (
    DualTransformerModelConfig,
    ExprModelConfig,
    LightningModuleDual,
    LightningModuleExpr,
    LightningModuleMLP,
    LightningModuleName,
    LightningModulePatch,
    MLPModelConfig,
    NameModelConfig,
    PatchModelConfig,
)
from .train_config import TrainerConfig

# Get logger
logger = get_logger("train_usual")


# Model type to config class mapping
MODEL_CONFIG_MAP = {
    "name": NameModelConfig,
    "patch": PatchModelConfig,
    "mlp": MLPModelConfig,
    "expr": ExprModelConfig,
    "dual": DualTransformerModelConfig,
}

# Model type to Lightning module class mapping
MODEL_TYPE_MAP = {
    "name": LightningModuleName,
    "patch": LightningModulePatch,
    "mlp": LightningModuleMLP,
    "expr": LightningModuleExpr,
    "dual": LightningModuleDual,
}

# Model type descriptions
MODEL_DESCRIPTIONS = {
    "name": "Name-based Transformer",
    "patch": "Patch-based Transformer",
    "mlp": "MLP-based",
    "expr": "Expression-based Transformer",
    "dual": "Dual Transformer",
}


# Type alias for model config classes
_ConfigT = NameModelConfig | PatchModelConfig | MLPModelConfig | ExprModelConfig | DualTransformerModelConfig
_ModelT = LightningModuleName | LightningModulePatch | LightningModuleMLP | LightningModuleExpr | LightningModuleDual


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

    def extract_val_acc_from_filename(filename: str) -> float:
        """Extract val_acc value from checkpoint filename."""
        pattern = r"val_acc=([\d.]+)"
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
            # Sort by val_acc value in filename (descending)
            best_checkpoints.sort(key=lambda x: extract_val_acc_from_filename(x.name), reverse=True)
            best_checkpoint = best_checkpoints[0]
            best_acc = extract_val_acc_from_filename(best_checkpoint.name)
            logger.info(
                f"Found {len(best_checkpoints)} checkpoint(s). Best: {best_checkpoint.name} (val_acc={best_acc:.4f})"
            )
            return best_checkpoint
        else:
            logger.warning(f"No checkpoint found in {checkpoint_dir}.")
            return None
    else:
        logger.warning(f"Checkpoint directory {checkpoint_dir} does not exist.")
        return None


def create_data_module_ann(
    data_args: TomicDataConfig,
    training_args: TrainerConfig,
    train_domain: str = "source",
) -> DomainDataModuleCommon:
    """Create AnnData-based data module for standard supervised learning.

    Args:
        data_args: Data configuration
        training_args: Trainer configuration
        train_domain: Which domain to use for training ("source", "target", or "both")

    Returns:
        Data module instance
    """
    data_module = DomainDataModuleCommon(
        data_config=data_args,
        train=train_domain,
        train_batch_size=training_args.train_batch_size,
        test_batch_size=training_args.test_batch_size,
        num_workers=training_args.num_workers,
        test_size=0.2,
        random_state=training_args.seed,
    )

    return data_module


def create_data_module_token(
    data_args: TomicDataConfig,
    training_args: TrainerConfig,
    train_domain: str = "source",
) -> DomainDataModuleCommon:
    """Create Token-based data module for standard supervised learning.
    Args:

        data_args: Data configuration
        training_args: Trainer configuration
        train_domain: Which domain to use for training ("source", "target", or "both")

    Returns:
        Data module instance
    """
    data_module = DomainDataModuleCommon(
        data_config=data_args,
        train=train_domain,
        train_batch_size=training_args.train_batch_size,
        test_batch_size=training_args.test_batch_size,
        num_workers=training_args.num_workers,
        test_size=0.2,
        random_state=training_args.seed,
    )

    return data_module


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
        filename="epoch={epoch:03d}-step={step:09d}-val_acc={val/accuracy:.4f}",
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
    """Extract source and target test results from test output.

    Args:
        all_results: List of test results dictionaries

    Returns:
        Tuple of (source_test_results, target_test_results)
    """
    source_results = (
        {k.replace("test_source/", ""): v for k, v in all_results[0].items()} if len(all_results) > 0 else {}
    )
    target_results = (
        {k.replace("test_target/", ""): v for k, v in all_results[1].items()} if len(all_results) > 1 else {}
    )
    return source_results, target_results


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
    model_type: Literal["name", "patch", "mlp", "expr", "dual"],
    train_domain: str = "source",
):
    """Create data module based on model type.

    Args:
        data_args: Data configuration
        training_args: Trainer configuration
        model_type: Model type string
        train_domain: Which domain to use for training ("source", "target", or "both")

    Returns:
        Data module instance
    """
    # Token-based models (name, dual) use token data module
    # Others use ann data module
    if model_type in ["name", "dual"]:
        data_module = create_data_module_token(data_args, training_args, train_domain=train_domain)
    else:
        data_module = create_data_module_ann(data_args, training_args, train_domain=train_domain)

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
        model = LightningModuleName(
            seq_len=data_args.seq_len,
            hidden_size=model_args.hidden_size,
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    elif model_type == "patch":
        model = LightningModulePatch(
            seq_len=data_args.seq_len,
            hidden_size=model_args.hidden_size,
            patch_size=model_args.patch_size,
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    elif model_type == "mlp":
        model = LightningModuleMLP(
            seq_len=data_args.seq_len,
            hidden_dims=model_args.hidden_dims,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    elif model_type == "expr":
        model = LightningModuleExpr(
            seq_len=data_args.seq_len,
            hidden_size=model_args.hidden_size,
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            scheduler_type=training_args.scheduler_type,
            warmup_ratio=training_args.warmup_ratio,
            num_epochs=training_args.max_epochs,
            train_batch_size=training_args.train_batch_size,
        )
    elif model_type == "dual":
        model = LightningModuleDual(
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
    train_domain: str = "source",
) -> Path | None:
    """Train the model.

    Args:
        data_args: Data configuration
        model_args: Model architecture configuration
        training_args: Trainer configuration
        model_type: Model type string
        train_domain: Which domain to use for training ("source", "target", or "both")

    Returns:
        Best checkpoint path if training succeeds, None otherwise
    """
    pl.seed_everything(training_args.seed)

    # Create data module
    data_module = create_data_module(data_args, training_args, model_type, train_domain=train_domain)
    data_module.setup()

    # Update data_args with dimensions from data_module
    if hasattr(data_module, "train_dataset") and data_module.train_dataset is not None:
        if model_type in ["name", "dual"]:
            # For token datasets, vocab_size should be set from tokenizer
            if hasattr(data_module.train_dataset, "tokenizer"):
                data_args.seq_len = len(data_module.train_dataset.tokenizer.vocab)
            data_args.num_classes = len(set(data_module.train_dataset.labels))
        else:
            # For ann datasets (patch/mlp/expr), update seq_len from expr
            if hasattr(data_module.train_dataset, "expr"):
                # Datasetcommon provides expr attribute
                data_args.seq_len = data_module.train_dataset.expr.shape[1]
            elif hasattr(data_module.train_dataset, "data"):
                # Fallback for other dataset types
                data_args.seq_len = data_module.train_dataset.data.shape[1]
            data_args.num_classes = len(set(data_module.train_dataset.labels))

    # Create model
    model = create_model(data_args, model_args, training_args, model_type)
    # Create callbacks
    early_stop_callback, checkpoint_callback = create_callbacks(training_args)
    # Create trainer
    trainer = create_trainer(training_args, [early_stop_callback, checkpoint_callback])

    # Train the model
    logger.info("=" * 80)
    logger.info(f"Starting Training: {MODEL_DESCRIPTIONS[model_type]} (train_domain={train_domain})")
    logger.info("=" * 80)
    trainer.fit(model, datamodule=data_module)

    # Log training completion
    return log_training_completion(checkpoint_callback, training_args, MODEL_DESCRIPTIONS[model_type])


def test(
    data_args: TomicDataConfig,
    model_args: _ConfigT,
    training_args: TrainerConfig,
    model_type: Literal["name", "patch", "mlp", "expr", "dual"] = "name",
    train_domain: str = "source",
) -> dict:
    """Test the model.

    Args:
        data_args: Data configuration
        model_args: Model architecture configuration
        training_args: Trainer configuration
        model_type: Model type string
        train_domain: Which domain was used for training ("source", "target", or "both")

    Returns:
        Test results dictionary
    """
    pl.seed_everything(training_args.seed)

    # Create data module
    data_module = create_data_module(data_args, training_args, model_type, train_domain=train_domain)
    data_module.setup()

    # Find checkpoint
    best_model_path = find_checkpoint(Path(training_args.default_root_dir), training_args.checkpoint_path)
    if best_model_path is None:
        raise FileNotFoundError("No checkpoint found. Please provide checkpoint_path or train the model first.")

    # Load model from checkpoint
    LightningModuleClass = MODEL_TYPE_MAP[model_type]

    logger.info(f"Loading model from checkpoint: {best_model_path}")
    best_model = LightningModuleClass.load_from_checkpoint(best_model_path)

    # Create trainer for testing
    test_trainer = create_test_trainer(training_args)

    # Evaluate on both source and target test sets
    logger.info("=" * 80)
    logger.info("Evaluating on SOURCE and TARGET test sets:")
    logger.info("=" * 80)
    all_results = test_trainer.test(best_model, datamodule=data_module, ckpt_path=best_model_path)

    # Extract results for source and target test sets
    source_results, target_results = extract_test_results(all_results)

    results = {
        "model_type": model_type,
        "train_domain": train_domain,
        "best_checkpoint": str(best_model_path) if best_model_path else None,
        "source_test_results": source_results,
        "target_test_results": target_results,
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
    logger.info(f"Source test accuracy: {source_results.get('accuracy', 'N/A')}")
    logger.info(f"Target test accuracy: {target_results.get('accuracy', 'N/A')}")

    return results


# ============================================================================
# Main Entry Point
# ============================================================================


def main(
    data_args: TomicDataConfig = None,
    model_args: _ConfigT = None,
    training_args: TrainerConfig = None,
    model_type: Literal["name", "patch", "mlp", "expr", "dual"] = "name",
    train_domain: str = "source",
):
    """Main function."""
    # Parse arguments from command line or JSON config
    if data_args is None or model_args is None or training_args is None or model_type is None:
        data_args, model_args, training_args = parse_args(model_type)
        # Extract train_domain from data_args if available
        if hasattr(data_args, "train"):
            train_domain = data_args.train

    if training_args.run_training:
        best_model_path = train(data_args, model_args, training_args, model_type, train_domain=train_domain)
        if best_model_path:
            training_args.checkpoint_path = str(best_model_path)

    if training_args.run_testing:
        test(data_args, model_args, training_args, model_type, train_domain=train_domain)


if __name__ == "__main__":
    main()
