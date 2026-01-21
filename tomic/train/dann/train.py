"""
Unified training script for DANN models.

This script trains Domain Adversarial Neural Networks supporting multiple model types:
- name: Name-based Transformer
- patch: Patch-based Transformer
- mlp: MLP encoder
- expr: Expression-based Transformer
- dual: Dual Transformer

Usage:
    python -m tomic.train.dann.train --model_type name --lr 1e-3
    python -m tomic.train.dann.train config.json
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
from ...model.dann import (
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
logger = get_logger("train_dann")


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
    "name": "Name-based Transformer DANN",
    "patch": "Patch-based Transformer DANN",
    "mlp": "MLP-based DANN",
    "expr": "Expression-based Transformer DANN",
    "dual": "Dual Transformer DANN",
}

# Type alias for model config classes
_ConfigT = NameModelConfig | PatchModelConfig | MLPModelConfig | ExprModelConfig | DualTransformerModelConfig
_ModelT = LightningModuleName | LightningModulePatch | LightningModuleMLP | LightningModuleExpr | LightningModuleDual


# ============================================================================
# Utility Functions
# ============================================================================


def find_checkpoint(save_dir: Path, checkpoint_path: str | None = None) -> Path | None:
    """Find checkpoint file."""

    def extract_val_tar_acc_from_filename(filename: str) -> float:
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

    checkpoint_dir = save_dir  # / "checkpoints"
    if checkpoint_dir.exists():
        best_checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
        if best_checkpoints:
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
    """Create training callbacks."""
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


def create_trainer(training_args: TrainerConfig, callbacks: list) -> pl.Trainer:
    """Create PyTorch Lightning trainer."""
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
    """Create PyTorch Lightning trainer for testing."""
    return pl.Trainer(devices=training_args.devices, precision=training_args.precision, logger=False)


def extract_test_results(all_results: list) -> tuple[dict, dict]:
    """Extract train and test results from test output."""
    train_results = {k.split("/")[0]: v for k, v in all_results[0].items()} if len(all_results) > 0 else {}
    test_results = {k.split("/")[0]: v for k, v in all_results[1].items()} if len(all_results) > 1 else {}
    return train_results, test_results


def log_training_completion(
    checkpoint_callback: ModelCheckpoint,
    training_args: TrainerConfig,
    model_name: str,
) -> Path | None:
    """Log training completion and return best checkpoint path."""
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
) -> tuple[TomicDataConfig, _ConfigT, TrainerConfig]:
    config_class = MODEL_CONFIG_MAP[model_type]
    parser = HfArgumentParser((TomicDataConfig, config_class, TrainerConfig))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        data_args, model_args, training_args = parser.parse_json_file(json_file=Path(sys.argv[1]).absolute())
    else:
        data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    return data_args, model_args, training_args


# ============================================================================
# Data Module Creation
# ============================================================================


def create_data_module(data_args: TomicDataConfig, training_args: TrainerConfig) -> DomainDataModuleTomic:
    """Create data module."""
    return DomainDataModuleTomic(
        data_config=data_args,
        train_batch_size=training_args.train_batch_size,
        test_batch_size=training_args.test_batch_size,
        num_workers=training_args.num_workers,
    )


# ============================================================================
# Model Creation
# ============================================================================


def create_model(
    data_args: TomicDataConfig,
    model_args: _ConfigT,
    training_args: TrainerConfig,
    model_type: Literal["name", "patch", "mlp", "expr", "dual"] = "name",
) -> _ModelT:
    """Create model instance based on model type."""
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
            gamma=training_args.gamma,
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
            dropout=model_args.dropout,
            activation=model_args.activation,
            num_heads=model_args.num_heads,
            num_layers=model_args.num_layers,
            num_classes=data_args.num_classes,
            lr=training_args.lr,
            gamma=training_args.gamma,
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
            gamma=training_args.gamma,
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
            gamma=training_args.gamma,
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
    """Train the model."""
    pl.seed_everything(training_args.seed)
    data_module = create_data_module(data_args, training_args)
    model = create_model(data_args, model_args, training_args, model_type)
    early_stop_callback, checkpoint_callback = create_callbacks(training_args)
    trainer = create_trainer(training_args, [early_stop_callback, checkpoint_callback])
    logger.info("=" * 80)
    logger.info(f"Starting Training: {MODEL_DESCRIPTIONS[model_type]}")
    logger.info("=" * 80)
    trainer.fit(model, datamodule=data_module)
    return log_training_completion(checkpoint_callback, training_args, MODEL_DESCRIPTIONS[model_type])


def test(
    data_args: TomicDataConfig,
    model_args: _ConfigT,
    training_args: TrainerConfig,
    model_type: Literal["name", "patch", "mlp", "expr", "dual"] = "name",
) -> dict:
    """Test the model."""
    pl.seed_everything(training_args.seed)
    data_module = create_data_module(data_args, training_args)
    data_module.setup()
    best_model_path = find_checkpoint(Path(training_args.default_root_dir), training_args.checkpoint_path)
    if best_model_path is None:
        raise FileNotFoundError("No checkpoint found. Please provide checkpoint_path or train the model first.")
    LightningModuleClass = MODEL_TYPE_MAP[model_type]
    logger.info(f"Loading model from checkpoint: {best_model_path}")
    best_model = LightningModuleClass.load_from_checkpoint(best_model_path)
    test_trainer = create_test_trainer(training_args)
    logger.info("=" * 80)
    logger.info("Evaluating on TRAIN and TEST sets:")
    logger.info("=" * 80)
    all_results = test_trainer.test(best_model, datamodule=data_module, ckpt_path=best_model_path)
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
    results_save_path = best_model_path.parent / "results.json"
    results_save_path.write_text(json.dumps(results, indent=2, default=str))
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
