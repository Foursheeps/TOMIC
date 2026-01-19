#!/usr/bin/env python3
"""
Load and test data using BaseDataModule from common.py.

This script loads preprocessed data and tests the DomainDataModulecommon
to ensure it works correctly with the scGPT format.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append("/your/path/to/TOMIC")

import torch

from datmp import get_logger
from datmp.dataset.dataconfig import DatmpDataConfig
from datmp.dataset.dataset4common import DomainDataModuleCommon

# Use unified logger
logger = get_logger("data_process")


def test_data_module(
    root_data_path: Path | str,
    binning: int | None = None,
    train: str = "source",
    train_batch_size: int = 8,
    test_batch_size: int = 8,
    num_workers: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Test DomainDataModuleCommon with preprocessed data.

    Args:
        root_data_path: Path to directory containing preprocessed data
        binning: Number of bins for expression value discretization (None for continuous).
                 If None, continuous expression values will be used.
        train: Which domain to use for training ("source", "target", or "both")
        train_batch_size: Batch size for training
        test_batch_size: Batch size for testing
        num_workers: Number of data loading workers
        test_size: Fraction of data to use for testing
        random_state: Random seed for reproducibility
    """
    root_data_path = Path(root_data_path)
    logger.info(f"Loading data from: {root_data_path}")

    # Load or create DatmpDataConfig
    config_path = root_data_path / "info_config.json"
    if config_path.exists():
        logger.info(f"Loading config from: {config_path}")
        data_config = DatmpDataConfig.from_json_or_kwargs(config_path, binning=binning)
    else:
        logger.warning(f"Config file not found at {config_path}, creating DatmpDataConfig manually")
        # Create minimal DatmpDataConfig - this may fail if required fields are missing
        data_config = DatmpDataConfig(
            root_data_path=root_data_path,
            binning=binning,
        )
        logger.warning("Warning: class_map and other fields may be missing. Data loading may fail.")

    # Create data module
    data_module = DomainDataModuleCommon(
        data_config=data_config,
        train=train,
        train_batch_size=train_batch_size,
        test_batch_size=test_batch_size,
        num_workers=num_workers,
        test_size=test_size,
        random_state=random_state,
    )

    # Setup data module (loads data and creates datasets)
    logger.info("Setting up data module...")
    data_module.setup()

    # Log dataset information
    logger.info("\n" + "=" * 80)
    logger.info("Dataset Information:")
    logger.info("=" * 80)

    logger.info(f"Train dataset size: {len(data_module.train_dataset)}")
    logger.info(f"Validation dataset size: {len(data_module.val_dataset)}")
    logger.info(f"Test dataset size: {len(data_module.test_dataset)}")

    # Log data shapes
    logger.info("\nData shapes:")
    if data_module.train_dataset.is_binned:
        train_expr_shape = data_module.train_dataset.binned.shape
        val_expr_shape = data_module.val_dataset.binned.shape
        test_expr_shape = data_module.test_dataset.binned.shape
    else:
        train_expr_shape = data_module.train_dataset.expr.shape
        val_expr_shape = data_module.val_dataset.expr.shape
        test_expr_shape = data_module.test_dataset.expr.shape
    logger.info(f"  Train: {train_expr_shape[0]} cells, {train_expr_shape[1]} genes")
    logger.info(f"  Val: {val_expr_shape[0]} cells, {val_expr_shape[1]} genes")
    logger.info(f"  Test: {test_expr_shape[0]} cells, {test_expr_shape[1]} genes")

    # Log binned data information
    logger.info("\nBinned data information:")
    if data_module.train_dataset.is_binned:
        train_binned_shape = data_module.train_dataset.binned.shape
        logger.info(f"  Train binned: {train_binned_shape}")
        logger.info(f"  Val binned: {data_module.val_dataset.binned.shape}")
        logger.info(f"  Test binned: {data_module.test_dataset.binned.shape}")
    else:
        logger.info("  Binning not performed (using continuous expression values)")

    # Log gene information
    logger.info("\nGene information:")
    logger.info(f"  Number of genes (from expression shape): {train_expr_shape[1]}")
    logger.info(f"  Class map: {data_module.class_map}")
    logger.info(f"  Number of classes: {data_module.num_classes}")

    # Test data loading
    logger.info("\n" + "=" * 80)
    logger.info("Testing Data Loading:")
    logger.info("=" * 80)

    # Get train dataloader
    train_loader = data_module.train_dataloader()
    logger.info(f"Train dataloader created: {len(train_loader)} batches")

    # Get a sample batch
    sample_batch = next(iter(train_loader))
    logger.info("\nSample batch keys:")
    for key in sample_batch.keys():
        value = sample_batch[key]
        if isinstance(value, torch.Tensor):
            logger.info(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            logger.info(f"  {key}: type={type(value)}, value={value}")

    # Test validation dataloader
    val_loader = data_module.val_dataloader()
    logger.info(f"\nValidation dataloader created: {len(val_loader)} batches")

    # Test test dataloader
    test_loaders = data_module.test_dataloader()
    logger.info(f"Test dataloaders created:")
    logger.info(f"  source_test: {len(test_loaders['source_test'])} batches")
    logger.info(f"  target_test: {len(test_loaders['target_test'])} batches")

    # Test a few batches and collect statistics
    logger.info("\n" + "=" * 80)
    logger.info("Testing Batch Iteration:")
    logger.info("=" * 80)

    batch_count = 0
    max_batches = 3

    # Collect statistics across all batches
    all_expr_min = []
    all_expr_max = []
    all_expr_ids_min = []
    all_expr_ids_max = []

    for batch in train_loader:
        batch_count += 1
        logger.info(f"\nBatch {batch_count}:")
        logger.info(f"  Gene IDs shape: {batch['gene_ids'].shape}")

        # Handle both binned and continuous expression values
        if "expr" in batch:
            logger.info(f"  Expression (expr) shape: {batch['expr'].shape}")
            logger.info(f"  Expression dtype: {batch['expr'].dtype}")
        if "expr_ids" in batch:
            logger.info(f"  Binned (expr_ids) shape: {batch['expr_ids'].shape}")
            logger.info(f"  Binned dtype: {batch['expr_ids'].dtype}")

        logger.info(f"  Labels shape: {batch['label'].shape}")
        logger.info(f"  Labels: {batch['label'].tolist()}")

        # Calculate min/max for this batch
        if "expr" in batch:
            expr_min = float(batch["expr"].min())
            expr_max = float(batch["expr"].max())
            all_expr_min.append(expr_min)
            all_expr_max.append(expr_max)
        else:
            expr_min = None
            expr_max = None

        if "expr_ids" in batch:
            expr_ids_min = int(batch["expr_ids"].min())
            expr_ids_max = int(batch["expr_ids"].max())
            all_expr_ids_min.append(expr_ids_min)
            all_expr_ids_max.append(expr_ids_max)
        else:
            expr_ids_min = None
            expr_ids_max = None

        # Verify data ranges for this batch
        logger.info(f"\n  Batch {batch_count} - Data Ranges:")
        if expr_min is not None:
            logger.info(f"    Expression: min={expr_min:.6f}, max={expr_max:.6f}")
        if expr_ids_min is not None:
            logger.info(f"    Binned: min={expr_ids_min}, max={expr_ids_max}")

        import matplotlib.pyplot as plt

        if "expr" in batch:
            plt.imshow(batch["expr"])
            plt.savefig(f"debug_common_expr_{batch_count}.png")
            plt.close()
        elif "expr_ids" in batch:
            plt.imshow(batch["expr_ids"])
            plt.savefig(f"debug_common_expr_ids_{batch_count}.png")
            plt.close()

        if batch_count >= max_batches:
            break

    # Print overall statistics across all batches
    logger.info("\n" + "=" * 80)
    logger.info("Overall Statistics Across All Batches:")
    logger.info("=" * 80)

    if all_expr_min:
        logger.info("\nExpression (expr):")
        logger.info(f"  Overall min: {min(all_expr_min):.6f}")
        logger.info(f"  Overall max: {max(all_expr_max):.6f}")
        logger.info(f"  Batch-wise min range: [{min(all_expr_min):.6f}, {max(all_expr_min):.6f}]")
        logger.info(f"  Batch-wise max range: [{min(all_expr_max):.6f}, {max(all_expr_max):.6f}]")

    if all_expr_ids_min:
        logger.info("\nBinned (expr_ids):")
        logger.info(f"  Overall min: {min(all_expr_ids_min)}")
        logger.info(f"  Overall max: {max(all_expr_ids_max)}")
        logger.info(f"  Batch-wise min range: [{min(all_expr_ids_min)}, {max(all_expr_ids_min)}]")
        logger.info(f"  Batch-wise max range: [{min(all_expr_ids_max)}, {max(all_expr_ids_max)}]")

    logger.info("\n" + "=" * 80)
    logger.info("Data module test completed successfully!")
    logger.info("=" * 80)


def main():
    """Main function to test data loading."""
    # Path to preprocessed data
    # Update this path to match your actual data directory
    data_path = Path("/your/path/to/TOMIC/expertments/data_process/GSE173958_processed/GSE173958_M1_1200")

    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("Please update the data_path in the script to point to your preprocessed data directory")
        return

    # Test with different training modes
    logger.info("\n" + "=" * 80)
    logger.info("Test: Standard Supervised Learning (Source Domain)")
    logger.info("=" * 80)
    logger.info("Note: DomainDataModuleCommon supports both binned and continuous expression values")
    test_data_module(
        root_data_path=data_path,
        binning=51,
        train="source",
        train_batch_size=256,
        test_batch_size=256,
        num_workers=0,
    )

    logger.info("\n" + "=" * 80)
    logger.info("Test: Standard Supervised Learning (Both Domains)")
    logger.info("=" * 80)
    test_data_module(
        root_data_path=data_path,
        binning=51,
        train="both",
        train_batch_size=256,
        test_batch_size=256,
        num_workers=0,
    )


if __name__ == "__main__":
    main()
