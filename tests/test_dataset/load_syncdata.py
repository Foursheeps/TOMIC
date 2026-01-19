#!/usr/bin/env python3
"""
Load and test synthetic data using DomainDataModuleDatmp.

This script loads preprocessed synthetic data and tests the data module
to ensure it works correctly.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append("/your/path/to/TOMIC")

import torch

from datmp import get_logger
from datmp.dataset.dataconfig import DatmpDataConfig
from datmp.dataset.dataset4da import DomainDataModuleDatmp

# Use unified logger
logger = get_logger("data_process")


def test_data_module(
    root_data_path: Path | str,
    binning: int | None = None,
    train_batch_size: int = 8,
    test_batch_size: int = 8,
    num_workers: int = 0,
    test_size: float = 0.2,
    random_state: int = 42,
) -> None:
    """
    Test DomainDataModuleDatmp with preprocessed synthetic data.

    Args:
        root_data_path: Path to directory containing preprocessed data
        binning: Number of bins for expression value discretization (None for continuous).
                 If None, continuous expression values will be used.
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
    data_module = DomainDataModuleDatmp(
        data_config=data_config,
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

    # Get dataset sizes from indices
    train_size = min(len(data_module.train_dataset.source_indices), len(data_module.train_dataset.target_indices))
    val_size = min(len(data_module.val_dataset.source_indices), len(data_module.val_dataset.target_indices))
    test_size_actual = min(len(data_module.test_dataset.source_indices), len(data_module.test_dataset.target_indices))

    logger.info(f"Train dataset size: {train_size}")
    logger.info(f"Validation dataset size: {val_size}")
    logger.info(f"Test dataset size: {test_size_actual}")

    # Log source and target information
    logger.info("\nSource (metastasis) samples:")
    if data_module.train_dataset.is_binned:
        train_source_shape = data_module.train_dataset.source_binned.shape
        val_source_shape = data_module.val_dataset.source_binned.shape
    else:
        train_source_shape = data_module.train_dataset.source_expr.shape
        val_source_shape = data_module.val_dataset.source_expr.shape
    logger.info(f"  Train: {train_source_shape[0]} cells, {train_source_shape[1]} genes")
    logger.info(f"  Val/Test: {val_source_shape[0]} cells, {val_source_shape[1]} genes")

    logger.info("\nTarget (primary) samples:")
    if data_module.train_dataset.is_binned:
        train_target_shape = data_module.train_dataset.target_binned.shape
        val_target_shape = data_module.val_dataset.target_binned.shape
    else:
        train_target_shape = data_module.train_dataset.target_expr.shape
        val_target_shape = data_module.val_dataset.target_expr.shape
    logger.info(f"  Train: {train_target_shape[0]} cells, {train_target_shape[1]} genes")
    logger.info(f"  Val/Test: {val_target_shape[0]} cells, {val_target_shape[1]} genes")

    # Log binned data information
    logger.info("\nBinned data information:")
    if data_module.train_dataset.is_binned:
        train_source_binned_shape = data_module.train_dataset.source_binned.shape
        logger.info(f"  Train source binned: {train_source_binned_shape}")
        logger.info(f"  Train target binned: {data_module.train_dataset.target_binned.shape}")
    else:
        logger.info("  Binning not performed (using continuous expression values)")

    # Log gene information (number of genes from expression data shape)
    logger.info("\nGene information:")
    logger.info(f"  Number of genes (from expression shape): {train_source_shape[1]}")
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
    test_loader = data_module.test_dataloader()
    logger.info(f"Test dataloader created: {len(test_loader)} batches")

    # Test a few batches and collect statistics
    logger.info("\n" + "=" * 80)
    logger.info("Testing Batch Iteration:")
    logger.info("=" * 80)

    batch_count = 0
    max_batches = 3

    # Collect statistics across all batches
    all_s_expr_min = []
    all_s_expr_max = []
    all_t_expr_min = []
    all_t_expr_max = []
    all_s_expr_ids_min = []
    all_s_expr_ids_max = []
    all_t_expr_ids_min = []
    all_t_expr_ids_max = []

    for batch in train_loader:
        batch_count += 1
        logger.info(f"\nBatch {batch_count}:")

        # Handle both binned and continuous expression values
        if "s_expr" in batch:
            logger.info(f"  Source expression (s_expr) shape: {batch['s_expr'].shape}")
            logger.info(f"  Source expression dtype: {batch['s_expr'].dtype}")
        if "t_expr" in batch:
            logger.info(f"  Target expression (t_expr) shape: {batch['t_expr'].shape}")
            logger.info(f"  Target expression dtype: {batch['t_expr'].dtype}")
        if "s_expr_ids" in batch:
            logger.info(f"  Source binned (s_expr_ids) shape: {batch['s_expr_ids'].shape}")
            logger.info(f"  Source binned dtype: {batch['s_expr_ids'].dtype}")
        if "t_expr_ids" in batch:
            logger.info(f"  Target binned (t_expr_ids) shape: {batch['t_expr_ids'].shape}")
            logger.info(f"  Target binned dtype: {batch['t_expr_ids'].dtype}")

        logger.info(f"  Source gene IDs shape: {batch['s_gene_ids'].shape}")
        logger.info(f"  Target gene IDs shape: {batch['t_gene_ids'].shape}")
        logger.info(f"  Source labels: {batch['s_label'].tolist()}")
        logger.info(f"  Target labels: {batch['t_label'].tolist()}")

        # Calculate min/max for this batch
        if "s_expr" in batch:
            s_expr_min = float(batch["s_expr"].min())
            s_expr_max = float(batch["s_expr"].max())
            all_s_expr_min.append(s_expr_min)
            all_s_expr_max.append(s_expr_max)
        else:
            s_expr_min = None
            s_expr_max = None

        if "t_expr" in batch:
            t_expr_min = float(batch["t_expr"].min())
            t_expr_max = float(batch["t_expr"].max())
            all_t_expr_min.append(t_expr_min)
            all_t_expr_max.append(t_expr_max)
        else:
            t_expr_min = None
            t_expr_max = None

        if "s_expr_ids" in batch:
            s_expr_ids_min = int(batch["s_expr_ids"].min())
            s_expr_ids_max = int(batch["s_expr_ids"].max())
            all_s_expr_ids_min.append(s_expr_ids_min)
            all_s_expr_ids_max.append(s_expr_ids_max)
        else:
            s_expr_ids_min = None
            s_expr_ids_max = None

        if "t_expr_ids" in batch:
            t_expr_ids_min = int(batch["t_expr_ids"].min())
            t_expr_ids_max = int(batch["t_expr_ids"].max())
            all_t_expr_ids_min.append(t_expr_ids_min)
            all_t_expr_ids_max.append(t_expr_ids_max)
        else:
            t_expr_ids_min = None
            t_expr_ids_max = None

        # Verify data ranges for this batch
        logger.info(f"\n  Batch {batch_count} - Data Ranges:")
        if s_expr_min is not None:
            logger.info(f"    Source expression: min={s_expr_min:.6f}, max={s_expr_max:.6f}")
        if t_expr_min is not None:
            logger.info(f"    Target expression: min={t_expr_min:.6f}, max={t_expr_max:.6f}")
        if s_expr_ids_min is not None:
            logger.info(f"    Source binned: min={s_expr_ids_min}, max={s_expr_ids_max}")
        if t_expr_ids_min is not None:
            logger.info(f"    Target binned: min={t_expr_ids_min}, max={t_expr_ids_max}")

        import matplotlib.pyplot as plt

        if "s_expr" in batch:
            plt.imshow(batch["s_expr"])
            plt.savefig(f"debug_s_expr_{batch_count}.png")
            plt.close()
        elif "s_expr_ids" in batch:
            plt.imshow(batch["s_expr_ids"])
            plt.savefig(f"debug_s_expr_ids_{batch_count}.png")
            plt.close()

        if "t_expr" in batch:
            plt.imshow(batch["t_expr"])
            plt.savefig(f"debug_t_expr_{batch_count}.png")
            plt.close()
        elif "t_expr_ids" in batch:
            plt.imshow(batch["t_expr_ids"])
            plt.savefig(f"debug_t_expr_ids_{batch_count}.png")
            plt.close()

        if batch_count >= max_batches:
            break

    # Print overall statistics across all batches
    logger.info("\n" + "=" * 80)
    logger.info("Overall Statistics Across All Batches:")
    logger.info("=" * 80)

    if all_s_expr_min:
        logger.info("\nSource Expression (s_expr):")
        logger.info(f"  Overall min: {min(all_s_expr_min):.6f}")
        logger.info(f"  Overall max: {max(all_s_expr_max):.6f}")
        logger.info(f"  Batch-wise min range: [{min(all_s_expr_min):.6f}, {max(all_s_expr_min):.6f}]")
        logger.info(f"  Batch-wise max range: [{min(all_s_expr_max):.6f}, {max(all_s_expr_max):.6f}]")

    if all_t_expr_min:
        logger.info("\nTarget Expression (t_expr):")
        logger.info(f"  Overall min: {min(all_t_expr_min):.6f}")
        logger.info(f"  Overall max: {max(all_t_expr_max):.6f}")
        logger.info(f"  Batch-wise min range: [{min(all_t_expr_min):.6f}, {max(all_t_expr_min):.6f}]")
        logger.info(f"  Batch-wise max range: [{min(all_t_expr_max):.6f}, {max(all_t_expr_max):.6f}]")

    if all_s_expr_ids_min:
        logger.info("\nSource Binned (s_expr_ids):")
        logger.info(f"  Overall min: {min(all_s_expr_ids_min)}")
        logger.info(f"  Overall max: {max(all_s_expr_ids_max)}")
        logger.info(f"  Batch-wise min range: [{min(all_s_expr_ids_min)}, {max(all_s_expr_ids_min)}]")
        logger.info(f"  Batch-wise max range: [{min(all_s_expr_ids_max)}, {max(all_s_expr_ids_max)}]")

    if all_t_expr_ids_min:
        logger.info("\nTarget Binned (t_expr_ids):")
        logger.info(f"  Overall min: {min(all_t_expr_ids_min)}")
        logger.info(f"  Overall max: {max(all_t_expr_ids_max)}")
        logger.info(f"  Batch-wise min range: [{min(all_t_expr_ids_min)}, {max(all_t_expr_ids_min)}]")
        logger.info(f"  Batch-wise max range: [{min(all_t_expr_ids_max)}, {max(all_t_expr_ids_max)}]")

    logger.info("\n" + "=" * 80)
    logger.info("Data module test completed successfully!")
    logger.info("=" * 80)


def main():
    """Main function to test synthetic data loading."""
    # Path to preprocessed synthetic data
    # Update this path to match your actual data directory
    data_path = Path("/your/path/to/TOMIC/expertments/data_process/synthetic_processed/synthetic_400")

    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("Please update the data_path in the script to point to your preprocessed data directory")
        return

    # Test with binning enabled (discrete expression values)
    logger.info("\n" + "=" * 80)
    logger.info("Test: Discrete Expression (With Binning)")
    logger.info("=" * 80)
    logger.info("Note: DomainDataModuleDatmp supports both binned and continuous expression values")
    test_data_module(
        root_data_path=data_path,
        binning=51,
        train_batch_size=256,
        test_batch_size=256,
        num_workers=0,
    )


if __name__ == "__main__":
    main()
