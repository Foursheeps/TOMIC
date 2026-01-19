#!/usr/bin/env python3
"""
Process GSE173958 dataset using updated ScanpyPreprocessor.

This script processes GSE173958 M1, M2, and M1_M2 datasets using the updated
ScanpyPreprocessor.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append("/your/path/to/TOMIC")

import os
import random

import anndata as ad
import pandas as pd
import scanpy as sc

from datmp import get_logger
from datmp.dataset.preprocessing import (
    GET_GEN_FLAG,
    preprocess,
)

# Use unified logger
logger = get_logger("data_process")

# Configuration
DATA_DIR = "/your/path/to/raw_data/GSE173958_RAW"

# Organ types
ORGANS = [
    "PT",  # 原发
    "SS",  # 手术
    "Met",  # 腹膜转移
    "Liver",  # 肝转移
    "Lung",  # 肺转移
    "CTC",  # 循环肿瘤细胞
]

# M1 Sample names
M1_SAMPLE_NAMES = [
    "GSM5283482_M1-PT",
    "GSM5283484_M1-Met",
    "GSM5283485_M1-Liver",
    "GSM5283486_M1-Lung",
]

# M2 Sample names
M2_SAMPLE_NAMES = [
    "GSM5283488_M2-PT",
    "GSM5283489_M2-Met",
    "GSM5283490_M2-Liver",
    "GSM5283491_M2-Lung",
]

# Filter organs to keep
KEEP_ORGANS = ["PT", "Met", "Liver", "Lung"]

# Set random seed
random.seed(42)


def load_sample_data(sample_name: str, data_dir: str) -> ad.AnnData:
    """Load a single sample from the dataset."""
    matrix_file = os.path.join(data_dir, sample_name + "-matrix.mtx.gz")
    features_file = os.path.join(data_dir, sample_name + "-features.tsv.gz")
    barcodes_file = os.path.join(data_dir, sample_name + "-barcodes.tsv.gz")

    # Load matrix data
    adata = sc.read_mtx(matrix_file).T  # Transpose to have cells as rows

    # Load features and barcodes
    features = pd.read_csv(features_file, header=None, sep="\t")
    barcodes = pd.read_csv(barcodes_file, header=None, sep="\t")

    # Set gene information
    adata.var["gene_symbols"] = features[1].values
    adata.var_names = features[0]

    # Set cell information
    adata.obs["barcode"] = barcodes[0].values
    adata.obs_names = [f"{sample_name}_{bc}" for bc in adata.obs["barcode"]]
    adata.obs["sample"] = sample_name

    return adata


def combine_samples(sample_names: list[str], data_dir: str) -> ad.AnnData:
    """Load and combine multiple samples into a single AnnData object."""
    if not sample_names:
        raise ValueError("sample_names must be non-empty")

    adata_list = []
    features = None

    for sample_name in sample_names:
        adata = load_sample_data(sample_name, data_dir)
        adata_list.append(adata)
        if features is None:
            features_file = os.path.join(data_dir, sample_name + "-features.tsv.gz")
            features = pd.read_csv(features_file, header=None, sep="\t")

    # Combine all samples
    adata_combined = sc.concat(adata_list)
    adata_combined.var["gene_symbols"] = features[1].values

    return adata_combined


def add_organ_labels(adata: ad.AnnData, keep_organs: list[str] | None = None) -> ad.AnnData:
    """Add organ and dataset labels to the AnnData object."""
    if keep_organs is None:
        keep_organs = KEEP_ORGANS

    assert "sample" in adata.obs.columns, "'sample' column is required in adata.obs"

    # Extract organ from sample name
    adata.obs["organ"] = adata.obs["sample"].apply(lambda x: x.rsplit("-")[-1])

    # Filter to keep only specified organs
    adata = adata[adata.obs["organ"].isin(keep_organs)].copy()

    # Add dataset label (primary vs metastasis)
    adata.obs["dataset"] = adata.obs["organ"].apply(lambda x: "primary" if x in ["PT", "SS"] else "metastasis")

    # Drop unnecessary columns
    adata.obs.drop(columns=["barcode", "sample"], inplace=True)

    return adata


def random_replace_organs(adata: ad.AnnData) -> ad.AnnData:
    """Randomly replace PT organs with metastasis organs."""
    assert "organ" in adata.obs.columns, "'organ' column is required in adata.obs"

    def random_replacements(x):
        if x == "PT":
            return random.choice(["Met", "Liver", "Lung"])
        return x

    adata.obs["organ"] = adata.obs["organ"].apply(random_replacements)
    return adata


def load_gse173958_data(
    use_batch: str = "M1",
    data_dir: str | None = None,
    keep_organs: list[str] | None = None,
    apply_random_replacements: bool = True,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """
    Load and prepare GSE173958 dataset without baseline HVG filtering.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if keep_organs is None:
        keep_organs = KEEP_ORGANS

        # Load and combine samples

        logger.info("Loading samples...")

    if use_batch == "M1":
        sample_names = M1_SAMPLE_NAMES
    elif use_batch == "M2":
        sample_names = M2_SAMPLE_NAMES
    elif use_batch == "M1_M2":
        sample_names = M1_SAMPLE_NAMES + M2_SAMPLE_NAMES
    else:
        raise ValueError(f"use_batch must be 'M1', 'M2', or 'M1_M2', got {use_batch}")

    adata_combined = combine_samples(sample_names, data_dir)

    # Add batch labels (M1 or M2)
    logger.info("Adding batch labels...")
    if use_batch == "M1_M2":
        # For M1_M2, determine batch from sample name (contains "M1-" or "M2-")
        adata_combined.obs["batch"] = adata_combined.obs["sample"].apply(lambda x: "M1" if "_M1-" in x else "M2")
    else:
        # For M1 or M2 only, use the batch name directly
        adata_combined.obs["batch"] = use_batch

    # Add organ labels
    logger.info("Adding organ labels...")
    adata_combined = add_organ_labels(adata_combined, keep_organs=keep_organs)

    # Randomly replace PT organs if requested
    if apply_random_replacements:
        logger.info("Applying random organ replacements...")
        adata_combined = random_replace_organs(adata_combined)

    # Convert sparse matrix to dense format
    adata_combined.X = adata_combined.X.toarray()

    # Split into primary and metastasis
    primary_adata = adata_combined[adata_combined.obs["dataset"] == "primary"].copy()
    metastasis_adata = adata_combined[adata_combined.obs["dataset"] == "metastasis"].copy()

    # Log value counts
    logger.info("\nBatch, organ and dataset value counts:")
    value_counts = adata_combined.obs[["batch", "organ", "dataset"]].value_counts()
    logger.info(f"\n{value_counts}")

    return adata_combined, primary_adata, metastasis_adata


def main(
    use_batch: str = "M1",  # "M1", "M2", or "M1_M2"
    output_base_dir: Path | str = "/your/path/to/TOMIC/expertments/data_process",
    data_dir: str | None = None,
    keep_organs: list[str] | None = None,
    apply_random_replacements: bool = True,
    n_highly_variable_genes: int = 2000,
    overwrite: bool = False,
) -> None:
    """
    Main function to process GSE173958 datasets.

    Args:
        dataset_type: Type of dataset to process ("M1", "M2", or "M1_M2")
        output_base_dir: Base directory for output files
        data_dir: Directory containing raw data files
        keep_organs: List of organs to keep
        apply_random_replacements: Whether to apply random organ replacements
        n_highly_variable_genes: Number of HVG genes (used for seq_len in config)
        overwrite: Whether to overwrite existing data
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = output_base_dir / f"GSE173958_{use_batch}_{n_highly_variable_genes}"

    # Check if output directory already exists
    files, generated_flag = GET_GEN_FLAG(output_dir)
    if not generated_flag and not overwrite:
        logger.info(f"Data has already been generated in {output_dir}: {files}")
        return None

    # 1. Load data without baseline HVG filtering
    concat_adata, primary_adata, metastasis_adata = load_gse173958_data(
        use_batch=use_batch,
        data_dir=data_dir,
        keep_organs=keep_organs,
        apply_random_replacements=apply_random_replacements,
    )

    # 2. Process dataset with updated preprocessing
    logger.info(f"Processing GSE173958 {use_batch}...")

    preprocess(
        output_dir=output_dir,
        concat_adata=concat_adata,
        primary_adata=primary_adata,
        metastasis_adata=metastasis_adata,
        n_highly_variable_genes=n_highly_variable_genes,
        batch_key="batch" if use_batch == "M1_M2" else None,
        raw_data_path=data_dir,
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
    )

    # 3. Log completion

    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process GSE173958 dataset")
    parser.add_argument(
        "--use_batch",
        type=str,
        default="M1",
        choices=["M1", "M2", "M1_M2"],
        help="Batch to process: M1, M2, or M1_M2",
    )
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/your/path/to/TOMIC/expertments/data_process/GSE173958",
        help="Base directory for output files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/your/path/to/raw_data/GSE173958_RAW",
        help="Directory containing raw data files",
    )
    parser.add_argument(
        "--n_highly_variable_genes",
        type=int,
        default=1200,
        help="Number of highly variable genes",
    )
    parser.add_argument(
        "--overwrite",
        type=int,
        default=0,
        help="Whether to overwrite existing data (0=False, 1=True)",
    )

    args = parser.parse_args()

    main(
        use_batch=args.use_batch,
        output_base_dir=args.output_base_dir,
        data_dir=args.data_dir,
        n_highly_variable_genes=args.n_highly_variable_genes,
        overwrite=bool(args.overwrite),
    )
