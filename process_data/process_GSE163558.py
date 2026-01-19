#!/usr/bin/env python3
"""
Process GSE163558 dataset using updated ScanpyPreprocessor.

This script processes GSE163558 dataset which contains primary tumor (PT),
normal tissue (NT), lymph node (LN), and other (O) samples.
"""

import sys
from pathlib import Path

# Add project root to Python path
sys.path.append("/your/path/to/TOMIC")

import os
import random
import tarfile
import tempfile

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
DATA_DIR = "/your/path/to/raw/data/GSE163558_RAW"

# Sample names and types
# PT: Primary Tumor, NT: Normal Tissue, LN: Lymph Node, O: Other, P: Peritoneum, Li: Liver
SAMPLE_NAMES = [
    "GSM5004180_PT1",
    "GSM5004181_PT2",
    "GSM5004182_PT3",
    # "GSM5004183_NT1",
    "GSM5004184_LN1",
    "GSM5004185_LN2",
    "GSM5004186_O1",
    "GSM5004187_P1",
    "GSM5004188_Li1",
    "GSM5004189_Li2",
]

# Filter organs to keep (PT and NT as primary, LN, O1, P1, Li as metastasis)
# Note: Based on notebook debugging, extraction uses [:2] from suffix
# PT1 -> PT, NT1 -> NT, LN1 -> LN, O1 -> O1, P1 -> P1, Li1 -> Li
# Default keep_organs matches notebook test: ["PT", "NT", "LN", "O"]
# But to match actual extraction results, use: ["PT", "NT", "LN", "O1", "P1", "Li"]
# Updated to include all metastasis organs based on notebook debugging
KEEP_ORGANS = ["PT", "LN", "O1", "P1", "Li"]

# Set random seed
random.seed(42)


def extract_tar_if_needed(tar_path: str, extract_dir: str) -> str:
    """Extract tar file if needed and return the data directory."""
    if os.path.isdir(tar_path):
        return tar_path

    # Extract to temporary directory
    logger.info(f"Extracting {tar_path} to {extract_dir}...")
    os.makedirs(extract_dir, exist_ok=True)

    with tarfile.open(tar_path, "r") as tar:
        tar.extractall(extract_dir)

    return extract_dir


def load_sample_data(sample_name: str, data_dir: str) -> ad.AnnData:
    """Load a single sample from the dataset."""
    matrix_file = os.path.join(data_dir, sample_name + "_matrix.mtx.gz")
    features_file = os.path.join(data_dir, sample_name + "_features.tsv.gz")
    barcodes_file = os.path.join(data_dir, sample_name + "_barcodes.tsv.gz")

    # Load matrix data
    adata = sc.read_mtx(matrix_file).T  # Transpose to have cells as rows

    # Load features and barcodes
    features = pd.read_csv(features_file, header=None, sep="\t")
    barcodes = pd.read_csv(barcodes_file, header=None, sep="\t")

    # Set gene information
    adata.var["gene_symbols"] = features[1].values if features.shape[1] > 1 else features[0].values
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
            features_file = os.path.join(data_dir, sample_name + "_features.tsv.gz")
            features = pd.read_csv(features_file, header=None, sep="\t")

    # Combine all samples
    adata_combined = sc.concat(adata_list)
    adata_combined.var["gene_symbols"] = features[1].values if features.shape[1] > 1 else features[0].values

    return adata_combined


def add_organ_labels(adata: ad.AnnData, keep_organs: list[str] | None = None) -> ad.AnnData:
    """Add organ and dataset labels to the AnnData object."""
    if keep_organs is None:
        keep_organs = KEEP_ORGANS

    assert "sample" in adata.obs.columns, "'sample' column is required in adata.obs"

    # Extract organ type from sample name
    # Format: GSM5004180_PT1 -> PT, GSM5004188_Li1 -> Li, etc.
    # Based on notebook debugging: use first 2 characters of suffix
    # PT1 -> PT, NT1 -> NT, LN1 -> LN, O1 -> O1, P1 -> P1, Li1 -> Li
    adata.obs["organ"] = adata.obs["sample"].apply(lambda x: x.split("_")[-1][:2])

    # Filter to keep only specified organs
    adata = adata[adata.obs["organ"].isin(keep_organs)].copy()

    # Add dataset label (primary vs metastasis)
    # PT and NT are primary, LN, O1, P1, Li are metastasis
    adata.obs["dataset"] = adata.obs["organ"].apply(lambda x: "primary" if x in ["PT", "NT"] else "metastasis")

    # Drop unnecessary columns
    adata.obs.drop(columns=["barcode", "sample"], inplace=True)

    return adata


def random_replace_organs(adata: ad.AnnData) -> ad.AnnData:
    """Randomly replace PT organs with metastasis organs."""
    assert "organ" in adata.obs.columns, "'organ' column is required in adata.obs"

    def random_replacements(x):
        if x == "PT":
            # Replace PT with random metastasis organ
            # Based on notebook: available metastasis organs are LN, Li, O1, P1
            # But after extraction, they become LN, Li, O1, P1
            return random.choice(["LN", "Li", "O1", "P1"])
        return x

    adata.obs["organ"] = adata.obs["organ"].apply(random_replacements)
    return adata


def load_gse163558_data(
    data_dir: str | None = None,
    keep_organs: list[str] | None = None,
    apply_random_replacements: bool = True,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """
    Load and prepare GSE163558 dataset without baseline HVG filtering.
    """
    if data_dir is None:
        data_dir = DATA_DIR
    if keep_organs is None:
        keep_organs = KEEP_ORGANS

    logger.info("Loading samples...")

    # Use data directory directly (already extracted)
    # Extract tar file if needed
    if os.path.isfile(data_dir) and data_dir.endswith(".tar"):
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_dir = extract_tar_if_needed(data_dir, temp_dir)
            # Load and combine samples
            adata_combined = combine_samples(SAMPLE_NAMES, extracted_dir)
    else:
        # Load and combine samples directly from directory
        adata_combined = combine_samples(SAMPLE_NAMES, data_dir)

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
    logger.info("\nOrgan and dataset value counts:")
    value_counts = adata_combined.obs[["organ", "dataset"]].value_counts()
    logger.info(f"\n{value_counts}")

    return adata_combined, primary_adata, metastasis_adata


def main(
    output_base_dir: Path | str = "/your/path/to/TOMIC/expertments/data_process",
    data_dir: str | None = None,
    keep_organs: list[str] | None = None,
    apply_random_replacements: bool = True,
    n_highly_variable_genes: int = 2000,
    overwrite: bool = False,
) -> None:
    """
    Main function to process GSE163558 datasets.

    Args:
        output_base_dir: Base directory for output files
        data_dir: Directory containing raw data files (tar file or extracted directory)
        keep_organs: List of organs to keep
        apply_random_replacements: Whether to apply random organ replacements
        n_highly_variable_genes: Number of HVG genes (used for seq_len in config)
        overwrite: Whether to overwrite existing data
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = output_base_dir / f"GSE163558_{n_highly_variable_genes}"

    # Check if output directory already exists
    files, generated_flag = GET_GEN_FLAG(output_dir)
    if not generated_flag and not overwrite:
        logger.info(f"Data has already been generated in {output_dir}: {files}")
        return None

    # 1. Load data without baseline HVG filtering
    concat_adata, primary_adata, metastasis_adata = load_gse163558_data(
        data_dir=data_dir,
        keep_organs=keep_organs,
        apply_random_replacements=apply_random_replacements,
    )

    # 2. Process dataset with updated preprocessing
    logger.info("Processing GSE163558...")

    preprocess(
        output_dir=output_dir,
        concat_adata=concat_adata,
        primary_adata=primary_adata,
        metastasis_adata=metastasis_adata,
        n_highly_variable_genes=n_highly_variable_genes,
        batch_key=None,
        raw_data_path=data_dir,
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
    )

    # 3. Log completion
    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Process GSE163558 dataset")
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/your/path/to/TOMIC/data_process/GSE163558",
        help="Base directory for output files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/your/path/to/raw/data/GSE163558_RAW",
        help="Directory containing raw data files (tar file or extracted directory)",
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
        output_base_dir=args.output_base_dir,
        data_dir=args.data_dir,
        n_highly_variable_genes=args.n_highly_variable_genes,
        overwrite=bool(args.overwrite),
    )
