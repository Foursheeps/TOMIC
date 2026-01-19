#!/usr/bin/env python3
"""
Process GSE123902 dataset using updated ScanpyPreprocessor.

This script processes GSE123902 dataset which contains primary tumor (PRIMARY),
metastasis (BRAIN, ADRENAL, BONE) samples in CSV format.
Based on notebook debugging: CSV files are already in cells × genes format.
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
DATA_DIR = "/your/path/to/raw/data/GSE123902_RAW"

# Sample names - based on notebook debugging
# Format: GSM3516662_MSK_LX653_PRIMARY_TUMOUR_dense.csv.gz
# Note: Based on notebook, we use a subset of samples that have compatible gene names
SAMPLE_NAMES = [
    "GSM3516663_MSK_LX661_PRIMARY_TUMOUR",
    "GSM3516664_MSK_LX666_METASTASIS",
    "GSM3516665_MSK_LX675_PRIMARY_TUMOUR",
    "GSM3516667_MSK_LX676_PRIMARY_TUMOUR",
    "GSM3516668_MSK_LX255B_METASTASIS",
    "GSM3516669_MSK_LX679_PRIMARY_TUMOUR",
    "GSM3516670_MSK_LX680_PRIMARY_TUMOUR",
    "GSM3516671_MSK_LX681_METASTASIS",
    "GSM3516672_MSK_LX682_PRIMARY_TUMOUR",
    "GSM3516674_MSK_LX684_PRIMARY_TUMOUR",
    "GSM3516677_MSK_LX699_METASTASIS",
    "GSM3516678_MSK_LX701_METASTASIS",
]

# Organ mapping for specific samples (from notebook debugging)
# This maps sample names to specific organ types
MAP_ORGAN = {
    "GSM3516663_MSK_LX661_PRIMARY_TUMOUR": "PRIMARY",
    "GSM3516664_MSK_LX666_METASTASIS": "BONE",
    "GSM3516665_MSK_LX675_PRIMARY_TUMOUR": "PRIMARY",
    "GSM3516667_MSK_LX676_PRIMARY_TUMOUR": "PRIMARY",
    "GSM3516668_MSK_LX255B_METASTASIS": "BRAIN",
    "GSM3516669_MSK_LX679_PRIMARY_TUMOUR": "PRIMARY",
    "GSM3516670_MSK_LX680_PRIMARY_TUMOUR": "PRIMARY",
    "GSM3516671_MSK_LX681_METASTASIS": "BRAIN",
    "GSM3516672_MSK_LX682_PRIMARY_TUMOUR": "PRIMARY",
    "GSM3516674_MSK_LX684_PRIMARY_TUMOUR": "PRIMARY",
    "GSM3516677_MSK_LX699_METASTASIS": "ADRENAL",
    "GSM3516678_MSK_LX701_METASTASIS": "BRAIN",
}

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
    """Load a single sample from the dataset (CSV format)."""
    csv_file = os.path.join(data_dir, sample_name + "_dense.csv.gz")

    # Load CSV data
    # Note: Based on notebook debugging, CSV files are already in cells × genes format
    # The first column is cell barcodes (index), columns are genes
    df = pd.read_csv(csv_file, index_col=0, compression="gzip")

    # Convert gene index to string to ensure compatibility
    # The gene names might be integers, convert to string
    df.columns = df.columns.astype(str)

    # Create AnnData object
    adata = ad.AnnData(df)

    # Set gene information
    adata.var_names = df.columns
    adata.var["gene_symbols"] = df.columns.values

    # Set cell information
    adata.obs_names = [f"{sample_name}_{bc}" for bc in df.index]
    adata.obs["barcode"] = df.index.values
    adata.obs["sample"] = sample_name

    return adata


def combine_samples(sample_names: list[str], data_dir: str) -> ad.AnnData:
    """Load and combine multiple samples into a single AnnData object."""
    if not sample_names:
        raise ValueError("sample_names must be non-empty")

    adata_list = []
    all_genes = None

    for sample_name in sample_names:
        logger.info(f"Loading sample: {sample_name}...")
        adata = load_sample_data(sample_name, data_dir)
        adata_list.append(adata)
        if all_genes is None:
            all_genes = set(adata.var_names)
        else:
            # Use union to get all genes (outer join)
            all_genes = set(adata.var_names) | all_genes
        logger.info(f"  Genes: {adata.n_vars}, Total genes: {len(all_genes)}")

    # Combine all samples using outer join with fill_value=0
    # Based on notebook: use outer join to keep all genes
    logger.info("Combining samples...")
    adata_combined = sc.concat(adata_list, join="outer", fill_value=0)
    logger.info(f"  Combined total genes: {adata_combined.n_vars}")

    return adata_combined


def add_organ_labels(adata: ad.AnnData, keep_organs: list[str] | None = None) -> ad.AnnData:
    """Add organ and dataset labels to the AnnData object."""
    assert "sample" in adata.obs.columns, "'sample' column is required in adata.obs"

    # Extract organ type from MAP_ORGAN
    def extract_organ_type(sample_name: str) -> str:
        """Extract organ type from sample name using MAP_ORGAN."""
        return MAP_ORGAN.get(sample_name, "UNKNOWN")

    adata.obs["organ"] = adata.obs["sample"].apply(extract_organ_type)

    # Filter to keep only specified organs
    if keep_organs:
        adata = adata[adata.obs["organ"].isin(keep_organs)].copy()

    # Add dataset label (primary vs metastasis)
    # PRIMARY is primary, others (BRAIN, ADRENAL, BONE) are metastasis
    adata.obs["dataset"] = adata.obs["organ"].apply(lambda x: "primary" if x == "PRIMARY" else "metastasis")

    # Drop unnecessary columns
    adata.obs.drop(columns=["barcode", "sample"], inplace=True)

    return adata


def random_replace_organs(adata: ad.AnnData) -> ad.AnnData:
    """Randomly replace PRIMARY organs with metastasis organs."""
    assert "organ" in adata.obs.columns, "'organ' column is required in adata.obs"

    def random_replacements(x):
        if x == "PRIMARY":
            # Replace PRIMARY with random metastasis organ
            return random.choice(["BONE", "BRAIN", "ADRENAL"])
        return x

    adata.obs["organ"] = adata.obs["organ"].apply(random_replacements)
    return adata


def load_gse123902_data(
    data_dir: str | None = None,
    keep_organs: list[str] | None = None,
    apply_random_replacements: bool = True,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """
    Load and prepare GSE123902 dataset without baseline HVG filtering.
    """
    if data_dir is None:
        data_dir = DATA_DIR

    logger.info("Loading samples...")

    # Use predefined sample names from notebook debugging
    # This ensures we use samples with compatible gene names
    sample_names = SAMPLE_NAMES
    logger.info(f"Using {len(sample_names)} predefined samples: {sample_names[:3]}...")

    # Use data directory directly (already extracted)
    # Extract tar file if needed
    if os.path.isfile(data_dir) and data_dir.endswith(".tar"):
        with tempfile.TemporaryDirectory() as temp_dir:
            extracted_dir = extract_tar_if_needed(data_dir, temp_dir)
            # Load and combine samples
            adata_combined = combine_samples(sample_names, extracted_dir)
    else:
        # Load and combine samples directly from directory
        adata_combined = combine_samples(sample_names, data_dir)

    # Add organ labels
    logger.info("Adding organ labels...")
    adata_combined = add_organ_labels(adata_combined, keep_organs=keep_organs)

    # Randomly replace PRIMARY organs if requested
    if apply_random_replacements:
        logger.info("Applying random organ replacements...")
        adata_combined = random_replace_organs(adata_combined)

    # Convert to dense format if sparse
    if hasattr(adata_combined.X, "toarray"):
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
    Main function to process GSE123902 datasets.

    Args:
        output_base_dir: Base directory for output files
        data_dir: Directory containing raw data files (tar file or extracted directory)
        keep_organs: List of organs to keep (PRIMARY, BRAIN, ADRENAL, BONE)
        apply_random_replacements: Whether to apply random organ replacements
        n_highly_variable_genes: Number of HVG genes (used for seq_len in config)
        overwrite: Whether to overwrite existing data
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    output_dir = output_base_dir / f"GSE123902_{n_highly_variable_genes}"

    # Check if output directory already exists
    files, generated_flag = GET_GEN_FLAG(output_dir)
    if not generated_flag and not overwrite:
        logger.info(f"Data has already been generated in {output_dir}: {files}")
        return None

    # 1. Load data without baseline HVG filtering
    concat_adata, primary_adata, metastasis_adata = load_gse123902_data(
        data_dir=data_dir,
        keep_organs=keep_organs,
        apply_random_replacements=apply_random_replacements,
    )

    # 2. Process dataset with updated preprocessing
    logger.info("Processing GSE123902...")

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

    parser = argparse.ArgumentParser(description="Process GSE123902 dataset")
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/your/path/to/TOMIC/expertments/data_process/GSE123902",
        help="Base directory for output files",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="/your/path/to/raw/data/GSE123902_RAW",
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
