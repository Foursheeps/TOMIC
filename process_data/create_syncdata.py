#!/usr/bin/env python3
"""
Process synthetic data using updated ScanpyPreprocessor.

This script generates synthetic data and processes it using the updated
ScanpyPreprocessor, following the structure of process_GSE173958.py.
"""

import sys
from collections.abc import Callable
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.datasets import make_classification

# Add project root to Python path
sys.path.append("/your/path/to/TOMIC")

from tomic import get_logger
from tomic.dataset.preprocessing import (
    GET_GEN_FLAG,
    preprocess,
)

# Use unified logger
logger = get_logger("data_process")


def get_total_cells_to_simulate(cells_per_organ: dict[str, int]) -> int:
    """
    Calculate total number of cells to simulate based on cells_per_organ mapping.

    The calculation uses the formula: max_cells * n_organs, where max_cells is the
    maximum number of cells per organ and n_organs is the number of organs.

    Args:
        cells_per_organ: Dictionary mapping organ names (str) to number of cells (int) per organ.
            All values must be positive integers.

    Returns:
        Total number of cells (int): max_cells * n_organs, where:
        - max_cells: Maximum value in cells_per_organ.values()
        - n_organs: Number of keys in cells_per_organ

    Raises:
        ValueError: If cells_per_organ is empty or contains non-positive values
    """
    if not cells_per_organ:
        raise ValueError("cells_per_organ must be non-empty")
    values = list(cells_per_organ.values())
    if any(v <= 0 for v in values):
        raise ValueError("All values in cells_per_organ must be positive integers")
    max_cells = max(values)
    return len(values) * max_cells


def simulate_count_matrix(
    X_ref: np.ndarray,
    method: str = "zinb",
    theta: float = 1.0,
    dropout_baseline: float = 0.5,
    dropout_slope: float = 5.0,
    per_gene: bool = False,
    scale_factor: float = 50.0,
    random_state: int | None = None,
    verbose: bool = True,
) -> np.ndarray:
    """
    Generate count matrix from reference continuous expression matrix using NB or ZINB.

    Args:
        X_ref: Reference continuous expression matrix (non-negative, typically in [0, 1])
        method: Method to use, "nb" for Negative Binomial or "zinb" for Zero-Inflated NB
        theta: Controls overdispersion (smaller = more dispersed). Must be positive.
        dropout_baseline: Dropout baseline (for ZINB only, larger = higher overall dropout).
            Should be in [0, 1].
        dropout_slope: Dropout slope with expression (for ZINB only, larger = more dropout at low expression).
            Must be positive.
        per_gene: Whether to set different dropout baseline per gene (for ZINB only)
        scale_factor: Scaling factor to map continuous expression to expected counts (sequencing depth).
            Must be positive.
        random_state: Random seed for reproducibility
        verbose: Whether to log zero fraction statistics

    Returns:
        Count matrix with same shape as X_ref, containing non-negative integer counts

    Raises:
        ValueError: If method is not "nb" or "zinb", or if parameters are invalid
    """
    # Input validation
    if method not in ("nb", "zinb"):
        raise ValueError(f"method must be 'nb' or 'zinb', got {method}")
    if theta <= 0:
        raise ValueError(f"theta must be positive, got {theta}")
    if scale_factor <= 0:
        raise ValueError(f"scale_factor must be positive, got {scale_factor}")
    if method == "zinb" and dropout_slope <= 0:
        raise ValueError(f"dropout_slope must be positive for ZINB, got {dropout_slope}")

    rng = np.random.RandomState(random_state)
    X_nonneg = np.maximum(X_ref, 0.0)

    # Calculate NB mean mu
    mu = X_nonneg * scale_factor + 1e-8

    # Gamma-Poisson mixture to generate NB counts
    gamma_shape = max(theta, 1e-8)
    gamma_scale = mu / gamma_shape
    gamma_means = rng.gamma(shape=gamma_shape, scale=gamma_scale, size=mu.shape)
    nb_counts = rng.poisson(lam=gamma_means).astype(int)

    if method == "nb":
        if verbose:
            zero_elements = np.sum(nb_counts == 0)
            zero_fraction = zero_elements / nb_counts.size
            logger.debug(f"NB Counts - Zero fraction: {zero_fraction:.4f} ({zero_elements}/{nb_counts.size})")
        return nb_counts

    elif method == "zinb":
        # Calculate dropout probability and apply zero inflation
        if per_gene:
            gene_mean = X_nonneg.mean(axis=0, keepdims=True)
            baseline_matrix = dropout_baseline * (1.0 - (gene_mean / (gene_mean.max() + 1e-8)))
        else:
            baseline_matrix = dropout_baseline

        # Normalize expression to 0..1 for dropout link
        max_val = X_nonneg.max()
        if max_val > 0:
            X_norm = X_nonneg / max_val
        else:
            X_norm = X_nonneg

        pi = expit(baseline_matrix - dropout_slope * X_norm)
        # Set positions to zero with probability pi (zero inflation)
        drop_mask = rng.binomial(1, pi).astype(bool)
        zinb_counts = nb_counts.copy()
        zinb_counts[drop_mask] = 0

        if verbose:
            zero_elements = np.sum(zinb_counts == 0)
            zero_fraction = zero_elements / zinb_counts.size
            logger.debug(f"ZINB Counts - Zero fraction: {zero_fraction:.4f} ({zero_elements}/{zinb_counts.size})")
        return zinb_counts

    else:
        # This should never be reached due to validation above, but kept for safety
        raise ValueError(f"Invalid method: {method}. Choose 'nb' or 'zinb'.")


def generate_reference_distributions_with_mapping(
    sample_cells_mapping: dict[str, int] | None = None,
    n_genes: int = 200,
    n_highly_variable_genes: int = 100,
    class_sep: float = 1.0,
    mapping_func: Callable[[np.ndarray], np.ndarray] | None = None,
    metastasis_noise_weight: float = 0.1,
    primary_noise_weight: float = 0.1,
    random_state: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate reference distributions for primary and metastasis tumors using classification data
    and establish relationship between them via mapping function.

    The function:
    1. Generates metastasis reference distribution using sklearn.make_classification
    2. Maps metastasis distribution to primary distribution using mapping_func
    3. Adds noise to both distributions
    4. Normalizes values to [0, 1] range
    5. Optionally adjusts sample counts according to sample_cells_mapping

    Args:
        sample_cells_mapping: Dictionary mapping organ names (str) to number of cells (int) per organ.
            If None, uses default values (1000 samples, 4 classes).
        n_genes: Number of genes (features). Must be positive. Default: 200.
        n_highly_variable_genes: Number of highly variable genes (informative features).
            Must be <= n_genes. Default: 100.
        class_sep: Class separation parameter (controls cluster tightness). Must be positive.
            Larger values create more separated clusters. Default: 1.0.
        mapping_func: Function to map metastasis distribution to primary distribution.
            Should take a numpy array and return a numpy array of the same shape.
            If None, uses default ReLU transformation: max(0, 0.4*x^2 + 0.6*x - 0.1).
        metastasis_noise_weight: Metastasis noise weight. Should be in [0, 1].
            Controls the amount of noise added to metastasis distribution. Default: 0.1.
        primary_noise_weight: Primary noise weight. Should be in [0, 1].
            Controls the amount of noise added to primary distribution. Default: 0.1.
        random_state: Random seed for reproducibility. If None, uses random seed.

    Returns:
        Tuple of (primary_ref, metastasis_ref, primary_labels, metastasis_labels):
        - primary_ref (np.ndarray): Primary tumor reference distribution as continuous expression matrix.
            Shape: (n_samples, n_genes), values in [0, 1].
        - metastasis_ref (np.ndarray): Metastasis reference distribution as continuous expression matrix.
            Shape: (n_samples, n_genes), values in [0, 1].
        - primary_labels (np.ndarray): Primary tumor cluster labels. Shape: (n_samples,).
            If sample_cells_mapping is provided, labels are organ names (str).
            Otherwise, labels are integers (0 to n_classes-1).
        - metastasis_labels (np.ndarray): Metastasis labels. Shape: (n_samples,).
            Same as primary_labels.

    Raises:
        ValueError: If n_genes <= 0, n_highly_variable_genes <= 0, or
            n_highly_variable_genes > n_genes, or class_sep <= 0.
    """
    # Input validation
    if n_genes <= 0:
        raise ValueError(f"n_genes must be positive, got {n_genes}")
    if n_highly_variable_genes <= 0:
        raise ValueError(f"n_highly_variable_genes must be positive, got {n_highly_variable_genes}")
    if n_highly_variable_genes > n_genes:
        raise ValueError(f"n_highly_variable_genes ({n_highly_variable_genes}) must be <= n_genes ({n_genes})")
    if class_sep <= 0:
        raise ValueError(f"class_sep must be positive, got {class_sep}")

    if mapping_func is None:
        # Default mapping function: ReLU transformation
        def default_mapping_func(x):
            return np.maximum(0, 0.4 * x**2 + 0.6 * x - 0.1)

        mapping_func = default_mapping_func

    def add_noise(data: np.ndarray, noise_weight: float, rng: np.random.RandomState) -> np.ndarray:
        noise = rng.randn(*data.shape)
        return noise_weight * noise + (1 - noise_weight) * data

    rng = np.random.RandomState(random_state)

    # Generate metastasis reference distribution
    n_samples = get_total_cells_to_simulate(sample_cells_mapping) if sample_cells_mapping else 1000
    n_classes = len(sample_cells_mapping) if sample_cells_mapping else 4

    metastasis_ref, labels = make_classification(
        n_samples=n_samples,
        n_features=n_genes,
        n_informative=n_highly_variable_genes,
        n_redundant=n_genes - n_highly_variable_genes,
        n_classes=n_classes,
        n_clusters_per_class=1,
        class_sep=class_sep,
        random_state=rng,
    )

    # Add noise and normalize
    metastasis_ref = add_noise(metastasis_ref, metastasis_noise_weight, rng)
    metastasis_ref = np.clip(metastasis_ref, 0, 1)  # Ensure data in (0, 1) range

    # Generate primary reference distribution via mapping function
    primary_ref = mapping_func(metastasis_ref)
    primary_ref = add_noise(primary_ref, primary_noise_weight, rng)
    primary_ref = np.clip(primary_ref, 0, 1)  # Ensure data in (0, 1) range

    if sample_cells_mapping is not None:
        # Adjust sample counts according to sample_cells_mapping
        primary_indices = []
        metastasis_indices = []
        start_idx = 0
        for organ, n_cells in sample_cells_mapping.items():
            end_idx = start_idx + n_cells
            primary_indices.extend(range(start_idx, end_idx))
            metastasis_indices.extend(range(start_idx, end_idx))
            start_idx += n_cells
        primary_ref = primary_ref[primary_indices, :]
        metastasis_ref = metastasis_ref[metastasis_indices, :]
        labels = labels[primary_indices]
        labels_categories = list(sample_cells_mapping.keys())
        labels = np.array([labels_categories[label] for label in labels])

    return primary_ref, metastasis_ref, labels, labels


def create_synthetic_ann_data(
    cells_per_organ: dict[str, int] | None = None,
    n_genes: int = 3000,
    n_highly_variable_genes: int = 400,
    class_sep: float = 10.0,
    mapping_func: Callable[[np.ndarray], np.ndarray] | None = None,
    metastasis_noise_weight: float = 0.8,
    primary_noise_weight: float = 0.3,
    count_method: str = "zinb",
    count_theta: float = 0.05,
    count_scale_factor: float = 50.0,
    random_state: int = 42,
    verbose: bool = True,
) -> tuple[ad.AnnData, ad.AnnData, ad.AnnData]:
    """
    Create synthetic AnnData for domain adaptation experiments.

    This function follows the methodology from notebook/01_gen_synthetic_data_p3.ipynb:
    1. Generate reference distributions for primary and metastasis tumors
    2. Convert to count matrices using NB or ZINB
    3. Create AnnData objects

    Args:
        cells_per_organ: Dictionary mapping organ names to number of cells
            (default: {"Liver": 40000, "Lung": 30002, "Stomach": 20000, "Peritoneum": 45000})
        n_genes: Number of genes (features). Must be positive.
        n_highly_variable_genes: Number of highly variable genes. Must be <= n_genes.
        class_sep: Class separation (controls cluster tightness). Must be positive.
        mapping_func: Function to map metastasis to primary distribution.
            If None, uses default ReLU transformation: max(0, 0.4*x^2 + 0.6*x - 0.1)
        metastasis_noise_weight: Metastasis noise weight. Should be in [0, 1].
        primary_noise_weight: Primary noise weight. Should be in [0, 1].
        count_method: Count matrix method ("nb" or "zinb")
        count_theta: Theta parameter for count matrix generation. Must be positive.
        count_scale_factor: Scale factor for count matrix generation. Must be positive.
        random_state: Random seed for reproducibility
        verbose: Whether to log progress information

    Returns:
        Tuple of (concat_adata, primary_adata, metastasis_adata):
        - concat_adata: Combined AnnData object containing both primary and metastasis data
        - primary_adata: Primary AnnData object
        - metastasis_adata: Metastasis AnnData object

    Raises:
        ValueError: If n_highly_variable_genes > n_genes or if count_method is invalid
    """
    # Input validation
    if n_genes <= 0:
        raise ValueError(f"n_genes must be positive, got {n_genes}")
    if n_highly_variable_genes <= 0:
        raise ValueError(f"n_highly_variable_genes must be positive, got {n_highly_variable_genes}")
    if n_highly_variable_genes > n_genes:
        raise ValueError(f"n_highly_variable_genes ({n_highly_variable_genes}) must be <= n_genes ({n_genes})")
    if class_sep <= 0:
        raise ValueError(f"class_sep must be positive, got {class_sep}")
    if count_method not in ("nb", "zinb"):
        raise ValueError(f"count_method must be 'nb' or 'zinb', got {count_method}")
    if count_theta <= 0:
        raise ValueError(f"count_theta must be positive, got {count_theta}")
    if count_scale_factor <= 0:
        raise ValueError(f"count_scale_factor must be positive, got {count_scale_factor}")

    if cells_per_organ is None:
        cells_per_organ = {
            "Liver": 40000,
            "Lung": 30002,
            "Stomach": 20000,
            "Peritoneum": 45000,
        }

    if mapping_func is None:
        # Default mapping function: ReLU transformation
        def default_mapping_func(x):
            return np.maximum(0, 0.4 * x**2 + 0.6 * x - 0.1)

        mapping_func = default_mapping_func

    if verbose:
        logger.info("Generating reference distributions...")

    # Generate reference distributions
    primary_ref, metastasis_ref, primary_labels, metastasis_labels = generate_reference_distributions_with_mapping(
        sample_cells_mapping=cells_per_organ,
        n_genes=n_genes,
        n_highly_variable_genes=n_highly_variable_genes,
        class_sep=class_sep,
        mapping_func=mapping_func,
        metastasis_noise_weight=metastasis_noise_weight,
        primary_noise_weight=primary_noise_weight,
        random_state=random_state,
    )

    if verbose:
        logger.info(f"Primary Reference Shape: {primary_ref.shape}")
        logger.info(f"Metastasis Reference Shape: {metastasis_ref.shape}")
        logger.info("Converting to count matrices...")

    # Convert to count matrices
    count_primary = simulate_count_matrix(
        primary_ref,
        method=count_method,
        theta=count_theta,
        scale_factor=count_scale_factor,
        random_state=random_state,
        verbose=verbose,
    )
    count_metastasis = simulate_count_matrix(
        metastasis_ref,
        method=count_method,
        theta=count_theta,
        scale_factor=count_scale_factor,
        random_state=random_state + 1,  # Different seed for metastasis
        verbose=verbose,
    )

    # Generate gene names
    gene_names = [f"gene_{i:06d}" for i in range(n_genes)]

    # Create variable metadata
    var_sim = pd.DataFrame(index=gene_names)

    # Create observation metadata
    n_samples = count_primary.shape[0]
    primary_obs = pd.DataFrame(
        {"organ": primary_labels, "dataset": "primary"},
        index=[f"cell_{i:06d}" for i in range(n_samples)],
    )
    metastasis_obs = pd.DataFrame(
        {"organ": metastasis_labels, "dataset": "metastasis"},
        index=[f"cell_{i:06d}" for i in range(n_samples)],
    )

    # Create AnnData objects
    primary_adata = ad.AnnData(X=count_primary, obs=primary_obs, var=var_sim)
    metastasis_adata = ad.AnnData(X=count_metastasis, obs=metastasis_obs, var=var_sim)

    # Concatenate
    concat_adata = ad.concat({"primary": primary_adata, "metastasis": metastasis_adata})
    concat_adata.obs_names_make_unique()

    # Mark highly variable genes
    concat_adata.var["highly_variable"] = [i < n_highly_variable_genes for i in range(n_genes)]

    if verbose:
        logger.info("Created synthetic AnnData:")
        logger.info(f"  Total samples: {len(concat_adata)}")
        logger.info(f"  Total genes: {len(concat_adata.var)}")
        logger.info(f"  Highly variable genes: {n_highly_variable_genes}")
        logger.info(f"  Organs: {list(cells_per_organ.keys())}")
        logger.info(f"  Primary samples: {sum(concat_adata.obs['dataset'] == 'primary')}")
        logger.info(f"  Metastasis samples: {sum(concat_adata.obs['dataset'] == 'metastasis')}")

    return concat_adata, primary_adata, metastasis_adata


def main(
    output_base_dir: Path | str = "/your/path/to/TOMIC/expertments/data_process",
    cells_per_organ: dict[str, int] | None = None,
    n_genes: int = 3000,
    n_highly_variable_genes: int = 400,
    class_sep: int = 10,
    mapping_func: Callable[["np.ndarray"], "np.ndarray"] | None = None,
    metastasis_noise_weight: float = 0.8,
    primary_noise_weight: float = 0.3,
    count_method: str = "zinb",
    count_theta: float = 0.05,
    count_scale_factor: float = 50.0,
    random_state: int = 42,
    overwrite: bool = False,
) -> None:
    """
    Main function to generate and process synthetic datasets.

    Args:
        output_base_dir: Base directory for output files
        cells_per_organ: Dictionary mapping organ names to number of cells
            (default: {"Liver": 40000, "Lung": 30002, "Stomach": 20000, "Peritoneum": 45000})
        n_genes: Number of genes (features). Must be positive.
        n_highly_variable_genes: Number of highly variable genes. Must be <= n_genes.
        class_sep: Class separation (controls cluster tightness). Must be positive.
        mapping_func: Function to map metastasis to primary distribution.
            If None, uses default ReLU transformation: max(0, 0.4*x^2 + 0.6*x - 0.1)
        metastasis_noise_weight: Metastasis noise weight. Should be in [0, 1].
        primary_noise_weight: Primary noise weight. Should be in [0, 1].
        count_method: Count matrix method ("nb" or "zinb")
        count_theta: Theta parameter for count matrix generation. Must be positive.
        count_scale_factor: Scale factor for count matrix generation. Must be positive.
        random_state: Random seed for reproducibility
        overwrite: Whether to overwrite existing data
    """
    output_base_dir = Path(output_base_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Create output directory
    sum_cells = sum(cells_per_organ.values())
    output_dir = output_base_dir / f"C{sum_cells}G{n_genes}H{n_highly_variable_genes}S{class_sep}"

    # Check if output directory already exists
    files, generated_flag = GET_GEN_FLAG(output_dir)
    print(files, generated_flag, overwrite, output_dir)
    if not generated_flag and not overwrite:
        logger.info(f"Data has already been generated in {output_dir}: {files}")
        return None

    # 1. Generate synthetic data
    logger.info("Generating synthetic data...")
    concat_adata, primary_adata, metastasis_adata = create_synthetic_ann_data(
        cells_per_organ=cells_per_organ,
        n_genes=n_genes,
        n_highly_variable_genes=n_highly_variable_genes,
        class_sep=class_sep,
        mapping_func=mapping_func,
        metastasis_noise_weight=metastasis_noise_weight,
        primary_noise_weight=primary_noise_weight,
        count_method=count_method,
        count_theta=count_theta,
        count_scale_factor=count_scale_factor,
        random_state=random_state,
        verbose=True,
    )

    # Log value counts
    logger.info("\nOrgan and dataset value counts:")
    value_counts = concat_adata.obs[["organ", "dataset"]].value_counts()
    logger.info(f"\n{value_counts}")

    # 2. Process dataset with updated preprocessing
    logger.info("Processing synthetic data...")

    preprocess(
        output_dir=output_dir,
        concat_adata=concat_adata,
        primary_adata=primary_adata,
        metastasis_adata=metastasis_adata,
        n_highly_variable_genes=n_highly_variable_genes,
        batch_key=None,
        raw_data_path="synthetic",
        filter_gene_by_counts=False,
        filter_cell_by_counts=False,
    )

    # 3. Log completion
    logger.info(f"All results saved to: {output_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_base_dir",
        type=str,
        default="/your/path/to/TOMIC/expertments/data_process/Synthetic",
    )
    parser.add_argument(
        "--cells_per_organ", type=str, default="dict(Liver=2000,Lung=1500,Stomach=1000,Peritoneum=2500)"
    )
    parser.add_argument("--n_genes", type=int, default=1200)
    parser.add_argument("--n_highly_variable_genes", type=int, default=400)
    parser.add_argument("--class_sep", type=int, default=10)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--overwrite", type=int, default=0)
    args = parser.parse_args()
    overwrite_flag = False if args.overwrite == 0 else True
    cells_per_organ = eval(args.cells_per_organ)
    main(
        output_base_dir=args.output_base_dir,
        cells_per_organ=cells_per_organ,
        n_genes=args.n_genes,
        n_highly_variable_genes=args.n_highly_variable_genes,
        class_sep=args.class_sep,
        random_state=args.random_state,
        overwrite=overwrite_flag,
    )
