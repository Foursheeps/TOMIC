"""
Preprocessing utilities for single-cell RNA-seq data.

This module provides preprocessing classes and functions for AnnData objects,
including normalization, scaling, dimensionality reduction, and visualization.
"""

import json
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import scanpy as sc

from ..logger import get_logger
from .scgpt_preprocess import Preprocessor

plt.style.use("ggplot")


# Use unified logger
logger = get_logger("dataset.preprocessing")

VOCAB_PATH = "vocab.txt"
INFO_CONFIG = "info_config.json"
PRIMARY_METASTASIS_H5AD = "synthetic_primary_metastasis.h5ad"
DOMAIN_UMAP = "Domain"
ORGAN_UMAP = "Organ"
PRIMARY_VS_ORGAN_UMAP = "Primary vs Organ"
METASTASIS_UMAP = "Metastasis"
PRIMARY_UMAP = "Primary"


def GET_GEN_FLAG(m: Path) -> bool:
    files = [
        VOCAB_PATH,
        INFO_CONFIG,
        PRIMARY_METASTASIS_H5AD,
        f"umap{DOMAIN_UMAP}.pdf",  # scanpy adds "umap_" prefix and ".png" extension
        f"umap{ORGAN_UMAP}.pdf",  # scanpy adds "umap_" prefix and ".png" extension
        f"umap{PRIMARY_VS_ORGAN_UMAP}.pdf",
        f"umap{METASTASIS_UMAP}.pdf",
        f"umap{PRIMARY_UMAP}.pdf",
    ]
    flag = any([not (m / file).exists() for file in files])
    return files, flag


class DatmpPreprocessor:
    """
    Preprocessor for single-cell RNA-seq data using scanpy pipeline.

    This class provides a standardized preprocessing pipeline including:
    - Normalization
    - Log transformation
    - Highly variable gene selection
    - Scaling
    - PCA
    - Neighbor graph construction
    - UMAP embedding
    """

    def __init__(
        self,
        use_key: str | None = "X",
        filter_gene_by_counts: int | bool = 3,
        filter_cell_by_counts: int | bool = False,
        normalize_total: float | bool | None = 1e4,
        result_normed_key: str | None = "X_normed",
        log1p: bool = True,
        result_log1p_key: str | None = "X_log1p",
        subset_hvg: int | bool = 1200,
        hvg_flavor: str = "seurat_v3",
        batch_key: str | None = "batch",
        dpi: int = 300,
        figsize: tuple[int, int] = (6, 6),
        **kwargs,
    ):
        """
        Initialize DatmpPreprocessor.

        Args:
            target_sum: Target sum for normalization. Must be positive.
                This is used as default for normalize_total if normalize_total is None.
            use_key: Key of AnnData to use for preprocessing. If None, uses "X".
            filter_gene_by_counts: Whether to filter genes by counts. If int, filter genes with
                counts less than this value. If False, no filtering.
            filter_cell_by_counts: Whether to filter cells by counts. If int, filter cells with
                counts less than this value. If False, no filtering.
            normalize_total: Target sum for normalization. If None, uses target_sum parameter.
            result_normed_key: Key to store normalized data. If None, stores directly in use_key.
            log1p: Whether to apply log1p transform. Default True.
            result_log1p_key: Key to store log1p transformed data. If None, stores directly in use_key.
            subset_hvg: Number of highly variable genes to select. If None, uses n_highly_variable_genes.
            hvg_use_key: Key to use for HVG calculation. If None, uses adata.X.
            hvg_flavor: Flavor of highly variable genes selection. Options: "seurat_v3", "seurat", "cell_ranger".
            batch_key: Key in adata.obs for batch information. Used for batch-aware HVG selection.
            dpi: Image resolution for plots. Must be positive.
            figsize: Figure size for plots as (width, height). Both values must be positive.

        """

        # Create Preprocessor instance in initialization
        self.preprocessor = Preprocessor(
            use_key=use_key,
            filter_gene_by_counts=filter_gene_by_counts,
            filter_cell_by_counts=filter_cell_by_counts,
            normalize_total=normalize_total,
            result_normed_key=result_normed_key,
            log1p=log1p,
            result_log1p_key=result_log1p_key,
            subset_hvg=subset_hvg,
            hvg_flavor=hvg_flavor,
        )

        # Store parameters
        self.batch_key = batch_key
        self.dpi = dpi
        self.figsize = figsize

    def preprocess(self, adata: ad.AnnData) -> ad.AnnData:
        """
        Apply standard scanpy preprocessing pipeline to AnnData.

        The preprocessing steps include:
        1. Store raw counts in adata.layers["raw_counts"]
        2. Filter genes/cells by counts (if enabled)
        3. Normalize total counts to target_sum
        4. Log transform (log1p)
        5. Select highly variable genes
        6. Calculate QC metrics
        7. Compute PCA
        8. Build neighbor graph
        9. Compute UMAP embedding

        Note: Binning is handled in DatasetDatmp.setup(), not in preprocessing.

        Args:
            adata: AnnData object to preprocess. Should contain raw count data in adata.X.

        Returns:
            Preprocessed AnnData object with:
            - adata.layers["X_normed"]: Normalized data
            - adata.layers["X_log1p"]: Log-transformed data
            - adata.layers["X_hvg"]: Highly variable genes data
            - adata.obsm["X_pca"]: PCA coordinates
            - adata.obsm["X_umap"]: UMAP coordinates
            - adata.uns["pca"]: PCA parameters
            - adata.uns["neighbors"]: Neighbor graph information
        """

        # Use Preprocessor instance created in __init__
        # Preprocessor modifies adata in place, no return value needed
        self.preprocessor(adata, batch_key=self.batch_key)

        return adata

    def plot_umap(
        self,
        adata: ad.AnnData,
        color: list[str] | str,
        title: str | None = None,
        save: str | None = None,
        show: bool = False,
        root_dir: Path | str | None = None,
    ) -> None:
        """
        Plot UMAP visualization.

        Args:
            adata: AnnData object to plot. Must have UMAP coordinates in adata.obsm["X_umap"].
            color: Column name(s) from adata.obs to color by. Can be a single string or list of strings.
                Each string should correspond to a column in adata.obs.
            title: Plot title. If None, scanpy will use default title.
            save: Filename to save plot. If None, plot is not saved. The filename should not include
                extension (scanpy will add appropriate extension based on format).
            show: Whether to display the plot interactively. If False, plot is only saved or not shown.
            root_dir: Directory to save plots. Only used if save is not None. If None and save is not None,
                uses scanpy's default figure directory.

        Returns:
            None. The plot is saved to file if save is specified, or displayed if show is True.

        Raises:
            KeyError: If color column(s) are not found in adata.obs
            ValueError: If UMAP coordinates are not found in adata.obsm["X_umap"]
        """
        if root_dir is not None:
            root_dir = Path(root_dir)
            root_dir.mkdir(parents=True, exist_ok=True)
            sc.settings.figdir = str(root_dir)

        adata_copy = adata.copy()
        sc.pp.normalize_total(adata_copy)
        sc.pp.log1p(adata_copy)
        sc.pp.pca(adata_copy)
        sc.pp.scale(adata_copy)
        sc.pp.neighbors(adata_copy)
        sc.tl.umap(adata_copy)

        sc.set_figure_params(dpi=self.dpi, figsize=self.figsize)

        if save is not None:
            sc.pl.umap(adata_copy, color=color, title=title, save=save, show=show)
        else:
            sc.pl.umap(adata_copy, color=color, title=title, show=show)

    # def preprocess_and_plot(
    #     self,
    #     adata: ad.AnnData,
    #     color: list[str] | str | None = None,
    #     title: str | None = None,
    #     save: str | None = None,
    #     show: bool = False,
    #     root_dir: Path | str | None = None,
    # ) -> ad.AnnData:
    #     """
    #     Preprocess data and optionally plot UMAP.

    #     This is a convenience method that combines preprocess() and plot_umap().
    #     If color is None, only preprocessing is performed without plotting.

    #     Args:
    #         adata: AnnData object to preprocess and plot. Should contain raw count data in adata.X.
    #         color: Column name(s) from adata.obs to color by. Can be a single string or list of strings.
    #             If None, no plot is generated and only preprocessing is performed.
    #         title: Plot title. If None, scanpy will use default title.
    #         save: Filename to save plot. If None, plot is not saved.
    #         show: Whether to display the plot interactively. If False, plot is only saved or not shown.
    #         root_dir: Directory to save plots. Only used if save is not None.

    #     Returns:
    #         Preprocessed AnnData object with:
    #         - adata.layers["raw_counts"]: Original raw counts
    #         - adata.X: Normalized, log-transformed, scaled data
    #         - adata.obsm["X_umap"]: UMAP coordinates
    #         - Other preprocessing results (see preprocess() documentation)

    #     Raises:
    #         KeyError: If color column(s) are not found in adata.obs (only if color is not None)
    #     """
    #     adata = self.preprocess(adata)

    #     if color is not None:
    #         self.plot_umap(adata, color=color, title=title, save=save, show=show, root_dir=root_dir)

    #     return adata


class MultiDatmpPreprocessor(DatmpPreprocessor):
    """
    Preprocessor for multiple Datmp objects (e.g., primary and metastasis datasets).

    Extends DatmpPreprocessor to handle preprocessing and visualization of
    multiple related datasets simultaneously.
    """

    def preprocess_multiple(
        self,
        *adatasets: ad.AnnData,
    ) -> tuple[ad.AnnData, ...]:
        """
        Preprocess multiple Datmp objects independently.

        Each AnnData object is preprocessed using the same preprocessing parameters
        (n_highly_variable_genes, max_value, target_sum) defined in the instance.

        Args:
            *adatasets: Variable number of Datmp objects to preprocess.
                Each should contain raw count data in adata.X.

        Returns:
            Tuple of preprocessed Datmp objects, in the same order as input.
            Each preprocessed Datmp has the same structure as returned by preprocess().

        Raises:
            ValueError: If no Datmp objects are provided
        """
        if not adatasets:
            raise ValueError("At least one AnnData object must be provided")
        return tuple(self.preprocess(adata) for adata in adatasets)

    def preprocess_and_plot_multiple(
        self,
        output_dir: Path | str,
        combined_adata: ad.AnnData,
        metastasis_adata: ad.AnnData | None = None,
        primary_adata: ad.AnnData | None = None,
        save_plots: bool = True,
    ) -> tuple[ad.AnnData, ad.AnnData | None, ad.AnnData | None]:
        """
        Preprocess multiple Datmp objects and generate UMAP visualizations.

        This method preprocesses the combined dataset and optionally the metastasis and primary
        datasets separately, then generates multiple UMAP plots:
        1. Combined UMAP colored by dataset and organ
        2. Combined UMAP colored by primary vs organs
        3. Metastasis UMAP colored by organ (if metastasis_adata is provided)
        4. Primary UMAP colored by organ (if primary_adata is provided)

        Args:
            output_dir: Directory to save plots. Will be created if it doesn't exist.
            combined_adata: Combined Datmp object containing all data. Must have 'dataset' and
                'organ' columns in obs.
            metastasis_adata: Metastasis Datmp object (optional). If provided, will be preprocessed
                separately and a metastasis-specific UMAP will be generated.
            primary_adata: Primary Datmp object (optional). If provided, will be preprocessed
                separately and a primary-specific UMAP will be generated.
            save_plots: Whether to save plots to files. If True, plots are saved to output_dir.
                If False, plots are only displayed (if show=True) or not shown.

        Returns:
            Tuple of (preprocessed_combined_adata, preprocessed_metastasis_adata, preprocessed_primary_adata):
            - preprocessed_combined_adata: Preprocessed combined AnnData object
            - preprocessed_metastasis_adata: Preprocessed metastasis AnnData object, or None if not provided
            - preprocessed_primary_adata: Preprocessed primary AnnData object, or None if not provided

        Raises:
            KeyError: If 'dataset' or 'organ' columns are missing from combined_adata.obs
        """
        # Input validation
        if "dataset" not in combined_adata.obs.columns:
            raise KeyError("'dataset' column is required in combined_adata.obs")
        if "organ" not in combined_adata.obs.columns:
            raise KeyError("'organ' column is required in combined_adata.obs")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        sc.set_figure_params(dpi=self.dpi, figsize=self.figsize)
        if save_plots:
            sc.settings.figdir = str(output_dir)

        # Preprocess all datasets
        combined_adata = self.preprocess(combined_adata)
        metastasis_adata = self.preprocess(metastasis_adata)
        primary_adata = self.preprocess(primary_adata)

        # Plot and save combined UMAP
        if save_plots:
            # self.plot_umap(
            #     combined_adata,
            #     color=["dataset", "organ"],
            #     save=UMAP_COMBINED_DATASET_ORGAN,
            #     show=False,
            #     root_dir=output_dir,
            # )
            # plot one by one
            self.plot_umap(
                combined_adata,
                color=["dataset"],
                save=DOMAIN_UMAP,
                show=False,
                root_dir=output_dir,
                title="Primary vs Metastasis",
            )
            self.plot_umap(
                combined_adata,
                color=["organ"],
                save=ORGAN_UMAP,
                show=False,
                root_dir=output_dir,
                title="Organs",
            )
        else:
            # self.plot_umap(combined_adata, color=["dataset", "organ"], show=False)
            # plot one by one
            self.plot_umap(combined_adata, color=["dataset"], show=False)
            self.plot_umap(combined_adata, color=["organ"], show=False)

        # Plot and save combined UMAP with primary vs organs
        def create_plot2_label(row):
            return "primary" if row["dataset"] == "primary" else row["organ"]

        combined_adata.obs["plot2_label"] = combined_adata.obs.apply(create_plot2_label, axis=1)
        if save_plots:
            self.plot_umap(
                combined_adata,
                color=["plot2_label"],
                title="Primary vs Metastasis Organs",
                save=PRIMARY_VS_ORGAN_UMAP,
                show=False,
                root_dir=output_dir,
            )
        else:
            self.plot_umap(
                combined_adata,
                color=["plot2_label"],
                title="Primary vs Metastasis Organs",
                show=False,
            )

        # Plot and save metastasis UMAP
        if metastasis_adata is not None:
            if save_plots:
                self.plot_umap(
                    metastasis_adata,
                    color=["organ"],
                    title="Metastasis Organs",
                    save=METASTASIS_UMAP,
                    show=False,
                    root_dir=output_dir,
                )
            else:
                self.plot_umap(metastasis_adata, color=["organ"], title="Metastasis Organs", show=False)

        # Plot and save primary UMAP
        if primary_adata is not None:
            if save_plots:
                self.plot_umap(
                    primary_adata,
                    color=["organ"],
                    title="Primary Organs",
                    save=PRIMARY_UMAP,
                    show=False,
                    root_dir=output_dir,
                )
            else:
                self.plot_umap(primary_adata, color=["organ"], title="Primary Organs", show=False)

        return combined_adata, metastasis_adata, primary_adata


def preprocess(
    output_dir: Path | str,
    concat_adata: ad.AnnData,
    primary_adata: ad.AnnData,
    metastasis_adata: ad.AnnData,
    n_highly_variable_genes: int = 1200,
    batch_key: str | None = "batch",
    special_token_map: dict[str, str] | None = None,
    **kwargs,
) -> None:
    """
    Preprocess data and save AnnData.

    This function performs the following steps:
    1. Save vocabulary to {output_dir}/{VOCAB_PATH}
    2. Preprocess data with scanpy (normalization, PCA, UMAP)
    3. Save preprocessed AnnData to {output_dir}/{PRIMARY_METASTASIS_H5AD}
    4. Generate configuration file and save to {output_dir}/{INFO_CONFIG}
    5. Generate UMAP plots and save to output_dir

    Args:
        output_dir: Output directory where all files will be saved. Must be a Path object.
            Will be created if it doesn't exist.
        concat_adata: Combined AnnData object containing all cells. Must have 'dataset' and
            'organ' columns in obs. Should contain raw count data in adata.X.
        primary_adata: Primary AnnData object. Used for generating primary-specific UMAP plots.
        metastasis_adata: Metastasis AnnData object. Used for generating metastasis-specific UMAP plots.
        n_highly_variable_genes: Number of highly variable genes. Must be positive.
            Used for config file seq_len. Note: HVG filtering is not performed in preprocessing.
        special_token_map: Map of special token names to their values. If None, uses default:
            {
                "cls": "[CLS]",
                "sep": "[SEP]",
                "pad": "[PAD]",
                "mask": "[MASK]",
                "unk": "[UNK]",
            }
        batch_key: Key in concat_adata.obs for batch information. Used for batch-aware HVG selection.

    Returns:
        None. All outputs are saved to files in output_dir:
        - {VOCAB_PATH}: Vocabulary list
        - {PRIMARY_METASTASIS_H5AD}: Preprocessed AnnData in H5AD format
        - {INFO_CONFIG}: Configuration file
        - UMAP plot files

    Raises:
        TypeError: If output_dir is not a Path object.
        ValueError: If n_highly_variable_genes is not positive.
        KeyError: If required columns ('dataset', 'organ') are missing from concat_adata.obs.
    """
    if not isinstance(output_dir, Path):
        output_dir = Path(output_dir)
    if n_highly_variable_genes <= 0:
        raise ValueError(f"n_highly_variable_genes must be positive, got {n_highly_variable_genes}")
    if "dataset" not in concat_adata.obs.columns:
        raise KeyError("'dataset' column is required in concat_adata.obs")
    if "organ" not in concat_adata.obs.columns:
        raise KeyError("'organ' column is required in concat_adata.obs")

    output_dir.mkdir(parents=True, exist_ok=True)
    vocab_file_path = output_dir / VOCAB_PATH
    h5ad_output_path = output_dir / PRIMARY_METASTASIS_H5AD
    config_output_path = output_dir / INFO_CONFIG

    # 1. Save vocabulary (final filtered genes will be saved after preprocessing)
    gene_vocab = concat_adata.var_names.tolist()
    if special_token_map is None:
        special_token_map = {
            "cls": "[CLS]",
            "sep": "[SEP]",
            "pad": "[PAD]",
            "mask": "[MASK]",
            "unk": "[UNK]",
        }

    all_tokens = list(special_token_map.values()) + gene_vocab
    logger.info(f"Saving vocabulary to {vocab_file_path} (length: {len(all_tokens)})")
    with open(vocab_file_path, "w") as f:
        for token in all_tokens:
            f.write(f"{token}\n")

    # 2. Preprocess data
    logger.info("Preprocessing data with scanpy...")

    preprocessor = MultiDatmpPreprocessor(
        subset_hvg=n_highly_variable_genes,
        batch_key=batch_key,
        **kwargs,
    )

    concat_adata, metastasis_adata, primary_adata = preprocessor.preprocess_and_plot_multiple(
        output_dir=output_dir,
        combined_adata=concat_adata,
        metastasis_adata=metastasis_adata,
        primary_adata=primary_adata,
        save_plots=True,
    )
    logger.info("Preprocessing completed.")
    logger.info(f"UMAP plots saved to: {output_dir}")

    # 3. Save AnnData
    logger.info(f"Saving AnnData to {h5ad_output_path}...")
    concat_adata.write_h5ad(h5ad_output_path)
    logger.info(f"AnnData saved to: {h5ad_output_path}")

    # 4. Generate config file
    class_map = {organ: idx for idx, organ in enumerate(sorted(concat_adata.obs["organ"].unique()))}

    logger.info(f"Saving configuration file to {config_output_path}...")
    with open(config_output_path, "w") as f:
        json.dump(
            {
                "class_map": class_map,
                "seq_len": n_highly_variable_genes,
                "num_classes": len(class_map),
                "vocab_size": len(all_tokens),
                "special_tokens": special_token_map,
                "raw_data_path": kwargs.get("raw_data_path", None),
            },
            f,
            indent=2,
        )
    logger.info(f"Configuration file saved to: {config_output_path}")
