import anndata as ad
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from .abc import DatasetABC, DomainDataModuleABC
from .dataconfig import DatmpDataConfig
from .preprocessing import PRIMARY_METASTASIS_H5AD
from .scgpt_preprocess import _digitize, _get_obs_rep


class OneDomainDataset(Dataset):
    def __init__(
        self,
        adata: ad.AnnData,
    ):
        # 1. load data
        """
        batch = {
            "s_gene_ids": list[str],
            "s_expr_ids": list[int],
            "s_expression": list[float],
            "s_label": int,
            "t_gene_ids": list[str],
            "t_expr_ids": list[int],
            "t_expression": list[float],
            "t_label": int,
        }

        """

        # load expression data
        expr = adata.layers["X_log1p"]
        expr = expr / expr.max(axis=1, keepdims=True)

        # sorted gene names by expression from high to low
        # Use np.flip to avoid negative stride issues with PyTorch
        self.ids = np.flip(np.argsort(expr, axis=1), axis=1).copy()

        self.labels = adata.obs["label"].values
        self.cell_ids = adata.obs_names.values

        # Initialize binned data (None if binning was not performed)
        self.binned = adata.layers["X_binned"] if "X_binned" in adata.layers else None

        if self.binned is not None:
            self.expr_binned = adata.layers["X_binned"]
            self.expr = None
        else:
            self.expr_binned = None
            self.expr = expr

    def __len__(self):
        return len(self.cell_ids)

    def __getitem__(self, idx):
        return_dict = {
            "cell_id": self.cell_ids[idx],
            "gene_ids": self.ids[idx],
            "expr_ids": self.expr_binned[idx] if self.expr_binned is not None else None,
            "expr": self.expr[idx] if self.expr is not None else None,
            "label": self.labels[idx],
        }
        # Remove None values to avoid DataLoader collate issues
        return_dict = {k: v for k, v in return_dict.items() if v is not None}
        return return_dict


class DatasetDatmp(DatasetABC):
    def __init__(
        self,
        anndata_source: ad.AnnData,
        anndata_target: ad.AnnData,
    ):
        # 1. load data
        """
        batch = {
            "s_gene_ids": list[str],
            "s_expr_ids": list[int],
            "s_expression": list[float],
            "s_label": int,
            "t_gene_ids": list[str],
            "t_expr_ids": list[int],
            "t_expression": list[float],
            "t_label": int,
        }

        """
        # Then create indices
        self.source_indices = np.arange(anndata_source.shape[0])
        self.target_indices = np.arange(anndata_target.shape[0])

        # Load expression data first
        source_expr = anndata_source.layers["X_log1p"]
        target_expr = anndata_target.layers["X_log1p"]
        source_expr = source_expr / source_expr.max(axis=1, keepdims=True)
        target_expr = target_expr / target_expr.max(axis=1, keepdims=True)

        # sorted gene names by expression from high to low
        # Use np.flip to avoid negative stride issues with PyTorch
        self.source_ids = np.flip(np.argsort(source_expr, axis=1), axis=1).copy()
        self.target_ids = np.flip(np.argsort(target_expr, axis=1), axis=1).copy()

        self.source_labels = anndata_source.obs["label"].values
        self.target_labels = anndata_target.obs["label"].values

        # Initialize binned data (None if binning was not performed)
        self.is_binned = "X_binned" in anndata_source.layers and "X_binned" in anndata_target.layers

        if self.is_binned:
            self.source_binned = anndata_source.layers["X_binned"]
            self.target_binned = anndata_target.layers["X_binned"]
            self.source_expr = None
            self.target_expr = None
        else:
            self.source_binned = None
            self.target_binned = None
            self.source_expr = source_expr
            self.target_expr = target_expr

        self.source_cell_ids = anndata_source.obs_names.values
        self.target_cell_ids = anndata_target.obs_names.values

    def __len__(self):
        return min(len(self.source_indices), len(self.target_indices))

    def __getitem__(self, idx):
        source_idx = self.source_indices[idx]
        target_idx = self.target_indices[idx]

        return_dict = {
            "s_cell_id": self.source_cell_ids[source_idx],
            "t_cell_id": self.target_cell_ids[target_idx],
            "s_gene_ids": self.source_ids[source_idx],
            "s_expr_ids": self.source_binned[source_idx] if self.is_binned else None,
            "s_expr": self.source_expr[source_idx] if not self.is_binned else None,
            "s_label": self.source_labels[source_idx],
            "t_gene_ids": self.target_ids[target_idx],
            "t_expr_ids": self.target_binned[target_idx] if self.is_binned else None,
            "t_expr": self.target_expr[target_idx] if not self.is_binned else None,
            "t_label": self.target_labels[target_idx],
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}
        return return_dict


class DomainDataModuleDatmp(DomainDataModuleABC):
    def __init__(
        self,
        data_config: DatmpDataConfig,
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
        """
        DataModule for Datmp format (gene name sequences with expression values).

        Args:
            train_batch_size: Batch size for training
            test_batch_size: Batch size for testing/validation
            num_workers: Number of data loading workers
            test_size: Fraction of data to use for testing
            random_state: Random seed for data splitting
            root_data_path: Path to data directory containing synthetic_primary_metastasis2.h5ad
            class_map: Mapping from organ names to class labels. If None, will be loaded from config file.
            binning: Number of bins for expression value discretization (None for continuous).
                    If None, will be loaded from config file.
        """
        super().__init__()

        self.data_config = data_config

        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        # default to 0 workers to avoid many open files; user can raise if safe
        self.num_workers = num_workers

        self.test_size = test_size
        self.random_state = random_state

        # Key for storing binned data in layers
        self.result_binned_key = "X_binned"

        # Expose data_config attributes for easy access (matching DatmpDataConfig structure)
        self.class_map = self.data_config.class_map
        self.num_classes = self.data_config.num_classes
        self.root_data_path = self.data_config.root_data_path

    def do_binning(self, adata: ad.AnnData, key_to_process: str | None = None):
        # logger.info("Binning data ...")
        n_bins = self.data_config.binning  # NOTE: the first bin is always a spectial for zero

        # Skip binning if binning is None
        if n_bins is None:
            return

        binned_rows = []
        bin_edges = []
        layer_data = _get_obs_rep(adata, layer=key_to_process)
        layer_data = layer_data.toarray() if issparse(layer_data) else layer_data
        if layer_data.min() < 0:
            raise ValueError(f"Assuming non-negative data, but got min value {layer_data.min()}.")
        for row in layer_data:
            if row.max() == 0:
                # logger.warning(
                #     "The input data contains all zero rows. Please make sure "
                #     "this is expected. You can use the `filter_cell_by_counts` "
                #     "arg to filter out all zero rows."
                # )
                binned_rows.append(np.zeros_like(row, dtype=np.int64))
                bin_edges.append(np.array([0] * n_bins))
                continue
            non_zero_ids = row.nonzero()
            non_zero_row = row[non_zero_ids]
            bins = np.quantile(non_zero_row, np.linspace(0, 1, n_bins - 1))
            # bins = np.sort(np.unique(bins))
            # NOTE: comment this line for now, since this will make the each category
            # has different relative meaning across datasets
            non_zero_digits = _digitize(non_zero_row, bins)
            assert non_zero_digits.min() >= 1
            assert non_zero_digits.max() <= n_bins - 1
            binned_row = np.zeros_like(row, dtype=np.int64)
            binned_row[non_zero_ids] = non_zero_digits
            binned_rows.append(binned_row)
            bin_edges.append(np.concatenate([[0], bins]))
        adata.layers[self.result_binned_key] = np.stack(binned_rows)
        adata.obsm["bin_edges"] = np.stack(bin_edges)

    def setup(self, stage=None):
        # Load AnnData
        adata = ad.read_h5ad(self.data_config.root_data_path / PRIMARY_METASTASIS_H5AD)

        # reomve nan-highly_variable genes
        adata = adata[:, adata.var["highly_variable"]].copy()

        # Ensure observation names are unique
        adata.obs_names_make_unique()

        # Add label column using class_map from config
        adata.obs["label"] = adata.obs["organ"].apply(lambda x: self.class_map[x])

        # Binning
        self.do_binning(adata, key_to_process=None)

        target = adata[adata.obs["dataset"] == "primary"]
        source = adata[adata.obs["dataset"] == "metastasis"]

        target_train, target_test = train_test_split(
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target.obs["organ"],
        )

        source_train, source_test = train_test_split(
            source,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=source.obs["organ"],
        )

        self.train_dataset = DatasetDatmp(source_train, target_train)
        self.val_dataset = DatasetDatmp(source_test, target_test)
        self.test_dataset = DatasetDatmp(source_test, target_test)

    def set4extract_embedding(self):
        adata = ad.read_h5ad(self.data_config.root_data_path / PRIMARY_METASTASIS_H5AD)

        # reomve nan-highly_variable genes
        adata = adata[:, adata.var["highly_variable"]].copy()

        # Ensure observation names are unique
        adata.obs_names_make_unique()

        # Add label column using class_map from config
        adata.obs["label"] = adata.obs["organ"].apply(lambda x: self.class_map[x])

        # Binning
        self.do_binning(adata, key_to_process=None)

        target = adata[adata.obs["dataset"] == "primary"]
        source = adata[adata.obs["dataset"] == "metastasis"]

        target_train, target_test = train_test_split(
            target,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=target.obs["organ"],
        )

        source_train, source_test = train_test_split(
            source,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=source.obs["organ"],
        )

        self.source_train_dataloader = DataLoader(
            OneDomainDataset(source_train),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.target_train_dataloader = DataLoader(
            OneDomainDataset(target_train),
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.source_test_dataloader = DataLoader(
            OneDomainDataset(source_test),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
        self.target_test_dataloader = DataLoader(
            OneDomainDataset(target_test),
            batch_size=self.test_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
