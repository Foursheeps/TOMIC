from typing import Literal

import anndata as ad
import numpy as np
import pytorch_lightning as pl
from scipy.sparse import issparse
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from .abc import DatasetABC
from .dataconfig import TomicDataConfig
from .preprocessing import PRIMARY_METASTASIS_H5AD
from .scgpt_preprocess import _digitize, _get_obs_rep


class Datasetcommon(DatasetABC):
    """
    Dataset for standard supervised learning using scGPT format.

    Returns single sample per batch (no domain separation):
    {
        "gene_ids": np.ndarray,  # sorted gene indices by expression
        "expr_ids": np.ndarray,  # binned expression values
        "expr": np.ndarray,      # continuous normalized expression values
        "label": int,            # class label
    }
    """

    def __init__(self, anndata: ad.AnnData):
        self.indices = np.arange(anndata.shape[0])
        self.cell_ids = anndata.obs_names.values

        # Load expression data first
        expr = anndata.layers["X_log1p"]
        expr = expr / expr.max(axis=1, keepdims=True)

        # sorted gene names by expression from high to low
        # Use np.flip to avoid negative stride issues with PyTorch
        self.gene_ids = np.flip(np.argsort(expr, axis=1), axis=1).copy()

        self.labels = anndata.obs["label"].values

        # Initialize binned data (None if binning was not performed)
        self.is_binned = "X_binned" in anndata.layers

        if self.is_binned:
            self.binned = anndata.layers["X_binned"]
            self.expr = None
        else:
            self.binned = None
            self.expr = expr

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        sample_idx = self.indices[idx]

        return_dict = {
            "cell_id": self.cell_ids[sample_idx],
            "gene_ids": self.gene_ids[sample_idx],
            "expr_ids": self.binned[sample_idx] if self.is_binned else None,
            "expr": self.expr[sample_idx] if not self.is_binned else None,
            "label": self.labels[sample_idx],
        }
        return_dict = {k: v for k, v in return_dict.items() if v is not None}
        return return_dict


class DomainDataModuleCommon(pl.LightningDataModule):
    def __init__(
        self,
        data_config: TomicDataConfig,
        train: Literal["source", "target", "both"] = "source",
        train_batch_size: int = 32,
        test_batch_size: int = 32,
        num_workers: int = 0,
        test_size: float = 0.2,
        random_state: int = 42,
    ):
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

        # Expose data_config attributes for easy access (matching TomicDataConfig structure)
        self.class_map = self.data_config.class_map
        self.num_classes = self.data_config.num_classes
        self.root_data_path = self.data_config.root_data_path

        if train not in ["source", "target", "both"]:
            raise ValueError(f"Invalid train: {train}. Must be 'source', 'target', or 'both'")
        self.train = train

        if not self.class_map:
            raise ValueError("class_map must be provided in data_config and cannot be empty")

        # Allow binning=None for models that don't need binning (e.g., patch/expr/mlp)
        # But if binning is set, it must be a positive integer
        if self.data_config.binning is not None and self.data_config.binning <= 0:
            raise ValueError("binning must be None or a positive integer")

        if not (0 < test_size < 1):
            raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

        self.train_dataset: Datasetcommon | None = None
        self.val_dataset: Datasetcommon | None = None
        self.test_dataset: Datasetcommon | None = None

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

        # Select training data based on train parameter
        if self.train == "source":
            train_adata = source_train
        elif self.train == "target":
            train_adata = target_train
        elif self.train == "both":
            # Concatenate source and target for training
            train_adata = ad.concat([source_train, target_train], join="outer")
        else:
            raise ValueError(f"Invalid train: {self.train}. Must be 'source', 'target', or 'both'")

        # Create datasets
        self.train_dataset = Datasetcommon(train_adata)
        self.val_dataset = Datasetcommon(source_test)
        self.test_dataset = Datasetcommon(target_test)

    def train_dataloader(self):
        if self.train_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        if self.val_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        if self.val_dataset is None or self.test_dataset is None:
            raise RuntimeError("Dataset not initialized. Call setup() first.")

        source_test_loader = DataLoader(
            self.val_dataset,  # val_dataset contains source_test
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        target_test_loader = DataLoader(
            self.test_dataset,  # test_dataset contains target_test
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return {"source_test": source_test_loader, "target_test": target_test_loader}
