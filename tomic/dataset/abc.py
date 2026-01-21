from abc import abstractmethod

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset


class RNADataset(Dataset):
    """
    Base dataset class for RNA data.
    Provides common functionality for source and target domain data.
    """

    def __init__(
        self,
    ):
        super().__init__()
        self.source_data = None
        self.source_indices = None

        self.target_data = None
        self.target_indices = None

    def __len__(self):
        return min(len(self.source_data), len(self.target_data))

    def shuffle_data(self):
        """Shuffle data indices at the beginning of each epoch."""
        np.random.shuffle(self.source_indices)
        np.random.shuffle(self.target_indices)


# Alias for backward compatibility and clearer naming
DatasetABC = RNADataset


class DomainDataModuleABC(pl.LightningDataModule):
    def __init__(
        self,
        train_batch_size=32,
        test_batch_size=32,
        num_workers=0,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        self.train_dataset: RNADataset = None
        self.val_dataset: RNADataset = None
        self.test_dataset: RNADataset = None

    @abstractmethod
    def setup(self, stage=None):
        raise NotImplementedError("Subclasses must implement this method")

    def train_dataloader(self):
        self.train_dataset.shuffle_data()
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
        )

    def val_dataloader(self):
        self.val_dataset.shuffle_data()
        return DataLoader(
            self.val_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        """
        Return multiple test dataloaders for evaluating on both train and test sets.

        Returns:
            dict: Dictionary of DataLoaders {"train": train_loader, "test": test_loader}
                - train_loader: DataLoader for training set evaluation (dataloader_idx=0)
                - test_loader: DataLoader for test set evaluation (dataloader_idx=1)

        Note:
            PyTorch Lightning will iterate over dictionary values in insertion order (Python 3.7+),
            so the order is preserved. The dataloader_idx will be 0 for "train" and 1 for "test".
        """
        # Return multiple datasets as a dictionary
        # Lightning will iterate over dictionary values in insertion order
        # Use dataloader_idx parameter in test_step to distinguish them
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,  # Don't shuffle for evaluation
            num_workers=self.num_workers,
            drop_last=False,  # Don't drop last for evaluation
        )
        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
        )
        return {"train": train_loader, "test": test_loader}
