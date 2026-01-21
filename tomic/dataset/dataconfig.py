"""
Configuration classes for DSN training scripts.

This module provides configuration dataclasses for data parameters.
Model architecture parameters are in individual training files.
Training hyperparameters are in train_config.py.
"""

import json
from pathlib import Path

"""
{
  "class_map": {
    "Liver": 0,
    "Lung": 1,
    "Met": 2
  },
  "seq_len": 1200,
  "num_classes": 3,
  "vocab_size": 31058,
  "special_tokens": {
    "cls": "[CLS]",
    "sep": "[SEP]",
    "pad": "[PAD]",
    "mask": "[MASK]",
    "unk": "[UNK]"
  },
  "raw_data_path": "/your/path/to/raw_data/GSE173958_RAW"
}
"""


class TomicDataConfig:
    """Data loading configuration parameters.

    Attributes:
        root_data_path: Root directory path for data files
        binning: Number of bins for expression value discretization. If None, uses continuous values.
        class_map: Class mapping dictionary from organ names to class labels
        seq_len: Sequence length (number of highly variable genes)
        num_classes: Number of classes (derived from class_map if not provided)
    """

    def __init__(
        self,
        root_data_path: Path | str | None = None,
        binning: int | None = None,
        class_map: dict[str, int] | None = None,
        num_classes: int | None = None,
        seq_len: int | None = None,
        **kwargs,
    ):
        # Convert root_data_path to Path if it's a string
        # Allow None for cases where it will be set from JSON file
        if root_data_path is not None:
            assert isinstance(root_data_path, Path | str), "root_data_path must be a Path or str"
            if isinstance(root_data_path, str):
                root_data_path = Path(root_data_path)

        self.root_data_path = root_data_path
        self.binning = binning
        self.class_map = class_map
        self.seq_len = seq_len
        self.num_classes = num_classes

    @staticmethod
    def from_json_or_kwargs(json_path: Path | str, **kwargs) -> "TomicDataConfig":
        """Load TomicDataConfig from JSON file or kwargs.
        If kwargs are provided, they will override the values in the JSON file.

        The JSON file should contain:
        - root_data_path: Path to data directory
        - binning: Optional binning parameter
        - class_map: Class mapping dictionary
        - seq_len: Sequence length

        Args:
            json_path: Path to JSON config file
            kwargs: Keyword arguments to override the values in the JSON file
        Returns:
            TomicDataConfig instance
        """

        data = dict()
        assert isinstance(json_path, Path | str), "json_path must be a Path or str"
        if isinstance(json_path, str):
            json_path = Path(json_path)
        data["root_data_path"] = json_path.parent
        with open(json_path) as f:
            data.update(json.load(f))

        data.update(kwargs)

        return TomicDataConfig(**data)

    def __repr__(self) -> str:
        return f"""
        TomicDataConfig(
            root_data_path={str(self.root_data_path)},
            binning={self.binning},
            class_map={self.class_map},
            num_classes={self.num_classes},
        )"""


if __name__ == "__main__":
    config_path = Path(
        "/your/path/to/TOMIC/expertments/data_process/GSE173958_processed/GSE173958_M1_1200/info_config.json"
    )
    data_config = TomicDataConfig.from_json_or_kwargs(config_path, binning=121)
    print(data_config)
