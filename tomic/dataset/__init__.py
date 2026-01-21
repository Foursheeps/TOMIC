"""
Dataset module for RNA data.

This module provides base dataset classes and implementations for different data formats:
- AnnData format (gene expression vectors)
- Token format (gene name sequences)
- Fusion format (gene names + expression)
"""

from .dataconfig import TomicDataConfig
from .dataset4common import DomainDataModuleCommon
from .dataset4da import DomainDataModuleTomic
from .preprocessing import (
    GET_GEN_FLAG,
    INFO_CONFIG,
    PRIMARY_METASTASIS_H5AD,
    VOCAB_PATH,
    MultiTomicPreprocessor,
    preprocess,
)

# Create class_map aliases for backward compatibility

__all__ = [
    "TomicDataConfig",
    "DomainDataModuleCommon",
    "DomainDataModuleTomic",
    "GET_GEN_FLAG",
    "INFO_CONFIG",
    "PRIMARY_METASTASIS_H5AD",
    "VOCAB_PATH",
    "MultiTomicPreprocessor",
    "preprocess",
]
