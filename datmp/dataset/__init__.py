"""
Dataset module for RNA data.

This module provides base dataset classes and implementations for different data formats:
- AnnData format (gene expression vectors)
- Token format (gene name sequences)
- Fusion format (gene names + expression)
"""

from .dataconfig import DatmpDataConfig
from .dataset4common import DomainDataModuleCommon
from .dataset4da import DomainDataModuleDatmp
from .preprocessing import (
    GET_GEN_FLAG,
    INFO_CONFIG,
    PRIMARY_METASTASIS_H5AD,
    VOCAB_PATH,
    MultiDatmpPreprocessor,
    preprocess,
)

# Create class_map aliases for backward compatibility

__all__ = [
    "DatmpDataConfig",
    "DomainDataModuleCommon",
    "DomainDataModuleDatmp",
    "GET_GEN_FLAG",
    "INFO_CONFIG",
    "PRIMARY_METASTASIS_H5AD",
    "VOCAB_PATH",
    "MultiDatmpPreprocessor",
    "preprocess",
]
