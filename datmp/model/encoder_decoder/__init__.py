"""
Backward compatibility module for model.py.

This module imports classes from model_patch.py and provides backward compatibility aliases.
For new code, please import directly from model_patch.py.
"""

from .dual import DualTransformerDecoder, DualTransformerEmbedding, DualTransformerEncoder
from .expr import ExpressionDecoder, ExpressionEmbedding, ExpressionTransformerEncoder
from .mlp import MLPDecoder, MLPEncoder
from .name import NameDecoder, NameEmbedding, NameTransformerEncoder
from .patch import PatchDecoder, PatchEmbedding, PatchTransformerEncoder

__all__ = [
    "DualTransformerDecoder",
    "DualTransformerEmbedding",
    "DualTransformerEncoder",
    "ExpressionDecoder",
    "ExpressionEmbedding",
    "ExpressionTransformerEncoder",
    "FusionDecoder",
    "FusionEmbedding",
    "FusionTransformerEncoder",
    "MLPDecoder",
    "MLPEncoder",
    "PatchDecoder",
    "PatchEmbedding",
    "PatchTransformerEncoder",
    "NameDecoder",
    "NameEmbedding",
    "NameTransformerEncoder",
]
