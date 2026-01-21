"""
Domain Separation Networks (DSN) Lightning Modules.

This package provides separate Lightning modules for different model types:
- Patch-based Transformer
- MLP encoder
- Token-based Transformer (name)
- Expression-based Transformer (expr)
"""

# Configuration classes

from .dual import DualTransformerModelConfig as DualTransformerModelConfig4DSN
from .dual import LightningModule as DualTransformerModel4DSN
from .expr import ExprModelConfig as ExprTransformerModelConfig4DSN
from .expr import LightningModule as ExprTransformerModel4DSN
from .mlp import LightningModule as MLPModel4DSN
from .mlp import MLPModelConfig as MLPModelConfig4DSN
from .name import LightningModule as NameTransformerModel4DSN
from .name import NameModelConfig as NameTransformerModelConfig4DSN
from .patch import LightningModule as PatchTransformerModel4DSN
from .patch import PatchModelConfig as PatchTransformerModelConfig4DSN

__all__ = [
    # config classes
    "DualTransformerModelConfig4DSN",
    "ExprTransformerModelConfig4DSN",
    "MLPModelConfig4DSN",
    "NameTransformerModelConfig4DSN",
    "PatchTransformerModelConfig4DSN",
    # lightning modules
    "DualTransformerModel4DSN",
    "ExprTransformerModel4DSN",
    "MLPModel4DSN",
    "NameTransformerModel4DSN",
    "PatchTransformerModel4DSN",
]
