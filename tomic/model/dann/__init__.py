"""
Domain Adversarial Neural Networks (DANN) Lightning Modules.

This package provides separate Lightning modules for different model types:
- Patch-based Transformer
- MLP encoder
- Token-based Transformer (name)
- Expression-based Transformer (expr)
"""

# Configuration classes
from .dual import DualTransformerModelConfig
from .dual import LightningModule as LightningModuleDual
from .expr import ExprModelConfig
from .expr import LightningModule as LightningModuleExpr
from .mlp import LightningModule as LightningModuleMLP
from .mlp import MLPModelConfig
from .name import LightningModule as LightningModuleName
from .name import NameModelConfig
from .patch import LightningModule as LightningModulePatch
from .patch import PatchModelConfig


def get_lightning_module(model_type: str):
    """
    Get the appropriate Lightning module based on model_type.

    Args:
        model_type: Model type ("patch", "mlp", "name", "expr", "dual")

    Returns:
        Lightning module class
    """
    module_map = {
        "patch": LightningModulePatch,
        "mlp": LightningModuleMLP,
        "name": LightningModuleName,
        "expr": LightningModuleExpr,
        "dual": LightningModuleDual,
    }
    if model_type not in module_map:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of {list(module_map.keys())}")
    return module_map[model_type]


__all__ = [
    # Model configurations
    "PatchModelConfig",
    "MLPModelConfig",
    "NameModelConfig",
    "ExprModelConfig",
    "DualTransformerModelConfig",
    # Lightning modules
    "LightningModulePatch",
    "LightningModuleMLP",
    "LightningModuleName",
    "LightningModuleExpr",
    "LightningModuleDual",
    "get_lightning_module",
]
