#!/usr/bin/env python3
"""
Test all encoder modules from tomic/model/encoder_decoder.

This script tests all encoder types to ensure they can process input data correctly:
- MLPEncoder: Vector-encoded data (floating-point vectors)
- PatchTransformerEncoder: Patch-based transformer encoder
- NameTransformerEncoder: Name-based transformer encoder (token IDs)
- ExpressionTransformerEncoder: Expression-based transformer encoder
- DualTransformerEncoder: Dual cross-attention encoder for name and expression

Usage:
    python expertments/test_encoder/test_encoders.py
    python expertments/test_encoder/test_encoders.py --encoder_type mlp
    python expertments/test_encoder/test_encoders.py --encoder_type all
"""

import sys
from pathlib import Path

sys.path.append("/your/path/to/TOMIC")

import torch
from tomic import get_logger
from tomic.model.encoder_decoder.dual import DualTransformerEncoder
from tomic.model.encoder_decoder.expr import ExpressionTransformerEncoder
from tomic.model.encoder_decoder.mlp import MLPEncoder
from tomic.model.encoder_decoder.name import NameTransformerEncoder
from tomic.model.encoder_decoder.patch import PatchTransformerEncoder

logger = get_logger("test_encoder")

ENCODER_DESCRIPTIONS = {
    "mlp": "MLP Encoder",
    "patch": "Patch-based Transformer Encoder",
    "name": "Name-based Transformer Encoder",
    "expr": "Expression-based Transformer Encoder",
    "dual": "Dual Transformer Encoder",
}


def test_mlp_encoder(batch_size: int = 4, input_dim: int = 400, hidden_dims: list[int] = [128, 64]) -> bool:
    """Test MLPEncoder."""
    logger.info("\n" + "-" * 80)
    logger.info("Testing MLPEncoder")
    logger.info("-" * 80)

    try:
        encoder = MLPEncoder(input_dim=input_dim, hidden_dims=hidden_dims)
        x = torch.randn(batch_size, input_dim)
        logger.info(f"Input shape: {x.shape}")

        encoder.eval()
        with torch.no_grad():
            output = encoder(x)

        expected_shape = (batch_size, hidden_dims[-1])
        logger.info(f"Output shape: {output.shape}, Expected: {expected_shape}")

        assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"

        logger.info("✓ MLPEncoder test passed")
        return True
    except Exception as e:
        logger.error(f"✗ MLPEncoder test failed: {e}", exc_info=True)
        return False


def test_patch_encoder(
    batch_size: int = 4, seq_len: int = 400, hidden_size: int = 64, num_heads: int = 2, num_layers: int = 2
) -> bool:
    """Test PatchTransformerEncoder."""
    logger.info("\n" + "-" * 80)
    logger.info("Testing PatchTransformerEncoder")
    logger.info("-" * 80)

    try:
        if seq_len % hidden_size != 0:
            seq_len = (seq_len // hidden_size) * hidden_size
            logger.info(f"Adjusted seq_len to {seq_len}")

        encoder = PatchTransformerEncoder(
            seq_len=seq_len, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers
        )
        x = torch.randn(batch_size, seq_len)
        logger.info(f"Input shape: {x.shape}")

        encoder.eval()
        with torch.no_grad():
            output = encoder(x)

        num_patches = seq_len // hidden_size
        expected_shape = (batch_size, num_patches + 2, hidden_size)
        logger.info(f"Output shape: {output.shape}, Expected: {expected_shape}")

        assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"

        logger.info("✓ PatchTransformerEncoder test passed")
        return True
    except Exception as e:
        logger.error(f"✗ PatchTransformerEncoder test failed: {e}", exc_info=True)
        return False


def test_name_encoder(
    batch_size: int = 4,
    seq_len: int = 400,
    vocab_size: int = 1200,
    hidden_size: int = 64,
    num_heads: int = 2,
    num_layers: int = 2,
) -> bool:
    """Test NameTransformerEncoder."""
    logger.info("\n" + "-" * 80)
    logger.info("Testing NameTransformerEncoder")
    logger.info("-" * 80)

    try:
        # NameTransformerEncoder uses seq_len parameter as vocab_size for embedding
        # The actual input sequence length should match the positional encoding length
        encoder = NameTransformerEncoder(
            seq_len=vocab_size, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers
        )
        # Use vocab_size as the actual sequence length to match positional encoding
        actual_seq_len = vocab_size
        input_ids = torch.randint(0, vocab_size, (batch_size, actual_seq_len))
        logger.info(f"Input shape: {input_ids.shape}, dtype: {input_ids.dtype}")

        encoder.eval()
        with torch.no_grad():
            output = encoder(input_ids)

        expected_shape = (batch_size, actual_seq_len + 2, hidden_size)
        logger.info(f"Output shape: {output.shape}, Expected: {expected_shape}")

        assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"

        logger.info("✓ NameTransformerEncoder test passed")
        return True
    except Exception as e:
        logger.error(f"✗ NameTransformerEncoder test failed: {e}", exc_info=True)
        return False


def test_expr_encoder(
    batch_size: int = 4, seq_len: int = 400, hidden_size: int = 64, num_heads: int = 2, num_layers: int = 2
) -> bool:
    """Test ExpressionTransformerEncoder."""
    logger.info("\n" + "-" * 80)
    logger.info("Testing ExpressionTransformerEncoder")
    logger.info("-" * 80)

    try:
        encoder = ExpressionTransformerEncoder(
            seq_len=seq_len, hidden_size=hidden_size, num_heads=num_heads, num_layers=num_layers
        )
        expression = torch.rand(batch_size, seq_len)
        logger.info(f"Input shape: {expression.shape}, dtype: {expression.dtype}")

        encoder.eval()
        with torch.no_grad():
            output = encoder(expression)

        expected_shape = (batch_size, seq_len + 2, hidden_size)
        logger.info(f"Output shape: {output.shape}, Expected: {expected_shape}")

        assert output.shape == expected_shape, f"Shape mismatch: expected {expected_shape}, got {output.shape}"

        logger.info("✓ ExpressionTransformerEncoder test passed")
        return True
    except Exception as e:
        logger.error(f"✗ ExpressionTransformerEncoder test failed: {e}", exc_info=True)
        return False


def test_dual_encoder(
    batch_size: int = 4,
    seq_len: int = 400,
    vocab_size: int = 1200,
    hidden_size: int = 64,
    num_heads_cross_attn: int = 2,
    num_layers_cross_attn: int = 2,
    num_heads_encoder: int = 2,
    num_layers_encoder: int = 2,
    binning: int | None = None,
) -> bool:
    """Test DualTransformerEncoder."""
    logger.info("\n" + "-" * 80)
    logger.info("Testing DualTransformerEncoder")
    logger.info("-" * 80)

    try:
        # DualTransformerEncoder uses seq_len parameter as vocab_size for name embedding
        # The actual input sequence length should match the positional encoding length
        encoder = DualTransformerEncoder(
            seq_len=vocab_size,
            hidden_size=hidden_size,
            num_heads_cross_attn=num_heads_cross_attn,
            num_layers_cross_attn=num_layers_cross_attn,
            num_heads_encoder=num_heads_encoder,
            num_layers_encoder=num_layers_encoder,
            binning=binning,
        )

        # Use vocab_size as the actual sequence length to match positional encoding
        actual_seq_len = vocab_size
        name = torch.randint(0, vocab_size, (batch_size, actual_seq_len))
        if binning is not None:
            expr = torch.randint(0, binning, (batch_size, actual_seq_len))
        else:
            expr = torch.rand(batch_size, actual_seq_len)

        logger.info(f"Name shape: {name.shape}, dtype: {name.dtype}")
        logger.info(f"Expression shape: {expr.shape}, dtype: {expr.dtype}, binning: {binning}")

        encoder.eval()
        with torch.no_grad():
            aligned_emb, name_emb, expr_emb = encoder(name, expr)

        expected_aligned_shape = (batch_size, actual_seq_len + actual_seq_len + 2, hidden_size)
        expected_name_shape = (batch_size, actual_seq_len, hidden_size)
        expected_expr_shape = (batch_size, actual_seq_len, hidden_size)

        logger.info(f"Aligned output shape: {aligned_emb.shape}, Expected: {expected_aligned_shape}")
        logger.info(f"Name output shape: {name_emb.shape}, Expected: {expected_name_shape}")
        logger.info(f"Expression output shape: {expr_emb.shape}, Expected: {expected_expr_shape}")

        assert aligned_emb.shape == expected_aligned_shape, (
            f"Aligned shape mismatch: expected {expected_aligned_shape}, got {aligned_emb.shape}"
        )
        assert name_emb.shape == expected_name_shape, (
            f"Name shape mismatch: expected {expected_name_shape}, got {name_emb.shape}"
        )
        assert expr_emb.shape == expected_expr_shape, (
            f"Expression shape mismatch: expected {expected_expr_shape}, got {expr_emb.shape}"
        )

        logger.info("✓ DualTransformerEncoder test passed")
        return True
    except Exception as e:
        logger.error(f"✗ DualTransformerEncoder test failed: {e}", exc_info=True)
        return False


def test_encoder(encoder_type: str) -> bool:
    """Test a single encoder type."""
    logger.info("\n" + "=" * 80)
    logger.info(f"Testing {ENCODER_DESCRIPTIONS[encoder_type]} ({encoder_type})")
    logger.info("=" * 80)

    batch_size = 4
    seq_len = 400
    hidden_size = 64
    vocab_size = 1200

    if encoder_type == "mlp":
        return test_mlp_encoder(batch_size=batch_size, input_dim=seq_len, hidden_dims=[128, 64])
    elif encoder_type == "patch":
        return test_patch_encoder(batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size)
    elif encoder_type == "name":
        return test_name_encoder(batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, hidden_size=hidden_size)
    elif encoder_type == "expr":
        return test_expr_encoder(batch_size=batch_size, seq_len=seq_len, hidden_size=hidden_size)
    elif encoder_type == "dual":
        return test_dual_encoder(
            batch_size=batch_size, seq_len=seq_len, vocab_size=vocab_size, hidden_size=hidden_size, binning=51
        )
    else:
        logger.error(f"Unknown encoder_type: {encoder_type}")
        return False


def main():
    """Main function to test encoder modules."""
    import argparse

    parser = argparse.ArgumentParser(description="Test encoder modules")
    parser.add_argument(
        "--encoder_type",
        type=str,
        default="all",
        choices=["all", "mlp", "patch", "name", "expr", "dual"],
        help="Encoder type to test (default: all)",
    )

    args = parser.parse_args()

    if args.encoder_type == "all":
        encoder_types = ["mlp", "patch", "name", "expr", "dual"]
    else:
        encoder_types = [args.encoder_type]

    logger.info("=" * 80)
    logger.info("Encoder Modules Test Suite")
    logger.info("=" * 80)
    logger.info(f"Encoders to test: {encoder_types}")

    results = {}
    for encoder_type in encoder_types:
        success = test_encoder(encoder_type)
        results[encoder_type] = success

    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)
    for encoder_type, success in results.items():
        status = "✓ PASSED" if success else "✗ FAILED"
        logger.info(f"{ENCODER_DESCRIPTIONS[encoder_type]} ({encoder_type}): {status}")

    total = len(results)
    passed = sum(results.values())
    failed = total - passed

    logger.info(f"\nTotal: {total}, Passed: {passed}, Failed: {failed}")

    if failed > 0:
        logger.warning(f"\n{failed} test(s) failed. Please check the logs above for details.")
        sys.exit(1)
    else:
        logger.info("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
