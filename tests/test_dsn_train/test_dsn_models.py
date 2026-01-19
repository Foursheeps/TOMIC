#!/usr/bin/env python3
"""
Test DSN models with synthetic data.

This script tests all DSN model types (name, patch, mlp, expr, scgpt) using
synthetic data to ensure they can be trained and tested correctly.

Usage:
    python expertments/test_dsn/test_dsn_models.py
    python expertments/test_dsn/test_dsn_models.py --model_type mlp
    python expertments/test_dsn/test_dsn_models.py --model_type all
"""

import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to Python path
sys.path.append("/your/path/to/TOMIC")

from datmp import get_logger
from datmp.dataset.dataconfig import DatmpDataConfig
from datmp.model.dsn import (
    DualTransformerModelConfig4DSN,
    ExprTransformerModelConfig4DSN,
    MLPModelConfig4DSN,
    NameTransformerModelConfig4DSN,
    PatchTransformerModelConfig4DSN,
)
from datmp.train.dsn.train import main as train_dsn_func
from datmp.train.dsn.train_config import TrainerConfig

# Use unified logger
logger = get_logger("test_dsn_models")

# Default synthetic data path
DEFAULT_DATA_PATH = Path("/your/path/to/TOMIC/expertments/data_process/synthetic_processed/synthetic_400")
DEFAULT_OUTPUT_PATH = Path("/your/path/to/outputs/outputs/test_dsn_models")

# Test log file path (in the test_dsn directory)
TEST_LOG_FILE = Path(__file__).parent / "test_results.json"

# Model type to config class mapping
MODEL_CONFIG_MAP = {
    "name": NameTransformerModelConfig4DSN,
    "patch": PatchTransformerModelConfig4DSN,
    "mlp": MLPModelConfig4DSN,
    "expr": ExprTransformerModelConfig4DSN,
    "dual": DualTransformerModelConfig4DSN,
}

# Model type descriptions
MODEL_DESCRIPTIONS = {
    "name": "Name-based Transformer DSN",
    "patch": "Patch-based Transformer DSN",
    "mlp": "MLP-based DSN",
    "expr": "Expression-based Transformer DSN",
    "dual": "Dual-based Transformer DSN",
}


def load_test_log() -> dict:
    """Load test results from log file.

    Returns:
        Dictionary with test results, empty dict if file doesn't exist
    """
    if TEST_LOG_FILE.exists():
        try:
            with open(TEST_LOG_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to load test log: {e}. Starting with empty log.")
            return {}
    return {}


def save_test_log(test_log: dict):
    """Save test results to log file.

    Args:
        test_log: Dictionary with test results
    """
    try:
        TEST_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TEST_LOG_FILE, "w") as f:
            json.dump(test_log, f, indent=2, default=str)
        logger.info(f"Test log saved to: {TEST_LOG_FILE}")
    except OSError as e:
        logger.error(f"Failed to save test log: {e}")


def get_test_key(model_type: str, devices: int, binning: int | None) -> str:
    """Generate a unique key for test log entry.

    Args:
        model_type: Model type
        devices: Number of devices (1 or 2)
        binning: Binning value (None or int)

    Returns:
        Unique key string
    """
    binning_str = "none" if binning is None else str(binning)
    return f"{model_type}_dev{devices}_bin{binning_str}"


def check_model_passed(model_type: str, devices: int, binning: int | None, test_log: dict) -> bool:
    """Check if a model has already passed the test.

    Args:
        model_type: Model type to check
        devices: Number of devices (1 or 2)
        binning: Binning value (None or int)
        test_log: Test log dictionary

    Returns:
        True if model has passed, False otherwise
    """
    test_key = get_test_key(model_type, devices, binning)
    if test_key not in test_log:
        return False

    model_result = test_log[test_key]
    if not isinstance(model_result, dict):
        return False

    # Check if test passed
    if model_result.get("status") == "passed":
        return True

    return False


def update_test_log(
    model_type: str,
    devices: int,
    binning: int | None,
    status: str,
    data_path: Path,
    output_path: Path,
    error_message: str | None = None,
):
    """Update test log with new result.

    Args:
        model_type: Model type that was tested
        devices: Number of devices used (1 or 2)
        binning: Binning value used (None or int)
        status: Test status ("passed" or "failed")
        data_path: Data path used
        output_path: Output path used
        error_message: Error message if test failed
    """
    test_log = load_test_log()
    test_key = get_test_key(model_type, devices, binning)

    test_log[test_key] = {
        "model_type": model_type,
        "devices": devices,
        "binning": binning,
        "status": status,
        "data_path": str(data_path),
        "output_path": str(output_path),
        "timestamp": datetime.now().isoformat(),
        "error_message": error_message,
    }

    save_test_log(test_log)


def test_dsn_models(model_type: str, devices: int = 1, binning: int | None = 50) -> tuple[bool, bool]:
    """Test DSN models.

    Args:
        model_type: Model type to test ("name", "patch", "mlp", "expr", "dual")
        devices: Number of devices to use (1 or 2, default: 1)
        binning: Number of bins for expression discretization (None or int, default: 50)

    Returns:
        Tuple of (success: bool, skipped: bool)
    """
    if model_type not in MODEL_CONFIG_MAP:
        raise ValueError(f"Unknown model_type: {model_type}. Must be one of {list(MODEL_CONFIG_MAP.keys())}")
    # Load data config
    config_path = DEFAULT_DATA_PATH / "info_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data_args = DatmpDataConfig.from_json_or_kwargs(config_path, binning=binning)
    data_args.root_data_path = DEFAULT_DATA_PATH

    # Create model config
    base_config = {
        "dropout": 0.1,
        "activation": "gelu",
    }
    if model_type == "mlp":
        model_config_dict = {
            **base_config,
            "hidden_dims": [32, 32, 32],
        }
    elif model_type == "dual":
        model_config_dict = {
            **base_config,
            "hidden_size": 40,
            "num_heads_cross_attn": 4,
            "num_heads_encoder": 4,
            "num_layers_cross_attn": 1,
            "num_layers_encoder": 1,
        }
    else:  # name, patch, expr
        model_config_dict = {
            **base_config,
            "hidden_size": 40,
            "num_heads": 4,
            "num_layers": 1,
        }
    model_args = MODEL_CONFIG_MAP[model_type](**model_config_dict)

    # Create training config
    training_args = TrainerConfig(
        scheduler_type="warmupcosine",
        lr=1e-4,
        patience=1,
        max_epochs=1,
        warmup_ratio=0.2,
        check_val_every_n_epoch=1,
        val_check_interval=None,
        monitor_metric="val/total_loss",
        mode="min",
        train_batch_size=512,
        test_batch_size=512,
        run_training=True,
        run_testing=True,
        default_root_dir=str(DEFAULT_OUTPUT_PATH / model_type),
        devices=devices,
        precision="16-mixed",
    )

    # Check if model has already passed
    test_log = load_test_log()
    if check_model_passed(model_type, devices, binning, test_log):
        logger.info("=" * 80)
        logger.info(
            f"⏭ Skipping {MODEL_DESCRIPTIONS[model_type]} ({model_type}) - "
            f"devices={devices}, binning={binning} - already passed"
        )
        logger.info("=" * 80)
        test_key = get_test_key(model_type, devices, binning)
        logger.info(f"Previous test: {test_log[test_key].get('timestamp', 'unknown')}")
        return True, True

    logger.info("=" * 80)
    logger.info(f"Testing {MODEL_DESCRIPTIONS[model_type]} ({model_type})")
    logger.info("=" * 80)
    logger.info(f"Data path: {DEFAULT_DATA_PATH}")
    logger.info(f"Output path: {DEFAULT_OUTPUT_PATH}")
    logger.info(f"Devices: {devices}")
    logger.info(f"Binning: {binning}")

    try:
        train_dsn_func(
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            model_type=model_type,
        )
        logger.info("=" * 80)
        logger.info(f"✓ Test passed for {MODEL_DESCRIPTIONS[model_type]} ({model_type})")
        logger.info("=" * 80)

        # Update test log
        update_test_log(
            model_type=model_type,
            devices=devices,
            binning=binning,
            status="passed",
            data_path=DEFAULT_DATA_PATH,
            output_path=DEFAULT_OUTPUT_PATH,
        )

        return True, False
    except Exception as e:
        logger.error("=" * 80)
        logger.error(
            f"✗ Test failed for {MODEL_DESCRIPTIONS[model_type]} ({model_type}) - devices={devices}, binning={binning}"
        )
        logger.error("=" * 80)
        logger.error(f"Error: {e}", exc_info=True)

        # Update test log
        update_test_log(
            model_type=model_type,
            devices=devices,
            binning=binning,
            status="failed",
            data_path=DEFAULT_DATA_PATH,
            output_path=DEFAULT_OUTPUT_PATH,
            error_message=str(e),
        )

        return False, False


def run_single_test(
    model_type: str, devices: int, binning: int | None, test_num: int, total_tests: int
) -> tuple[str, bool, bool]:
    """Run a single test configuration.

    Args:
        model_type: Model type to test
        devices: Number of devices (1 or 2)
        binning: Binning value (None or int)
        test_num: Current test number
        total_tests: Total number of tests

    Returns:
        Tuple of (test_key, success, skipped)
    """
    test_key = get_test_key(model_type, devices, binning)
    logger.info(f"\n[{test_num}/{total_tests}] Testing: {model_type}, devices={devices}, binning={binning}")

    success, was_skipped = test_dsn_models(
        model_type=model_type,
        devices=devices,
        binning=binning,
    )
    return test_key, success, was_skipped


def test_name_dev1_binnone() -> tuple[str, bool, bool]:
    """Test name model with 1 device and binning=None."""
    return run_single_test("name", 1, None, 1, 12)


def test_name_dev2_binnone() -> tuple[str, bool, bool]:
    """Test name model with 2 devices and binning=None."""
    return run_single_test("name", 2, None, 2, 12)


def test_patch_dev1_binnone() -> tuple[str, bool, bool]:
    """Test patch model with 1 device and binning=None."""
    return run_single_test("patch", 1, None, 3, 12)


def test_patch_dev2_binnone() -> tuple[str, bool, bool]:
    """Test patch model with 2 devices and binning=None."""
    return run_single_test("patch", 2, None, 4, 12)


def test_mlp_dev1_binnone() -> tuple[str, bool, bool]:
    """Test mlp model with 1 device and binning=None."""
    return run_single_test("mlp", 1, None, 5, 12)


def test_mlp_dev2_binnone() -> tuple[str, bool, bool]:
    """Test mlp model with 2 devices and binning=None."""
    return run_single_test("mlp", 2, None, 6, 12)


def test_expr_dev1_binnone() -> tuple[str, bool, bool]:
    """Test expr model with 1 device and binning=None."""
    return run_single_test("expr", 1, None, 7, 12)


def test_expr_dev2_binnone() -> tuple[str, bool, bool]:
    """Test expr model with 2 devices and binning=None."""
    return run_single_test("expr", 2, None, 8, 12)


def test_dual_dev1_binnone() -> tuple[str, bool, bool]:
    """Test dual model with 1 device and binning=None."""
    return run_single_test("dual", 1, None, 9, 12)


def test_dual_dev2_binnone() -> tuple[str, bool, bool]:
    """Test dual model with 2 devices and binning=None."""
    return run_single_test("dual", 2, None, 10, 12)


def test_dual_dev1_bin50() -> tuple[str, bool, bool]:
    """Test dual model with 1 device and binning=50."""
    return run_single_test("dual", 1, 50, 11, 12)


def test_dual_dev2_bin50() -> tuple[str, bool, bool]:
    """Test dual model with 2 devices and binning=50."""
    return run_single_test("dual", 2, 50, 12, 12)


def main():
    """Main function to test DSN models."""
    data_path = DEFAULT_DATA_PATH
    output_path = DEFAULT_OUTPUT_PATH

    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        logger.info("Please generate synthetic data first using:")
        logger.info("  python expertments/data_process/process_syncdata.py")
        sys.exit(1)

    logger.info("=" * 80)
    logger.info("DSN Models Test Suite")
    logger.info("=" * 80)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output path: {output_path}")
    logger.info("Binning: None for all models except dual (dual: None and 50)")
    logger.info(f"Test log file: {TEST_LOG_FILE}")

    # Load test log to check for passed models
    test_log = load_test_log()
    if test_log:
        logger.info("\nChecking test log for previously passed models...")
        test_configs = [
            ("name", 1, None),
            ("name", 2, None),
            ("patch", 1, None),
            ("patch", 2, None),
            ("mlp", 1, None),
            ("mlp", 2, None),
            ("expr", 1, None),
            ("expr", 2, None),
            ("dual", 1, None),
            ("dual", 2, None),
            ("dual", 1, 50),
            ("dual", 2, 50),
        ]
        for model_type, devices, binning in test_configs:
            if check_model_passed(model_type, devices, binning, test_log):
                logger.info(f"  {model_type} (devices={devices}, binning={binning}): Already passed (will be skipped)")

    # Define all test functions in order
    test_functions = [
        test_name_dev1_binnone,
        test_mlp_dev1_binnone,
        test_patch_dev1_binnone,
        test_expr_dev1_binnone,
        test_dual_dev1_binnone,
        test_dual_dev1_bin50,
        test_name_dev2_binnone,
        test_mlp_dev2_binnone,
        test_patch_dev2_binnone,
        test_expr_dev2_binnone,
        test_dual_dev2_binnone,
        test_dual_dev2_bin50,
    ]

    # Execute all tests sequentially
    results = {}
    skipped = {}
    for test_func in test_functions:
        test_key, success, was_skipped = test_func()
        results[test_key] = success
        skipped[test_key] = was_skipped

    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("Test Summary")
    logger.info("=" * 80)

    model_types = ["name", "patch", "mlp", "expr", "dual"]
    devices_list = [1, 2]

    for model_type in model_types:
        logger.info(f"\n{MODEL_DESCRIPTIONS[model_type]} ({model_type}):")
        binning_list = [None, 50] if model_type == "dual" else [None]
        for devices in devices_list:
            for binning in binning_list:
                test_key = get_test_key(model_type, devices, binning)
                if skipped.get(test_key, False):
                    status = "⏭ SKIPPED (already passed)"
                elif results.get(test_key, False):
                    status = "✓ PASSED"
                else:
                    status = "✗ FAILED"
                logger.info(f"  devices={devices}, binning={binning}: {status}")

    total = len(results)
    passed = sum(1 for s, sk in zip(results.values(), skipped.values()) if s and not sk)
    failed = sum(1 for s in results.values() if not s)
    skipped_count = sum(1 for sk in skipped.values() if sk)

    logger.info(f"\nTotal: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped_count}")
    logger.info(f"Test log saved to: {TEST_LOG_FILE}")

    if failed > 0:
        logger.warning(f"\n{failed} test(s) failed. Please check the logs above for details.")
        sys.exit(1)
    else:
        logger.info("\nAll tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()
