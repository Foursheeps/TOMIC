#!/usr/bin/env python3
"""Test ADDA models with synthetic data."""

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.append("/your/path/to/TOMIC")

from tomic.dataset.dataconfig import TomicDataConfig
from tomic.model.adda import ExprModelConfig, MLPModelConfig, NameModelConfig, PatchModelConfig
from tomic.train.adda.train import main as train_adda_func
from tomic.train.adda.train_config import TrainerConfig

# Paths
DATA_PATH = Path("/your/path/to/TOMIC/expertments/data_process/synthetic_processed/synthetic_400")
OUTPUT_PATH = Path("/your/path/to/outputs/outputs/test_adda_models")
TEST_LOG_FILE = Path(__file__).parent / "test_results.json"

# Model config mapping
MODEL_CONFIG_MAP = {
    "name": NameModelConfig,
    "patch": PatchModelConfig,
    "mlp": MLPModelConfig,
    "expr": ExprModelConfig,
}

# Training hyperparameters
trainer_args = dict(
    scheduler_type="warmupcosine",
    lr=1e-4,
    patience=1,
    max_epochs=1,
    pretrain_epochs=1,
    warmup_ratio=0.2,
    check_val_every_n_epoch=1,
    val_check_interval=None,
    monitor_metric="val/target_accuracy",
    mode="max",
    train_batch_size=512,
    test_batch_size=512,
    run_training=True,
    run_testing=True,
    devices=1,
    precision="16-mixed",
)

# Model architecture parameters
base_config = dict(dropout=0.1, activation="gelu")
model_configs = {
    "mlp": dict(**base_config, hidden_dims=[32, 32, 32]),
    "default": dict(**base_config, hidden_size=40, num_heads=4, num_layers=1),
}

# Test configurations: (model_type, devices)
test_configs = [
    ("name", 1),
    ("patch", 1),
    ("mlp", 1),
    ("expr", 1),
]


def load_test_log():
    if TEST_LOG_FILE.exists():
        try:
            return json.load(open(TEST_LOG_FILE))
        except (json.JSONDecodeError, OSError):
            return {}
    return {}


def save_test_log(test_log):
    try:
        TEST_LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        json.dump(test_log, open(TEST_LOG_FILE, "w"), indent=2, default=str)
    except OSError:
        pass


def get_test_key(model_type, devices):
    return f"{model_type}_dev{devices}"


def check_passed(model_type, devices, test_log):
    key = get_test_key(model_type, devices)
    return test_log.get(key, {}).get("status") == "passed"


def update_log(model_type, devices, status, error=None):
    test_log = load_test_log()
    test_log[get_test_key(model_type, devices)] = {
        "model_type": model_type,
        "devices": devices,
        "status": status,
        "timestamp": datetime.now().isoformat(),
        "error_message": error,
    }
    save_test_log(test_log)


def test_model(model_type, devices):
    config_path = DATA_PATH / "info_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")

    data_args = TomicDataConfig.from_json_or_kwargs(config_path, binning=50, root_data_path=DATA_PATH)

    # Create model and training configs
    model_config = model_configs.get(model_type, model_configs["default"])
    model_args = MODEL_CONFIG_MAP[model_type](**model_config)
    training_args = TrainerConfig(**trainer_args, default_root_dir=str(OUTPUT_PATH / model_type))

    # Check if already passed
    if check_passed(model_type, devices, load_test_log()):
        return True, True

    # Run test
    try:
        train_adda_func(
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            model_type=model_type,
        )
        update_log(model_type, devices, "passed")
        return True, False
    except Exception as e:
        update_log(model_type, devices, "failed", str(e))
        return False, False


def main():
    if not DATA_PATH.exists():
        print(f"Error: Data path does not exist: {DATA_PATH}")
        sys.exit(1)

    results, skipped = {}, {}
    for model_type, devices in test_configs:
        key = get_test_key(model_type, devices)
        success, was_skipped = test_model(model_type, devices)
        results[key] = success
        skipped[key] = was_skipped

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    for model_type in ["name", "patch", "mlp", "expr"]:
        key = get_test_key(model_type, 1)
        status = "⏭ SKIPPED" if skipped.get(key) else ("✓ PASSED" if results.get(key) else "✗ FAILED")
        print(f"{model_type}: {status}")

    total = len(results)
    passed = sum(results.values()) - sum(skipped.values())
    failed = sum(1 for v in results.values() if not v)
    skipped_count = sum(skipped.values())
    print(f"\nTotal: {total}, Passed: {passed}, Failed: {failed}, Skipped: {skipped_count}")
    sys.exit(1 if failed > 0 else 0)


if __name__ == "__main__":
    main()
