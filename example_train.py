"""
Example training script for TOMIC.

This script demonstrates how to train TOMIC models using different domain adaptation approaches.
You can modify the paths and parameters according to your needs.

Available training methods:
- DSN (Domain Separation Network) - Recommended for TOMIC
- DANN (Domain Adversarial Neural Network)
- ADDA (Adversarial Discriminative Domain Adaptation)
- Usual (Standard supervised learning)

Available model types:
- "mlp": Multi-layer perceptron baseline
- "patch": Patch-based Transformer model
- "expr": Expression-based Transformer model
- "name": Gene name-based Transformer model (ranked tokenization) - Recommended
- "dual": Dual Transformer model combining name and expression features
"""

import sys
from pathlib import Path

# Add TOMIC to Python path (modify this to your actual TOMIC path)
sys.path.append("/your/path/to/TOMIC")

from tomic.dataset.dataconfig import TomicDataConfig

# ============================================================================
# Model Configuration Imports
# ============================================================================
# Choose one of the following training methods and corresponding model configs:
# Option 1: DSN (Domain Separation Network) - Recommended for TOMIC
from tomic.model.dsn import (
    DualTransformerModelConfig4DSN,
    ExprTransformerModelConfig4DSN,
    MLPModelConfig4DSN,
    NameTransformerModelConfig4DSN,
    PatchTransformerModelConfig4DSN,
)
from tomic.train.dsn.train import test, train
from tomic.train.dsn.train_config import TrainerConfig

# Option 2: DANN (Domain Adversarial Neural Network)
# from tomic.model.dann import (
#     DualTransformerModelConfig,
#     ExprModelConfig,
#     MLPModelConfig,
#     NameModelConfig,
#     PatchModelConfig,
# )
# from tomic.train.dann.train import test, train
# from tomic.train.dann.train_config import TrainerConfig

# Option 3: ADDA (Adversarial Discriminative Domain Adaptation)
# from tomic.model.adda import (
#     DualTransformerModelConfig,
#     ExprModelConfig,
#     MLPModelConfig,
#     NameModelConfig,
#     PatchModelConfig,
# )
# from tomic.train.adda.train import test, train
# from tomic.train.adda.train_config import TrainerConfig

# Option 4: Usual (Standard supervised learning)
# from tomic.model.usual import (
#     DualTransformerModelConfig,
#     ExprModelConfig,
#     MLPModelConfig,
#     NameModelConfig,
#     PatchModelConfig,
# )
# from tomic.train.usual.train import test, train
# from tomic.train.usual.train_config import TrainerConfig


def main():
    """Example training function for TOMIC."""
    # ============================================================================
    # Step 1: Configure data paths
    # ============================================================================
    # Path to your data directory containing info_config.json
    # The data directory should contain:
    # - info_config.json: Data configuration file
    # - source domain data (metastatic cells)
    # - target domain data (primary tumor cells)
    data_path = Path("/path/to/your/data")

    # Path to save checkpoints and logs
    output_dir = Path("/path/to/output")

    # ============================================================================
    # Step 2: Create data configuration
    # ============================================================================
    # Load data configuration from info_config.json
    # This file should contain class_map, seq_len, num_classes, vocab_size, etc.
    data_args = TomicDataConfig.from_json_or_kwargs(
        data_path / "info_config.json",
    )

    # ============================================================================
    # Step 3: Configure model architecture
    # ============================================================================
    # Choose model type: "mlp", "patch", "expr", "name", or "dual"
    # Recommended: "name" (Name Transformer) for best performance
    model_type = "name"

    # Configure model architecture based on model type
    if model_type == "name":
        # Name Transformer model (ranked gene name-based tokenization)
        model_args = NameTransformerModelConfig4DSN(
            dropout=0.1,
            activation="gelu",
            hidden_size=40,
            num_heads=4,
            num_layers=1,
        )
    elif model_type == "patch":
        # Patch-based Transformer model
        model_args = PatchTransformerModelConfig4DSN(
            dropout=0.1,
            activation="gelu",
            hidden_size=40,
            patch_size=40,
            num_heads=4,
            num_layers=1,
        )
    elif model_type == "expr":
        # Expression-based Transformer model
        model_args = ExprTransformerModelConfig4DSN(
            dropout=0.1,
            activation="gelu",
            hidden_size=40,
            num_heads=4,
            num_layers=1,
        )
    elif model_type == "mlp":
        # Multi-layer perceptron baseline
        model_args = MLPModelConfig4DSN(
            dropout=0.1,
            activation="gelu",
            hidden_dims=[32, 32],
        )
    elif model_type == "dual":
        # Dual Transformer model (combines name and expression)
        model_args = DualTransformerModelConfig4DSN(
            dropout=0.1,
            activation="gelu",
            hidden_size=40,
            num_heads_cross_attn=4,
            num_layers_cross_attn=1,
            num_heads_encoder=4,
            num_layers_encoder=1,
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # ============================================================================
    # Step 4: Configure training parameters
    # ============================================================================
    training_args = TrainerConfig(
        # Data paths
        default_root_dir=str(output_dir),
        # Training hyperparameters
        lr=1e-4,  # Learning rate
        max_epochs=80,  # Maximum number of training epochs
        scheduler_type="warmupcosine",  # Learning rate scheduler
        warmup_ratio=0.1,  # Warmup ratio for scheduler
        # Batch sizes
        train_batch_size=128,  # Training batch size
        test_batch_size=128,  # Testing batch size
        num_workers=4,  # Number of data loading workers
        # PyTorch Lightning Trainer configuration
        seed=2025,  # Random seed for reproducibility
        devices=1,  # Number of GPUs (set to number of available GPUs)
        precision="16-mixed",  # Mixed precision training (faster, less memory)
        log_every_n_steps=50,  # Logging frequency
        check_val_every_n_epoch=1,  # Validation frequency
        # Callback configuration
        patience=10,  # Early stopping patience
        monitor_metric="val/target_accuracy",  # Metric to monitor for early stopping
        mode="max",  # "max" for accuracy/AUC, "min" for loss
        save_top_k=2,  # Number of best checkpoints to save
        # DSN loss weights (only for DSN training method)
        alpha=4.0,  # Reconstruction loss weight
        beta=0.25,  # Difference loss weight
        gamma=0.1,  # DANN loss weight
    )

    # ============================================================================
    # Step 5: Train the model
    # ============================================================================
    print("=" * 80)
    print("Starting TOMIC Training")
    print("=" * 80)
    print(f"Data path: {data_path}")
    print(f"Output directory: {output_dir}")
    print(f"Model type: {model_type}")
    print("Training method: DSN")
    print("=" * 80)

    # Train the model
    checkpoint_path = train(
        data_args=data_args,
        model_args=model_args,
        training_args=training_args,
        model_type=model_type,
    )

    if checkpoint_path:
        print(f"\n✓ Training completed! Best checkpoint saved at: {checkpoint_path}")
    else:
        print("\n⚠ Training completed but no checkpoint was saved.")

    # ============================================================================
    # Step 6: Test the model (optional)
    # ============================================================================
    if checkpoint_path:
        print("\n" + "=" * 80)
        print("Testing the trained model")
        print("=" * 80)

        # Update checkpoint path in training args
        training_args.checkpoint_path = str(checkpoint_path)

        # Test the model
        results = test(
            data_args=data_args,
            model_args=model_args,
            training_args=training_args,
            model_type=model_type,
        )

        # Print test results
        print("\nTest Results:")
        print("-" * 80)
        print("Source Domain (Metastatic Cells):")
        print(f"  Accuracy: {results.get('train/accuracy', 'N/A'):.4f}")
        print(f"  AUC: {results.get('train/auc', 'N/A'):.4f}")
        print(f"  F1 Score (macro): {results.get('train/f1_macro', 'N/A'):.4f}")
        print("\nTarget Domain (Primary Tumor Cells):")
        print(f"  Accuracy: {results.get('test/accuracy', 'N/A'):.4f}")
        print(f"  AUC: {results.get('test/auc', 'N/A'):.4f}")
        print(f"  F1 Score (macro): {results.get('test/f1_macro', 'N/A'):.4f}")
        print("=" * 80)

        # Save results to file (optional)
        import json

        results_file = output_dir / "test_results.json"
        with open(results_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nFull results saved to: {results_file}")


if __name__ == "__main__":
    main()
