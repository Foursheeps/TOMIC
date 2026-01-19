import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.append("/your/path/to/TOMIC")

from datmp.dataset.dataconfig import DatmpDataConfig
from datmp.model.dann import (
    DualTransformerModelConfig,
    ExprModelConfig,
    MLPModelConfig,
    NameModelConfig,
    PatchModelConfig,
)
from datmp.train.dann.train import main as train_func
from datmp.train.dann.train_config import TrainerConfig


def train(
    model_type: str,
    data_args: DatmpDataConfig,
    training_args: TrainerConfig,
    base_model_args: dict = None,
    mlp_model_args: dict = None,
    transformer_model_args: dict = None,
    dual_model_args: dict = None,
):
    """Train DANN models.

    Args:
        model_type: Model type to train ("mlp", "patch", "expr", "name", "dual")
        data_args: Data configuration
        training_args: Training arguments
        base_model_args: Base model arguments (optional)
        mlp_model_args: MLP model arguments (optional)
        transformer_model_args: Transformer model arguments (optional)
        dual_model_args: Dual model arguments (optional)
    """
    models = {
        "mlp": MLPModelConfig(**base_model_args, **mlp_model_args),
        "patch": PatchModelConfig(**base_model_args, **transformer_model_args),
        "expr": ExprModelConfig(**base_model_args, **transformer_model_args),
        "name": NameModelConfig(**base_model_args, **transformer_model_args),
        "dual": DualTransformerModelConfig(**base_model_args, **dual_model_args),
    }

    return train_func(
        data_args=data_args,
        model_args=models[model_type],
        training_args=training_args,
        model_type=model_type,
    )


def arg_parser():
    """Parse command line arguments."""
    parser = ArgumentParser(description="Train DANN models", allow_abbrev=False)

    # Model and data arguments
    parser.add_argument("--train_models", type=str, default="['mlp', 'patch', 'expr', 'name', 'dual']")
    parser.add_argument("--data_path", type=str, default=None)
    parser.add_argument("--bingings", type=str, default="[None, 50]")
    parser.add_argument("--default_root_dir", type=str, default=None)

    # TrainerConfig arguments - Training hyperparameters
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_epochs", type=int, default=200)
    parser.add_argument("--scheduler_type", type=str, default="warmupcosine")
    parser.add_argument("--warmup_ratio", type=float, default=0.1)

    # Batch sizes
    parser.add_argument("--train_batch_size", type=int, default=128)
    parser.add_argument("--test_batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)

    # PyTorch Lightning Trainer configuration
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--devices", type=int, default=1)
    parser.add_argument("--precision", type=str, default="16-mixed")
    parser.add_argument("--log_every_n_steps", type=int, default=50)
    parser.add_argument("--val_check_interval", type=int, default=None)
    parser.add_argument("--check_val_every_n_epoch", type=int, default=1)

    # Callback configuration
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--monitor_metric", type=str, default="val/target_accuracy")
    parser.add_argument("--mode", type=str, default="max")
    parser.add_argument("--save_top_k", type=int, default=2)

    # DANN-specific configuration
    parser.add_argument("--gamma", type=float, default=0.1)

    # Control flags
    parser.add_argument("--run_training", type=int, default=0)
    parser.add_argument("--run_testing", type=int, default=0)
    parser.add_argument("--checkpoint_path", type=str, default=None)

    # Model arguments
    # Base model arguments
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--activation", type=str, default="gelu")

    # MLP model arguments
    parser.add_argument("--hidden_dims", type=str, default="[32, 32]")

    # Transformer model arguments
    parser.add_argument("--hidden_size", type=int, default=40)
    parser.add_argument("--patch_size", type=int, default=40)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=1)

    # Dual Transformer model arguments
    parser.add_argument("--num_heads_cross_attn", type=int, default=4)
    parser.add_argument("--num_layers_cross_attn", type=int, default=1)
    parser.add_argument("--num_heads_encoder", type=int, default=4)
    parser.add_argument("--num_layers_encoder", type=int, default=1)

    args = parser.parse_args()

    args.run_training = args.run_training == 1
    args.run_testing = args.run_testing == 1

    # Parse bingings list
    bingings = eval(args.bingings) if args.bingings else [None]

    data_args = DatmpDataConfig.from_json_or_kwargs(
        Path(args.data_path) / "info_config.json",
    )

    train_args = TrainerConfig(**vars(args))

    base_model_args = dict(
        dropout=args.dropout,
        activation=args.activation,
    )
    mlp_model_args = dict(
        hidden_dims=eval(args.hidden_dims),
    )
    transformer_model_args = dict(
        hidden_size=args.hidden_size,
        patch_size=args.patch_size,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
    )
    dual_model_args = dict(
        hidden_size=args.hidden_size,
        num_heads_cross_attn=args.num_heads_cross_attn,
        num_layers_cross_attn=args.num_layers_cross_attn,
        num_heads_encoder=args.num_heads_encoder,
        num_layers_encoder=args.num_layers_encoder,
    )

    train_models = eval(args.train_models)
    return (
        data_args,
        train_args,
        base_model_args,
        mlp_model_args,
        transformer_model_args,
        dual_model_args,
        train_models,
        bingings,
    )


if __name__ == "__main__":
    (
        data_args,
        train_args,
        base_model_args,
        mlp_model_args,
        transformer_model_args,
        dual_model_args,
        train_models,
        bingings,
    ) = arg_parser()

    # Create default root directory
    default_root_dir = Path(train_args.default_root_dir)

    for model_type in train_models:
        if "dual" == model_type:
            # train two models with different binning
            for binning in bingings:
                data_args.binning = binning
                train_args.default_root_dir = default_root_dir / f"{model_type}_{binning}"
                train(
                    model_type=model_type,
                    data_args=data_args,
                    training_args=train_args,
                    base_model_args=base_model_args,
                    mlp_model_args=mlp_model_args,
                    transformer_model_args=transformer_model_args,
                    dual_model_args=dual_model_args,
                )

        else:
            data_args.binning = None
            train_args.default_root_dir = default_root_dir / f"{model_type}"
            train(
                model_type=model_type,
                data_args=data_args,
                training_args=train_args,
                base_model_args=base_model_args,
                mlp_model_args=mlp_model_args,
                transformer_model_args=transformer_model_args,
                dual_model_args=dual_model_args,
            )
