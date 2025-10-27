import argparse

import seisbench.data as sbd
import seisbench.models as sbm
from seisbench.data import BenchmarkDataset
import seisbench
seisbench.use_backup_repository()

from my_project.tutorial.tutorial import (
    test_load_data,
    test_generator,
)

# Import model classes
from my_project.models.phasenet_mag.model import PhaseNetMag
from my_project.models.phasenetLSTM.model import PhaseNetLSTM
from my_project.models.phasenetLSTM.modelv2 import PhaseNetConvLSTM
from my_project.models.AMAG_v2.model import MagnitudeNet
from my_project.models.EQTransformer.model import EQTransformerMag
from my_project.models.ViT.model import ViTMagnitudeEstimator
from my_project.models.UMamba_mag.model import UMambaMag

# Import unified training and inference functions
from my_project.utils.unified_training import (
    train_phase_model,
    evaluate_phase_model_unified,
    train_magnitude_model,
    evaluate_magnitude_model,
)

from my_project.utils.utils import plot_training_history


def extract_model_params(args, model_type):
    """Extract relevant model parameters from command line arguments"""
    params = {}

    # Common parameters
    if hasattr(args, "filter_factor") and args.filter_factor != 1:
        params["filter_factor"] = args.filter_factor
    if hasattr(args, "early_stopping_patience") and args.early_stopping_patience != 10:
        params["early_stopping_patience"] = args.early_stopping_patience

    # Universal training parameters for magnitude models
    if model_type in ["phasenet_mag", "amag_v2", "eqtransformer_mag", "vit_mag", "umamba_mag"]:
        if hasattr(args, "learning_rate") and args.learning_rate is not None:
            params["learning_rate"] = args.learning_rate
        if hasattr(args, "batch_size") and args.batch_size is not None:
            params["batch_size"] = args.batch_size

    # PhaseNetLSTM specific parameters
    if model_type == "phasenet_lstm":
        if hasattr(args, "lstm_hidden_size") and args.lstm_hidden_size is not None:
            params["lstm_hidden_size"] = args.lstm_hidden_size
        if hasattr(args, "lstm_num_layers") and args.lstm_num_layers != 1:
            params["lstm_num_layers"] = args.lstm_num_layers
        if hasattr(args, "lstm_bidirectional"):
            params["lstm_bidirectional"] = args.lstm_bidirectional

    # PhaseNetConvLSTM specific parameters
    elif model_type == "phasenet_conv_lstm":
        if hasattr(args, "convlstm_hidden") and args.convlstm_hidden != 64:
            params["convlstm_hidden"] = args.convlstm_hidden

    # PhaseNetMag specific parameters
    elif model_type == "phasenet_mag":
        if hasattr(args, "norm") and args.norm != "std":
            params["norm"] = args.norm

    # MagnitudeNet specific parameters
    elif model_type == "amag_v2":
        if hasattr(args, "lstm_hidden") and args.lstm_hidden != 128:
            params["lstm_hidden"] = args.lstm_hidden
        if hasattr(args, "lstm_layers") and args.lstm_layers != 2:
            params["lstm_layers"] = args.lstm_layers
        if hasattr(args, "dropout") and args.dropout != 0.2:
            params["dropout"] = args.dropout

    # EQTransformerMag specific parameters
    elif model_type == "eqtransformer_mag":
        if hasattr(args, "lstm_blocks") and args.lstm_blocks != 3:
            params["lstm_blocks"] = args.lstm_blocks
        if hasattr(args, "drop_rate") and args.drop_rate != 0.1:
            params["drop_rate"] = args.drop_rate
        if hasattr(args, "norm") and args.norm != "std":
            params["norm"] = args.norm
        # EQTransformerMag-specific training parameters
        if hasattr(args, "warmup_epochs") and args.warmup_epochs != 5:
            params["warmup_epochs"] = args.warmup_epochs

    # ViT specific parameters
    elif model_type == "vit_mag":
        if hasattr(args, "patch_size") and args.patch_size != 5:
            params["patch_size"] = args.patch_size
        if hasattr(args, "embed_dim") and args.embed_dim != 100:
            params["embed_dim"] = args.embed_dim
        if hasattr(args, "num_transformer_blocks") and args.num_transformer_blocks != 4:
            params["num_transformer_blocks"] = args.num_transformer_blocks
        if hasattr(args, "num_heads") and args.num_heads != 4:
            params["num_heads"] = args.num_heads
        if hasattr(args, "dropout") and args.dropout != 0.1:
            params["dropout"] = args.dropout
        if hasattr(args, "final_dropout") and args.final_dropout != 0.5:
            params["final_dropout"] = args.final_dropout
        if hasattr(args, "norm") and args.norm != "std":
            params["norm"] = args.norm

    # UMamba specific parameters
    elif model_type == "umamba_mag":
        if hasattr(args, "n_stages") and args.n_stages != 4:
            params["n_stages"] = args.n_stages
        if hasattr(args, "kernel_size") and args.kernel_size != 7:
            params["kernel_size"] = args.kernel_size
        if hasattr(args, "n_blocks_per_stage") and args.n_blocks_per_stage != 2:
            params["n_blocks_per_stage"] = args.n_blocks_per_stage
        if hasattr(args, "deep_supervision") and args.deep_supervision:
            params["deep_supervision"] = args.deep_supervision
        if hasattr(args, "norm") and args.norm != "std":
            params["norm"] = args.norm

    return params


def create_phase_model(model_type: str, **kwargs):
    """Create a phase model based on model_type"""
    if model_type == "phasenet":
        return sbm.PhaseNet(
            phases="PSN", norm="std", default_args={"blinding": (200, 200)}
        )
    elif model_type == "phasenet_lstm":
        # Extract PhaseNetLSTM-specific parameters with defaults
        filter_factor = kwargs.get("filter_factor", 1)
        lstm_hidden_size = kwargs.get("lstm_hidden_size", None)
        lstm_num_layers = kwargs.get("lstm_num_layers", 1)
        lstm_bidirectional = kwargs.get("lstm_bidirectional", True)

        return PhaseNetLSTM(
            filter_factor=filter_factor,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_bidirectional=lstm_bidirectional,
        )
    elif model_type == "phasenet_conv_lstm":
        # Extract PhaseNetConvLSTM-specific parameters with defaults
        filter_factor = kwargs.get("filter_factor", 1)
        convlstm_hidden = kwargs.get("convlstm_hidden", 64)

        return PhaseNetConvLSTM(
            filter_factor=filter_factor,
            convlstm_hidden=convlstm_hidden,
        )
    else:
        raise ValueError(f"Unknown phase model type: {model_type}")


def create_magnitude_model(model_type: str, **kwargs):
    """Create a magnitude model based on model_type"""
    if model_type == "phasenet_mag":
        # Extract PhaseNetMag-specific parameters with defaults
        filter_factor = kwargs.get("filter_factor", 1)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)

        return PhaseNetMag(
            in_channels=in_channels,
            sampling_rate=sampling_rate,
            norm=norm,
            filter_factor=filter_factor,
        )
    elif model_type == "amag_v2":
        # Extract MagnitudeNet-specific parameters with defaults
        filter_factor = kwargs.get("filter_factor", 1)
        lstm_hidden = kwargs.get("lstm_hidden", 128)
        lstm_layers = kwargs.get("lstm_layers", 2)
        dropout = kwargs.get("dropout", 0.2)

        return MagnitudeNet(
            in_channels=3,
            sampling_rate=100,
            filter_factor=filter_factor,
            lstm_hidden=lstm_hidden,
            lstm_layers=lstm_layers,
            dropout=dropout,
        )
    elif model_type == "eqtransformer_mag":
        # Extract EQTransformerMag-specific parameters with defaults
        lstm_blocks = kwargs.get("lstm_blocks", 3)
        drop_rate = kwargs.get("drop_rate", 0.1)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)

        return EQTransformerMag(
            in_channels=in_channels,
            in_samples=3001,  # 30 seconds at 100Hz
            sampling_rate=sampling_rate,
            lstm_blocks=lstm_blocks,
            drop_rate=drop_rate,
            norm=norm,
        )
    elif model_type == "vit_mag":
        # Extract ViT-specific parameters with defaults
        conv_channels = kwargs.get("conv_channels", [64, 32, 32, 32])
        pool_sizes = kwargs.get("pool_sizes", [2, 2, 2, 5])
        patch_size = kwargs.get("patch_size", 5)
        embed_dim = kwargs.get("embed_dim", 100)
        num_transformer_blocks = kwargs.get("num_transformer_blocks", 4)
        num_heads = kwargs.get("num_heads", 4)
        transformer_mlp_dim = kwargs.get("transformer_mlp_dim", 200)
        final_mlp_dims = kwargs.get("final_mlp_dims", [1000, 500])
        dropout = kwargs.get("dropout", 0.1)
        final_dropout = kwargs.get("final_dropout", 0.5)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)

        return ViTMagnitudeEstimator(
            in_channels=in_channels,
            sampling_rate=sampling_rate,
            conv_channels=conv_channels,
            pool_sizes=pool_sizes,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_transformer_blocks=num_transformer_blocks,
            num_heads=num_heads,
            transformer_mlp_dim=transformer_mlp_dim,
            final_mlp_dims=final_mlp_dims,
            dropout=dropout,
            final_dropout=final_dropout,
            norm=norm,
        )
    elif model_type == "umamba_mag":
        # Extract UMamba-specific parameters with defaults
        n_stages = kwargs.get("n_stages", 4)
        features_per_stage = kwargs.get("features_per_stage", [8, 16, 32, 64])
        kernel_size = kwargs.get("kernel_size", 7)
        strides = kwargs.get("strides", [2, 2, 2, 2])
        n_blocks_per_stage = kwargs.get("n_blocks_per_stage", 2)
        n_conv_per_stage_decoder = kwargs.get("n_conv_per_stage_decoder", 2)
        deep_supervision = kwargs.get("deep_supervision", False)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)

        return UMambaMag(
            in_channels=in_channels,
            sampling_rate=sampling_rate,
            norm=norm,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_size=kernel_size,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,
            deep_supervision=deep_supervision,
        )
    else:
        raise ValueError(f"Unknown magnitude model type: {model_type}")


def get_model_name(model_type: str, data_name: str, **kwargs):
    """Generate model name for saving based on model type and parameters"""
    if model_type == "phasenet":
        return f"PhaseNet_{data_name}"
    elif model_type == "phasenet_lstm":
        filter_factor = kwargs.get("filter_factor", 1)
        lstm_hidden_size = kwargs.get("lstm_hidden_size", None)
        lstm_num_layers = kwargs.get("lstm_num_layers", 1)
        lstm_bidirectional = kwargs.get("lstm_bidirectional", True)
        return f"PhaseNetLSTM_{data_name}_f{filter_factor}_h{lstm_hidden_size or 'auto'}_l{lstm_num_layers}_{'bi' if lstm_bidirectional else 'uni'}"
    elif model_type == "phasenet_conv_lstm":
        filter_factor = kwargs.get("filter_factor", 1)
        lstm_hidden_size = kwargs.get("lstm_hidden_size", None)
        lstm_num_layers = kwargs.get("lstm_num_layers", 1)
        lstm_bidirectional = kwargs.get("lstm_bidirectional", True)
        return f"PhaseNetConvLSTM_{data_name}_f{filter_factor}_h{lstm_hidden_size or 'auto'}_l{lstm_num_layers}_{'bi' if lstm_bidirectional else 'uni'}"
    elif model_type == "phasenet_mag":
        return f"PhaseNetMag_{data_name}"
    elif model_type == "amag_v2":
        return f"magnitudenet_v1"
    elif model_type == "eqtransformer_mag":
        return f"EQTransformerMag_{data_name}"
    elif model_type == "vit_mag":
        return f"ViTMag_{data_name}"
    elif model_type == "umamba_mag":
        return f"UMambaMag_{data_name}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_phase_unified(
    data: BenchmarkDataset, model_type: str, epochs: int = 5, **kwargs
):
    """Unified phase model training function"""
    print(f"\nTraining {model_type} on {data.name} for {epochs} epochs...")

    model = create_phase_model(model_type, **kwargs)
    model_name = get_model_name(model_type, data.name, **kwargs)

    if model_type == "phasenet":
        learning_rate = kwargs.get("learning_rate", 1e-2)
    else:  # phasenet_lstm or phasenet_conv_lstm
        learning_rate = kwargs.get("learning_rate", 1e-3)

    return train_phase_model(
        model=model,
        model_name=model_name,
        data=data,
        learning_rate=learning_rate,
        epochs=epochs,
        **kwargs,
    )


def train_magnitude_unified(
    data: BenchmarkDataset, model_type: str, epochs: int = 50, **kwargs
):
    """Unified magnitude model training function"""
    print(f"\nTraining {model_type} on {data.name} for {epochs} epochs...")

    model = create_magnitude_model(model_type, **kwargs)
    model_name = get_model_name(model_type, data.name, **kwargs)

    # Set model-specific defaults for learning_rate and batch_size if not provided by user
    if model_type == "phasenet_mag":
        learning_rate = kwargs.pop("learning_rate", 1e-4)
        batch_size = kwargs.pop("batch_size", 256)
        optimizer_name = kwargs.get("optimizer_name", "Adam")
    elif model_type == "eqtransformer_mag":
        learning_rate = kwargs.pop("learning_rate", 1e-4)  # Lower LR for transformers
        batch_size = kwargs.pop("batch_size", 64)  # Smaller batch for memory efficiency
        optimizer_name = kwargs.get("optimizer_name", "AdamW")
        kwargs.setdefault("scheduler_factor", 0.5)
        kwargs.setdefault("gradient_clip", 1.0)
        kwargs.setdefault("warmup_epochs", 5)  # Transformer warmup
    elif model_type == "vit_mag":
        learning_rate = kwargs.pop("learning_rate", 1e-4)  # Lower LR for transformers
        batch_size = kwargs.pop("batch_size", 64)  # Smaller batch for memory efficiency
        optimizer_name = kwargs.get("optimizer_name", "AdamW")
        kwargs.setdefault("scheduler_factor", 0.5)
        kwargs.setdefault("gradient_clip", 1.0)
        kwargs.setdefault("warmup_epochs", 5)  # Transformer warmup
    else:  # amag_v2
        learning_rate = kwargs.pop("learning_rate", 1e-3)
        batch_size = kwargs.pop("batch_size", 256)
        optimizer_name = kwargs.get("optimizer_name", "AdamW")
        kwargs.setdefault("scheduler_factor", 0.5)
        kwargs.setdefault("gradient_clip", 1.0)

    return train_magnitude_model(
        model=model,
        model_name=model_name,
        data=data,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        **kwargs,
    )


def evaluate_phase_unified(
    data: BenchmarkDataset, model_type: str, model_path: str, **kwargs
):
    """Unified phase model evaluation function"""
    print(f"\nEvaluating {model_type} on {data.name}...")

    if model_type == "phasenet":
        model = sbm.PhaseNet(
            phases="PSN", norm="std", default_args={"blinding": (200, 200)}
        )
    elif model_type == "phasenet_lstm":
        # Extract PhaseNetLSTM-specific parameters with defaults
        filter_factor = kwargs.get("filter_factor", 1)
        lstm_hidden_size = kwargs.get("lstm_hidden_size", None)
        lstm_num_layers = kwargs.get("lstm_num_layers", 1)
        lstm_bidirectional = kwargs.get("lstm_bidirectional", True)

        model = PhaseNetLSTM(
            filter_factor=filter_factor,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            lstm_bidirectional=lstm_bidirectional,
        )
    elif model_type == "phasenet_conv_lstm":
        # Extract PhaseNetConvLSTM-specific parameters with defaults
        filter_factor = kwargs.get("filter_factor", 1)
        convlstm_hidden = kwargs.get("convlstm_hidden", 64)

        model = PhaseNetConvLSTM(
            filter_factor=filter_factor,
            convlstm_hidden=convlstm_hidden,
        )
    else:
        raise ValueError(f"Unknown phase model type: {model_type}")

    return evaluate_phase_model_unified(
        model=model, model_path=model_path, data=data, **kwargs
    )


def evaluate_magnitude_unified(
    data: BenchmarkDataset, model_type: str, model_path: str, **kwargs
):
    """Unified magnitude model evaluation function"""
    print(f"\nEvaluating {model_type} on {data.name}...")

    model = create_magnitude_model(model_type, **kwargs)

    # Extract only evaluation-specific parameters
    eval_kwargs = {}
    if "batch_size" in kwargs:
        eval_kwargs["batch_size"] = kwargs["batch_size"]
    if "plot_examples" in kwargs:
        eval_kwargs["plot_examples"] = kwargs["plot_examples"]
    if "num_examples" in kwargs:
        eval_kwargs["num_examples"] = kwargs["num_examples"]

    return evaluate_magnitude_model(
        model=model, model_path=model_path, data=data, **eval_kwargs
    )


def tutorial_test_load_and_generator(data: BenchmarkDataset):
    """Test data loading and generator functionality"""
    print("\n" + "=" * 50)
    print("TUTORIAL: TESTING DATA LOADING AND GENERATOR")
    print("=" * 50)

    test_load_data()
    test_generator()

    print("Data loading and generator tests completed!")


def tutorial_tests(data: BenchmarkDataset, model_path: str = ""):
    """Original PhaseNet tutorial tests (deprecated - use individual tutorial functions)"""
    print("\n" + "=" * 50)
    print("TUTORIAL: RUNNING ALL TESTS (DEPRECATED)")
    print("=" * 50)
    print(
        "Warning: This function is deprecated. Consider using individual tutorial functions:"
    )
    print("  - tutorial_test_load_and_generator")
    print("  - train_phase --model_type phasenet")
    print("  - eval_phase --model_type phasenet")
    print("=" * 50)

    # Run evaluation only (as in original)
    if model_path:
        model = create_phase_model("phasenet")
        results = evaluate_phase_model_unified(
            model=model, model_path=model_path, data=data
        )
        return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PhaseNet and Magnitude Prediction Workflows"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETHZ",
        choices=["ETHZ", "STEAD", "GEOFON", "MLAAPDE", "LENDB", "TXED"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to model checkpoint file (.pt) or training history file (.pt). Required for eval_phase, eval_mag, tutorial_evaluate_phasenet, and plot_history modes",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tutorial_test_load_and_generator",
        choices=[
            "train_phase",
            "train_mag",
            "eval_phase",
            "eval_mag",
            "tutorial_test_load_and_generator",
            "tutorial_train_phasenet",
            "tutorial_evaluate_phasenet",
            "tutorial",
            "plot_history",
        ],
        help="Workflow mode",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="phasenet",
        choices=[
            "phasenet",
            "phasenet_lstm",
            "phasenet_conv_lstm",
            "phasenet_mag",
            "amag_v2",
            "eqtransformer_mag",
            "vit_mag",
            "umamba_mag",
        ],
        help="Model type to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=None,
        help="Learning rate for training (default varies by model type)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Batch size for training (default varies by model type)",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display plots on screen and save PNG files (for eval_mag and plot_history modes)",
    )

    # Model parameter arguments
    parser.add_argument(
        "--filter_factor",
        type=int,
        default=1,
        help="Filter factor for model capacity (all models)",
    )
    parser.add_argument(
        "--lstm_hidden_size",
        type=int,
        default=None,
        help="LSTM hidden size for PhaseNetLSTM (None=auto)",
    )
    parser.add_argument(
        "--lstm_num_layers",
        type=int,
        default=1,
        help="Number of LSTM layers for PhaseNetLSTM",
    )
    parser.add_argument(
        "--lstm_bidirectional",
        action="store_true",
        default=True,
        help="Use bidirectional LSTM for PhaseNetLSTM",
    )
    parser.add_argument(
        "--convlstm_hidden",
        type=int,
        default=64,
        help="ConvLSTM hidden size for PhaseNetConvLSTM",
    )
    parser.add_argument(
        "--norm",
        type=str,
        default="std",
        choices=["std", "peak"],
        help="Normalization method for PhaseNetMag",
    )
    parser.add_argument(
        "--lstm_hidden", type=int, default=128, help="LSTM hidden size for MagnitudeNet"
    )
    parser.add_argument(
        "--lstm_layers",
        type=int,
        default=2,
        help="Number of LSTM layers for MagnitudeNet",
    )
    parser.add_argument(
        "--dropout", type=float, default=0.2, help="Dropout rate for MagnitudeNet"
    )
    parser.add_argument(
        "--lstm_blocks",
        type=int,
        default=3,
        help="Number of LSTM blocks for EQTransformerMag",
    )
    parser.add_argument(
        "--drop_rate",
        type=float,
        default=0.1,
        help="Dropout rate for EQTransformerMag",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Stop training if no improvement for N epochs",
    )

    # EQTransformerMag specific training parameters
    parser.add_argument(
        "--warmup_epochs",
        type=int,
        default=5,
        help="Number of warmup epochs for EQTransformerMag learning rate scheduler",
    )

    # ViT specific parameters
    parser.add_argument(
        "--patch_size",
        type=int,
        default=5,
        help="Patch size for ViT model",
    )
    parser.add_argument(
        "--embed_dim",
        type=int,
        default=100,
        help="Embedding dimension for ViT model",
    )
    parser.add_argument(
        "--num_transformer_blocks",
        type=int,
        default=4,
        help="Number of transformer blocks for ViT model",
    )
    parser.add_argument(
        "--num_heads",
        type=int,
        default=4,
        help="Number of attention heads for ViT model",
    )
    parser.add_argument(
        "--final_dropout",
        type=float,
        default=0.5,
        help="Final dropout rate for ViT model",
    )

    # UMamba model arguments
    parser.add_argument(
        "--n_stages",
        type=int,
        default=4,
        help="Number of encoder/decoder stages for UMamba model",
    )
    parser.add_argument(
        "--kernel_size",
        type=int,
        default=7,
        help="Kernel size for convolutions in UMamba model",
    )
    parser.add_argument(
        "--n_blocks_per_stage",
        type=int,
        default=2,
        help="Number of residual blocks per stage in UMamba model",
    )
    parser.add_argument(
        "--deep_supervision",
        action="store_true",
        help="Enable deep supervision for UMamba model",
    )

    args = parser.parse_args()

    if args.mode != "plot_history":
        # Load dataset
        print("\n" + "=" * 50)
        print("LOADING DATA")
        print("=" * 50)
        if args.dataset == "ETHZ":
            data = sbd.ETHZ(sampling_rate=100)
        elif args.dataset == "STEAD":
            data = sbd.STEAD(sampling_rate=100)
        elif args.dataset == "GEOFON":
            data = sbd.GEOFON(sampling_rate=100)
        elif args.dataset == "MLAAPDE":
            data = sbd.MLAAPDE(sampling_rate=100)
        elif args.dataset == "LENDB":
            data = sbd.LENDB(sampling_rate=100)
        elif args.dataset == "TXED":
            data = sbd.TXED(sampling_rate=100)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        print(f"{data.name} dataset loaded: {len(data)} samples")

    # Run appropriate workflow
    if args.mode == "train_phase":
        if args.model_type in ["phasenet", "phasenet_lstm", "phasenet_conv_lstm"]:
            model_params = extract_model_params(args, args.model_type)
            train_phase_unified(
                data, args.model_type, epochs=args.epochs, **model_params
            )
        else:
            print(f"Error: {args.model_type} is not a valid phase model type")
            print(
                "Valid phase model types: phasenet, phasenet_lstm, phasenet_conv_lstm"
            )
            exit(1)
    elif args.mode == "train_mag":
        if args.model_type in [
            "phasenet_mag",
            "amag_v2",
            "eqtransformer_mag",
            "vit_mag",
            "umamba_mag"
        ]:
            model_params = extract_model_params(args, args.model_type)
            train_magnitude_unified(
                data, args.model_type, epochs=args.epochs, **model_params
            )
        else:
            print(f"Error: {args.model_type} is not a valid magnitude model type")
            print(
                "Valid magnitude model types: phasenet_mag, amag_v2, eqtransformer_mag, vit_mag, umamba_mag"
            )
            exit(1)
    elif args.mode == "eval_phase":
        if not args.model_path:
            print("Error: --model_path is required for eval_phase mode")
            exit(1)
        if args.model_type in ["phasenet", "phasenet_lstm", "phasenet_conv_lstm"]:
            model_params = extract_model_params(args, args.model_type)
            evaluate_phase_unified(
                data, args.model_type, args.model_path, **model_params
            )
        else:
            print(f"Error: {args.model_type} is not a valid phase model type")
            print(
                "Valid phase model types: phasenet, phasenet_lstm, phasenet_conv_lstm"
            )
            exit(1)
    elif args.mode == "eval_mag":
        if not args.model_path:
            print("Error: --model_path is required for eval_mag mode")
            exit(1)
        if args.model_type in [
            "phasenet_mag",
            "amag_v2",
            "eqtransformer_mag",
            "vit_mag",
        ]:
            model_params = extract_model_params(args, args.model_type)
            # Use smaller batch size for transformer models due to complexity
            batch_size = (
                64 if args.model_type in ["eqtransformer_mag", "vit_mag"] else 256
            )
            evaluate_magnitude_unified(
                data,
                args.model_type,
                args.model_path,
                plot_examples=args.plot,
                num_examples=5,
                batch_size=batch_size,
                **model_params,
            )
        else:
            print(f"Error: {args.model_type} is not a valid magnitude model type")
            print(
                "Valid magnitude model types: phasenet_mag, amag_v2, eqtransformer_mag, vit_mag"
            )
            exit(1)
    elif args.mode == "tutorial":
        tutorial_tests(data, model_path=args.model_path)
    elif args.mode == "tutorial_test_load_and_generator":
        tutorial_test_load_and_generator(data)
    elif args.mode == "tutorial_train_phasenet":
        # Use the unified training function for tutorial PhaseNet training
        model_params = extract_model_params(args, "phasenet")
        train_phase_unified(data, "phasenet", epochs=args.epochs, **model_params)
    elif args.mode == "tutorial_evaluate_phasenet":
        if not args.model_path:
            print("Error: --model_path is required for tutorial_evaluate_phasenet mode")
            print("Usage: --model_path path/to/model.pt")
            exit(1)
        # Use the unified evaluation function for tutorial PhaseNet evaluation
        model_params = extract_model_params(args, "phasenet")
        evaluate_phase_unified(data, "phasenet", args.model_path, **model_params)
    elif args.mode == "plot_history":
        if not args.model_path:
            print("Error: --model_path is required for plot_history mode")
            print("Usage: --model_path path/to/training_history_*.pt")
            exit(1)
        plot_training_history(args.model_path, show_plot=args.plot)
    else:
        print(f"Unknown mode: {args.mode}")

    print("=" * 50 + "\n")
