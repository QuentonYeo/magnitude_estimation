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
from my_project.models.phasenet_mag_v2.model import PhaseNetMagv2
from my_project.models.phasenetLSTM.model import PhaseNetLSTM
from my_project.models.phasenetLSTM.modelv2 import PhaseNetConvLSTM
from my_project.models.AMAG_v2.model import MagnitudeNet
from my_project.models.AMAG_v3.model import MagnitudeNet as MagnitudeNetV3
from my_project.models.EQTransformer.model import EQTransformerMag
from my_project.models.EQTransformer_v2.model import EQTransformerMagV2
from my_project.models.ViT.model import ViTMagnitudeEstimator
from my_project.models.UMamba_mag.model import UMambaMag
from my_project.models.UMamba_mag_v2.model import UMambaMag as UMambaMagV2
from my_project.models.UMamba_mag_v3.model import UMambaMag as UMambaMagV3
from my_project.models.MagNet.model import MagNet

# Import unified training and inference functions
from my_project.utils.unified_training import (
    train_phase_model,
    evaluate_phase_model_unified,
    train_magnitude_model,
    evaluate_magnitude_model,
)

from my_project.utils.utils import plot_training_history, plot_snr_distribution, get_mean_snr_series


def extract_model_params(args, model_type):
    """Extract relevant model parameters from command line arguments"""
    params = {}

    # Common parameters
    if hasattr(args, "filter_factor") and args.filter_factor != 1:
        params["filter_factor"] = args.filter_factor
    if hasattr(args, "early_stopping_patience") and args.early_stopping_patience != 5:
        params["early_stopping_patience"] = args.early_stopping_patience

    # Universal training parameters for magnitude models
    if model_type in ["phasenet_mag", "phasenet_mag_v2", "amag_v2", "amag_v3", "eqtransformer_mag", "eqtransformer_mag_v2", "vit_mag", "umamba_mag", "umamba_mag_v2", "umamba_mag_v3", "magnet"]:
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
    
    # PhaseNetMag V2 specific parameters (scalar output)
    elif model_type == "phasenet_mag_v2":
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

    # MagnitudeNet V3 specific parameters (AMAG V3)
    elif model_type == "amag_v3":
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

    # EQTransformerMag V2 specific parameters (scalar head)
    elif model_type == "eqtransformer_mag_v2":
        if hasattr(args, "lstm_blocks") and args.lstm_blocks != 3:
            params["lstm_blocks"] = args.lstm_blocks
        if hasattr(args, "drop_rate") and args.drop_rate != 0.1:
            params["drop_rate"] = args.drop_rate
        if hasattr(args, "norm") and args.norm != "std":
            params["norm"] = args.norm
        # EQTransformerMag V2-specific training parameters
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
        if hasattr(args, "pooling_type") and args.pooling_type != "avg":
            params["pooling_type"] = args.pooling_type
        if hasattr(args, "dropout") and args.dropout != 0.3:
            params["dropout"] = args.dropout
        # Parse list-like UMamba args (features_per_stage, strides, hidden_dims)
        if hasattr(args, "features_per_stage") and args.features_per_stage:
            try:
                params["features_per_stage"] = [int(x.strip()) for x in args.features_per_stage.split(",")]
            except:
                print(f"Warning: Could not parse features_per_stage '{args.features_per_stage}', using default")
        if hasattr(args, "strides") and args.strides:
            try:
                params["strides"] = [int(x.strip()) for x in args.strides.split(",")]
            except:
                print(f"Warning: Could not parse strides '{args.strides}', using default")
        if hasattr(args, "hidden_dims") and args.hidden_dims:
            try:
                params["hidden_dims"] = [int(x.strip()) for x in args.hidden_dims.split(",")]
            except:
                print(f"Warning: Could not parse hidden_dims '{args.hidden_dims}', using default")
        # Keep n_conv_per_stage_decoder for backward compatibility but it's deprecated
        if hasattr(args, "n_conv_per_stage_decoder") and args.n_conv_per_stage_decoder != 2:
            params["n_conv_per_stage_decoder"] = args.n_conv_per_stage_decoder

    # UMamba V2 specific parameters
    elif model_type == "umamba_mag_v2":
        if hasattr(args, "n_stages") and args.n_stages != 4:
            params["n_stages"] = args.n_stages
        if hasattr(args, "kernel_size") and args.kernel_size != 7:
            params["kernel_size"] = args.kernel_size
        if hasattr(args, "n_blocks_per_stage") and args.n_blocks_per_stage != 2:
            params["n_blocks_per_stage"] = args.n_blocks_per_stage
        if hasattr(args, "norm") and args.norm != "std":
            params["norm"] = args.norm
        if hasattr(args, "pooling_type") and args.pooling_type != "avg":
            params["pooling_type"] = args.pooling_type
        if hasattr(args, "dropout") and args.dropout != 0.3:
            params["dropout"] = args.dropout
        # Parse list-like UMamba V2 args
        if hasattr(args, "features_per_stage") and args.features_per_stage:
            try:
                params["features_per_stage"] = [int(x.strip()) for x in args.features_per_stage.split(",")]
            except:
                print(f"Warning: Could not parse features_per_stage '{args.features_per_stage}', using default")
        if hasattr(args, "strides") and args.strides:
            try:
                params["strides"] = [int(x.strip()) for x in args.strides.split(",")]
            except:
                print(f"Warning: Could not parse strides '{args.strides}', using default")
        if hasattr(args, "hidden_dims") and args.hidden_dims:
            try:
                params["hidden_dims"] = [int(x.strip()) for x in args.hidden_dims.split(",")]
            except:
                print(f"Warning: Could not parse hidden_dims '{args.hidden_dims}', using default")

    # UMamba V3 specific parameters (triple-head architecture)
    elif model_type == "umamba_mag_v3":
        if hasattr(args, "n_stages") and args.n_stages != 4:
            params["n_stages"] = args.n_stages
        if hasattr(args, "kernel_size") and args.kernel_size != 7:
            params["kernel_size"] = args.kernel_size
        if hasattr(args, "n_blocks_per_stage") and args.n_blocks_per_stage != 2:
            params["n_blocks_per_stage"] = args.n_blocks_per_stage
        if hasattr(args, "norm") and args.norm != "std":
            params["norm"] = args.norm
        if hasattr(args, "pooling_type") and args.pooling_type != "avg":
            params["pooling_type"] = args.pooling_type
        if hasattr(args, "dropout") and args.dropout != 0.3:
            params["dropout"] = args.dropout
        # Parse list-like UMamba V3 args
        if hasattr(args, "features_per_stage") and args.features_per_stage:
            try:
                params["features_per_stage"] = [int(x.strip()) for x in args.features_per_stage.split(",")]
            except:
                print(f"Warning: Could not parse features_per_stage '{args.features_per_stage}', using default")
        if hasattr(args, "strides") and args.strides:
            try:
                params["strides"] = [int(x.strip()) for x in args.strides.split(",")]
            except:
                print(f"Warning: Could not parse strides '{args.strides}', using default")
        if hasattr(args, "hidden_dims") and args.hidden_dims:
            try:
                params["hidden_dims"] = [int(x.strip()) for x in args.hidden_dims.split(",")]
            except:
                print(f"Warning: Could not parse hidden_dims '{args.hidden_dims}', using default")
        # V3-specific: loss weights and uncertainty head
        if hasattr(args, "scalar_weight") and args.scalar_weight != 0.7:
            params["scalar_weight"] = args.scalar_weight
        if hasattr(args, "temporal_weight") and args.temporal_weight != 0.25:
            params["temporal_weight"] = args.temporal_weight
        if hasattr(args, "use_uncertainty"):
            params["use_uncertainty"] = args.use_uncertainty
        # V3 ablation parameters
        if hasattr(args, "mamba_at_all_stages"):
            params["mamba_at_all_stages"] = args.mamba_at_all_stages
        if hasattr(args, "use_multiscale_fusion"):
            params["use_multiscale_fusion"] = args.use_multiscale_fusion

    # MagNet specific parameters
    elif model_type == "magnet":
        if hasattr(args, "lstm_hidden") and args.lstm_hidden != 100:
            params["lstm_hidden"] = args.lstm_hidden
        if hasattr(args, "dropout") and args.dropout != 0.2:
            params["dropout"] = args.dropout
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
    elif model_type == "phasenet_mag_v2":
        # Extract PhaseNetMag V2-specific parameters with defaults
        filter_factor = kwargs.get("filter_factor", 1)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)

        return PhaseNetMagv2(
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
    elif model_type == "amag_v3":
        # Extract MagnitudeNet V3-specific parameters with defaults
        filter_factor = kwargs.get("filter_factor", 1)
        lstm_hidden = kwargs.get("lstm_hidden", 128)
        lstm_layers = kwargs.get("lstm_layers", 2)
        dropout = kwargs.get("dropout", 0.2)

        return MagnitudeNetV3(
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
    elif model_type == "eqtransformer_mag_v2":
        # Extract EQTransformerMag V2-specific parameters with defaults
        lstm_blocks = kwargs.get("lstm_blocks", 3)
        drop_rate = kwargs.get("drop_rate", 0.1)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)

        return EQTransformerMagV2(
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
        transformer_mlp_dim1 = kwargs.get("transformer_mlp_dim1", 200)
        transformer_mlp_dim2 = kwargs.get("transformer_mlp_dim2", 100)
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
            transformer_mlp_dim1=transformer_mlp_dim1,
            transformer_mlp_dim2=transformer_mlp_dim2,
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
        n_conv_per_stage_decoder = kwargs.get("n_conv_per_stage_decoder", 2)  # Deprecated
        deep_supervision = kwargs.get("deep_supervision", False)  # Deprecated
        pooling_type = kwargs.get("pooling_type", "avg")
        hidden_dims = kwargs.get("hidden_dims", [128, 64])
        dropout = kwargs.get("dropout", 0.3)
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
            n_conv_per_stage_decoder=n_conv_per_stage_decoder,  # Deprecated but kept for compatibility
            deep_supervision=deep_supervision,  # Deprecated but kept for compatibility
            pooling_type=pooling_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif model_type == "umamba_mag_v2":
        # Extract UMamba V2-specific parameters with defaults
        n_stages = kwargs.get("n_stages", 4)
        features_per_stage = kwargs.get("features_per_stage", [8, 16, 32, 64])
        kernel_size = kwargs.get("kernel_size", 7)
        strides = kwargs.get("strides", [2, 2, 2, 2])
        n_blocks_per_stage = kwargs.get("n_blocks_per_stage", 2)
        pooling_type = kwargs.get("pooling_type", "avg")
        hidden_dims = kwargs.get("hidden_dims", [128, 64])
        dropout = kwargs.get("dropout", 0.3)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)

        return UMambaMagV2(
            in_channels=in_channels,
            sampling_rate=sampling_rate,
            norm=norm,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_size=kernel_size,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            pooling_type=pooling_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
        )
    elif model_type == "umamba_mag_v3":
        # Extract UMamba V3-specific parameters with defaults
        n_stages = kwargs.get("n_stages", 4)
        features_per_stage = kwargs.get("features_per_stage", [8, 16, 32, 64])
        kernel_size = kwargs.get("kernel_size", 7)
        strides = kwargs.get("strides", [2, 2, 2, 2])
        n_blocks_per_stage = kwargs.get("n_blocks_per_stage", 2)
        pooling_type = kwargs.get("pooling_type", "avg")
        hidden_dims = kwargs.get("hidden_dims", [192, 96])  # Updated default for multi-scale
        dropout = kwargs.get("dropout", 0.3)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)
        # V3-specific: triple-head parameters
        scalar_weight = kwargs.get("scalar_weight", 0.7)
        temporal_weight = kwargs.get("temporal_weight", 0.25)
        use_uncertainty = kwargs.get("use_uncertainty", False)

        return UMambaMagV3(
            in_channels=in_channels,
            sampling_rate=sampling_rate,
            norm=norm,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            kernel_size=kernel_size,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            pooling_type=pooling_type,
            hidden_dims=hidden_dims,
            dropout=dropout,
            scalar_weight=scalar_weight,
            temporal_weight=temporal_weight,
            use_uncertainty=use_uncertainty,
        )
    elif model_type == "magnet":
        # Extract MagNet-specific parameters with defaults
        lstm_hidden = kwargs.get("lstm_hidden", 100)
        dropout = kwargs.get("dropout", 0.2)
        norm = kwargs.get("norm", "std")
        in_channels = kwargs.get("in_channels", 3)
        sampling_rate = kwargs.get("sampling_rate", 100)

        return MagNet(
            in_channels=in_channels,
            sampling_rate=sampling_rate,
            norm=norm,
            lstm_hidden=lstm_hidden,
            dropout=dropout,
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
    elif model_type == "phasenet_mag_v2":
        return f"PhaseNetMagv2_{data_name}"
    elif model_type == "amag_v2":
        return f"magnitudenet_v1"
    elif model_type == "amag_v3":
        return f"AMAG_v3_{data_name}"
    elif model_type == "eqtransformer_mag":
        return f"EQTransformerMag_{data_name}"
    elif model_type == "eqtransformer_mag_v2":
        return f"EQTransformerMagV2_{data_name}"
    elif model_type == "vit_mag":
        return f"ViTMag_{data_name}"
    elif model_type == "umamba_mag":
        return f"UMambaMag_{data_name}"
    elif model_type == "umamba_mag_v2":
        return f"UMambaMag_v2_{data_name}"
    elif model_type == "umamba_mag_v3":
        return f"UMambaMag_v3_{data_name}"
    elif model_type == "magnet":
        return f"MagNet_{data_name}"
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def train_phase_unified(
    data: BenchmarkDataset, model_type: str, epochs: int = 5, device: int = None, quiet: bool = False, **kwargs
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
        device=device,
        quiet=quiet,
        **kwargs,
    )


def train_magnitude_unified(
    data: BenchmarkDataset, model_type: str, epochs: int = 50, device: int = None, quiet: bool = False, **kwargs
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
    elif model_type == "phasenet_mag_v2":
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
    elif model_type == "eqtransformer_mag_v2":
        learning_rate = kwargs.pop("learning_rate", 1e-4)  # Conservative for transformer stability
        batch_size = kwargs.pop("batch_size", 64)  # Match UMamba V3
        optimizer_name = kwargs.get("optimizer_name", "AdamW")
        kwargs.setdefault("scheduler_factor", 0.5)
        kwargs.setdefault("gradient_clip", 1.0)
        kwargs.setdefault("warmup_epochs", 5)
        kwargs.setdefault("early_stopping_patience", 15)
    elif model_type == "vit_mag":
        # UPDATED: Paper specifications with stability improvements
        # - Reduced patience from 10 to 5 epochs for faster response to instability
        # - Factor changed from √0.1 (0.316) to 0.5 for less aggressive reduction
        # - Added threshold to avoid noise-triggered reductions
        learning_rate = kwargs.pop("learning_rate", 1e-3)  # Paper: 0.001
        batch_size = kwargs.pop("batch_size", 64)  # Smaller batch for memory efficiency
        optimizer_name = kwargs.get("optimizer_name", "Adam")  # Paper uses Adam
        kwargs.setdefault("weight_decay", 0.0)  # Not specified in paper
        kwargs.setdefault("scheduler_factor", 0.5)  # UPDATED: was √0.1 ≈ 0.316
        kwargs.setdefault("scheduler_patience", 5)  # UPDATED: was 10 epochs
        kwargs.setdefault("scheduler_threshold", 1e-4)  # NEW: ignore small fluctuations
        kwargs.setdefault("gradient_clip", 1.0)
        kwargs.setdefault("warmup_epochs", 0)  # Not mentioned in paper
        kwargs.setdefault("early_stopping_patience", 25)  # Paper: 25 epochs
    elif model_type in ["umamba_mag", "umamba_mag_v2", "umamba_mag_v3"]:
        learning_rate = kwargs.pop("learning_rate", 1e-3)
        batch_size = kwargs.pop("batch_size", 64)
        optimizer_name = kwargs.get("optimizer_name", "AdamW")
        kwargs.setdefault("scheduler_factor", 0.5)
        kwargs.setdefault("gradient_clip", 1.0)
        kwargs.setdefault("early_stopping_patience", 15)
        kwargs.setdefault("warmup_epochs", 5)
    elif model_type == "magnet":
        # MagNet defaults (same as UMamba v3 methodology)
        learning_rate = kwargs.pop("learning_rate", 1e-3)
        batch_size = kwargs.pop("batch_size", 64)
        optimizer_name = kwargs.get("optimizer_name", "AdamW")
        kwargs.setdefault("scheduler_factor", 0.5)
        kwargs.setdefault("gradient_clip", 1.0)
        kwargs.setdefault("early_stopping_patience", 15)
        kwargs.setdefault("warmup_epochs", 5)
    else:
        # Default values for other magnitude models
        learning_rate = kwargs.pop("learning_rate", 1e-3)
        batch_size = kwargs.pop("batch_size", 256)
        optimizer_name = kwargs.get("optimizer_name", "Adam")
        kwargs.setdefault("gradient_clip", 1.0)
        kwargs.setdefault("warmup_epochs", 5)

    return train_magnitude_model(
        model=model,
        model_name=model_name,
        data=data,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        optimizer_name=optimizer_name,
        device=device,
        quiet=quiet,
        **kwargs,
    )


def evaluate_phase_unified(
    data: BenchmarkDataset, model_type: str, model_path: str, device: int = None, **kwargs
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
        model=model, model_path=model_path, data=data, device=device, **kwargs
    )


def evaluate_magnitude_unified(
    data: BenchmarkDataset, model_type: str, model_path: str, device: int = None, **kwargs
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
        model=model, model_path=model_path, data=data, device=device, **eval_kwargs
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
            "plot_snr",
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
            "phasenet_mag_v2",
            "amag_v2",
            "amag_v3",
            "eqtransformer_mag",
            "eqtransformer_mag_v2",
            "vit_mag",
            "umamba_mag",
            "umamba_mag_v2",
            "umamba_mag_v3",
            "magnet",
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
    parser.add_argument(
        "--num_examples",
        type=int,
        default=5,
        help="Number of example plots to generate during evaluation (for eval_mag mode with --plot enabled)",
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
        default=5,
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
    parser.add_argument(
        "--features_per_stage",
        type=str,
        default="8,16,32,64",
        help="Comma-separated list of feature sizes per UMamba stage (e.g. 8,16,32,64)",
    )
    parser.add_argument(
        "--strides",
        type=str,
        default="2,2,2,2",
        help="Comma-separated list of strides per UMamba stage (e.g. 2,2,2,2)",
    )
    parser.add_argument(
        "--n_conv_per_stage_decoder",
        type=int,
        default=2,
        help="Number of conv layers per decoder stage in UMamba model (V1 only)",
    )
    parser.add_argument(
        "--pooling_type",
        type=str,
        default="avg",
        choices=["avg", "max"],
        help="Pooling type for UMamba V2 model (avg or max)",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="192,96",
        help="Comma-separated list of hidden dimensions for UMamba V2/V3 regression head (e.g. 192,96 for V3 multi-scale, 128,64 for V2)",
    )
    parser.add_argument(
        "--scalar_weight",
        type=float,
        default=0.7,
        help="Weight for scalar magnitude loss in UMamba V3 training (recommended: 0.6-0.8). Combined with temporal_weight should be ≤ 1.0",
    )
    parser.add_argument(
        "--temporal_weight",
        type=float,
        default=0.25,
        help="Weight for temporal magnitude loss in UMamba V3 training (recommended: 0.2-0.4). Combined with scalar_weight should be ≤ 1.0",
    )
    parser.add_argument(
        "--use_uncertainty",
        action="store_true",
        help="Enable uncertainty head for automatic sample weighting in UMamba V3 (Kendall & Gal 2017). Recommended for large datasets (>100K samples)",
    )
    parser.add_argument(
        "--mamba_at_all_stages",
        action="store_true",
        help="[UMamba V3 Ablation] Add Mamba layers at ALL encoder stages instead of alternating (stages 1,3). Tests computational vs. performance trade-off",
    )
    parser.add_argument(
        "--use_multiscale_fusion",
        action="store_true",
        default=True,
        help="[UMamba V3 Ablation] Enable multi-scale feature fusion from all stages. If disabled (--no-use_multiscale_fusion), only uses final stage features",
    )
    parser.add_argument(
        "--no-use_multiscale_fusion",
        dest="use_multiscale_fusion",
        action="store_false",
        help="[UMamba V3 Ablation] Disable multi-scale fusion, use only final stage features (single-scale baseline)",
    )
    parser.add_argument(
        "--cuda",
        type=int,
        default=None,
        help="CUDA device number to use (e.g., 0, 1). If not provided, uses model.to_preferred_device()",
    )
    parser.add_argument(
        "--snr_threshold",
        type=float,
        default=None,
        help="SNR threshold in dB for filtering STEAD dataset (e.g., 10.0). Only traces with SNR >= threshold will be used. If not provided, uses all traces.",
    )
    parser.add_argument('--quiet', action='store_true', 
                    help='Disable tqdm progress bars for cleaner logs')
    parser.add_argument(
        "--resume_checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint file (.pt) to resume training from. Only supported for EQTransformerMagV2. The model will continue training from the checkpoint's epoch and save to the same directory.",
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
            
            # Apply SNR filtering if threshold is provided
            if args.snr_threshold is not None:
                print(f"\nApplying SNR filter: mean(trace_snr_db) >= {args.snr_threshold} dB")
                print(f"Dataset size before filtering: {len(data)} samples")

                mean_snr = get_mean_snr_series(data)
                mask = mean_snr >= args.snr_threshold
                # data.filter expects an array-like boolean mask aligned with metadata
                data.filter(mask.values)

                print(f"Dataset size after filtering: {len(data)} samples")
                print(f"Filtered out {int((~mask).sum())} samples with low SNR")
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

    # If requested, plot SNR distribution for the loaded dataset (don't exit)
    if args.mode == "plot_snr":
        try:
            out = plot_snr_distribution(data)
            print(f"SNR plot saved to: {out}")
            exit(0)
        except Exception as e:
            print(f"Error while plotting SNR distribution: {e}")

    # Run appropriate workflow
    if args.mode == "train_phase":
        if args.model_type in ["phasenet", "phasenet_lstm", "phasenet_conv_lstm"]:
            model_params = extract_model_params(args, args.model_type)
            train_phase_unified(
                data, args.model_type, epochs=args.epochs, device=args.cuda, quiet=args.quiet, **model_params
            )
        else:
            print(f"Error: {args.model_type} is not a valid phase model type")
            print(
                "Valid phase model types: phasenet, phasenet_lstm, phasenet_conv_lstm"
            )
            exit(1)
    elif args.mode == "train_mag":
        # Validate resume_checkpoint usage
        if args.resume_checkpoint and args.model_type != "eqtransformer_mag_v2":
            print(f"Error: --resume_checkpoint is currently only supported for eqtransformer_mag_v2")
            print(f"Requested model type: {args.model_type}")
            exit(1)
        
        if args.model_type in [
            "phasenet_mag",
            "phasenet_mag_v2",
            "amag_v2",
            "amag_v3",
            "eqtransformer_mag",
            "eqtransformer_mag_v2",
            "vit_mag",
            "umamba_mag",
            "umamba_mag_v2",
            "umamba_mag_v3",
            "magnet",
        ]:
            model_params = extract_model_params(args, args.model_type)
            # Add checkpoint_path to model_params if provided
            if args.resume_checkpoint:
                model_params["checkpoint_path"] = args.resume_checkpoint
            train_magnitude_unified(
                data, args.model_type, epochs=args.epochs, device=args.cuda, quiet=args.quiet, **model_params
            )
        else:
            print(f"Error: {args.model_type} is not a valid magnitude model type")
            print(
                "Valid magnitude model types: phasenet_mag, amag_v2, eqtransformer_mag, vit_mag, umamba_mag, umamba_mag_v2, umamba_mag_v3"
            )
            exit(1)
    elif args.mode == "eval_phase":
        if not args.model_path:
            print("Error: --model_path is required for eval_phase mode")
            exit(1)
        if args.model_type in ["phasenet", "phasenet_lstm", "phasenet_conv_lstm"]:
            model_params = extract_model_params(args, args.model_type)
            evaluate_phase_unified(
                data, args.model_type, args.model_path, device=args.cuda, **model_params
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
            "phasenet_mag_v2",
            "amag_v2",
            "amag_v3",
            "eqtransformer_mag",
            "eqtransformer_mag_v2",
            "vit_mag",
            "umamba_mag",
            "umamba_mag_v2",
            "umamba_mag_v3",
            "magnet",
        ]:
            model_params = extract_model_params(args, args.model_type)
            # Choose sensible defaults per model type to avoid OOMs
            if "batch_size" not in model_params:
                if args.model_type in ["eqtransformer_mag", "vit_mag", "umamba_mag", "umamba_mag_v2", "umamba_mag_v3"]:
                    model_params["batch_size"] = 64
                else:
                    model_params["batch_size"] = 256

            # Use the unified evaluation path for all magnitude models (UMamba included)
            evaluate_magnitude_unified(
                data,
                args.model_type,
                args.model_path,
                device=args.cuda,
                plot_examples=args.plot,
                num_examples=args.num_examples,
                **model_params,
            )
        else:
            print(f"Error: {args.model_type} is not a valid magnitude model type")
            print(
                "Valid magnitude model types: phasenet_mag, amag_v2, eqtransformer_mag, vit_mag, umamba_mag, umamba_mag_v2, umamba_mag_v3"
            )
            exit(1)
    elif args.mode == "tutorial":
        tutorial_tests(data, model_path=args.model_path)
    elif args.mode == "tutorial_test_load_and_generator":
        tutorial_test_load_and_generator(data)
    elif args.mode == "tutorial_train_phasenet":
        # Use the unified training function for tutorial PhaseNet training
        model_params = extract_model_params(args, "phasenet")
        train_phase_unified(data, "phasenet", epochs=args.epochs, device=args.cuda, quiet=args.quiet, **model_params)
    elif args.mode == "tutorial_evaluate_phasenet":
        if not args.model_path:
            print("Error: --model_path is required for tutorial_evaluate_phasenet mode")
            print("Usage: --model_path path/to/model.pt")
            exit(1)
        # Use the unified evaluation function for tutorial PhaseNet evaluation
        model_params = extract_model_params(args, "phasenet")
        evaluate_phase_unified(data, "phasenet", args.model_path, device=args.cuda, **model_params)
    elif args.mode == "plot_history":
        if not args.model_path:
            print("Error: --model_path is required for plot_history mode")
            print("Usage: --model_path path/to/training_history_*.pt")
            exit(1)
        plot_training_history(args.model_path, show_plot=args.plot)
    elif args.mode == "plot_snr":
        # Load dataset for SNR plotting
        print("\n" + "=" * 50)
        print("LOADING DATA FOR SNR ANALYSIS")
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
        plot_snr_distribution(data)
    else:
        print(f"Unknown mode: {args.mode}")

    print("=" * 50 + "\n")
