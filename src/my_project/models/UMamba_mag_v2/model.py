import torch
import torch.nn as nn
import numpy as np
from typing import Any, Union, List, Tuple, Type
from torch.amp import autocast

import seisbench.util as sbu
from seisbench.models.base import WaveformModel
from mamba_ssm import Mamba


class MambaLayer(nn.Module):
    """
    Mamba layer for processing sequential data with state space models.
    
    Args:
        dim: Model dimension
        d_state: SSM state expansion factor
        d_conv: Local convolution width
        expand: Block expansion factor
        channel_token: Whether to use channels as tokens (instead of spatial positions)
    """
    def __init__(self, dim, d_state=16, d_conv=4, expand=2, channel_token=False):
        super().__init__()
        self.dim = dim
        self.norm = nn.LayerNorm(dim)
        self.mamba = Mamba(
            d_model=dim,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        self.channel_token = channel_token

    def forward_patch_token(self, x):
        """Process spatial positions as tokens (standard mode)."""
        B, d_model = x.shape[:2]
        assert d_model == self.dim
        n_tokens = x.shape[2:].numel()
        img_dims = x.shape[2:]
        x_flat = x.reshape(B, d_model, n_tokens).transpose(-1, -2)
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.transpose(-1, -2).reshape(B, d_model, *img_dims)
        return out

    def forward_channel_token(self, x):
        """Process channels as tokens (for small spatial dimensions)."""
        B, n_tokens = x.shape[:2]
        d_model = x.shape[2:].numel()
        assert d_model == self.dim, f"d_model: {d_model}, self.dim: {self.dim}"
        img_dims = x.shape[2:]
        x_flat = x.flatten(2)
        assert x_flat.shape[2] == d_model, f"x_flat.shape[2]: {x_flat.shape[2]}, d_model: {d_model}"
        x_norm = self.norm(x_flat)
        x_mamba = self.mamba(x_norm)
        out = x_mamba.reshape(B, n_tokens, *img_dims)
        return out

    @autocast("cuda", enabled=False)
    def forward(self, x):
        if x.dtype == torch.float16:
            x = x.type(torch.float32)
        
        if self.channel_token:
            out = self.forward_channel_token(x)
        else:
            out = self.forward_patch_token(x)
        return out


class BasicResBlock(nn.Module):
    """
    Basic residual block with two convolutions and optional skip connection.
    
    Args:
        conv_op: Convolution operation (nn.Conv1d for 1D)
        input_channels: Number of input channels
        output_channels: Number of output channels
        norm_op: Normalization operation
        norm_op_kwargs: Normalization operation arguments
        kernel_size: Kernel size for convolutions
        padding: Padding for convolutions
        stride: Stride for first convolution
        use_1x1conv: Whether to use 1x1 conv for skip connection
        nonlin: Non-linearity activation function
        nonlin_kwargs: Non-linearity arguments
    """
    def __init__(
        self,
        conv_op,
        input_channels,
        output_channels,
        norm_op,
        norm_op_kwargs,
        kernel_size=3,
        padding=1,
        stride=1,
        use_1x1conv=False,
        nonlin=nn.LeakyReLU,
        nonlin_kwargs={'inplace': True}
    ):
        super().__init__()
        
        self.conv1 = conv_op(input_channels, output_channels, kernel_size, stride=stride, padding=padding)
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(output_channels, output_channels, kernel_size, padding=padding)
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv3 = conv_op(input_channels, output_channels, kernel_size=1, stride=stride)
        else:
            self.conv3 = None
                  
    def forward(self, x):
        y = self.conv1(x)
        y = self.act1(self.norm1(y))  
        y = self.norm2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        y += x
        return self.act2(y)


class LearnablePooling(nn.Module):
    """
    Learnable pooling that maintains linear time complexity.
    
    Uses element-wise gating (linear time) instead of attention (quadratic).
    Each channel learns its own importance weighting across time.
    
    Args:
        in_channels: Number of input channels
    """
    def __init__(self, in_channels):
        super().__init__()
        # Gate learns channel-wise importance without cross-time attention
        self.gate = nn.Sequential(
            nn.Conv1d(in_channels, in_channels, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: (batch, channels, time)
        # Element-wise gating - O(n) not O(n²)
        weights = self.gate(x)  # (batch, channels, time)
        weighted = x * weights
        # Sum over time dimension
        pooled = weighted.mean(dim=-1)  # (batch, channels)
        return pooled


class UMambaEncoder(nn.Module):
    """
    U-Mamba encoder with residual blocks and Mamba layers.
    
    This encoder progressively downsamples the input while increasing channel depth,
    incorporating Mamba layers at alternating stages for long-range dependency modeling.
    
    Args:
        input_size: Input spatial size (for 1D: (length,))
        input_channels: Number of input channels
        n_stages: Number of downsampling stages
        features_per_stage: Number of features at each stage
        conv_op: Convolution operation (nn.Conv1d)
        kernel_sizes: Kernel sizes for each stage
        strides: Stride values for each stage
        n_blocks_per_stage: Number of residual blocks per stage
        conv_bias: Whether to use bias in convolutions
        norm_op: Normalization operation
        norm_op_kwargs: Normalization arguments
        nonlin: Non-linearity function
        nonlin_kwargs: Non-linearity arguments
        return_skips: Whether to return skip connections
        stem_channels: Number of channels in stem (if None, uses features_per_stage[0])
    """
    def __init__(
        self,
        input_size: Tuple[int, ...],
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[nn.Module],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_bias: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        return_skips: bool = False,
        stem_channels: int = None,
    ):
        super().__init__()
        
        # Convert scalar parameters to lists
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages

        # Calculate padding sizes first (needed for accurate feature map size calculation)
        self.conv_pad_sizes = []
        for krnl in kernel_sizes:
            if isinstance(krnl, int):
                self.conv_pad_sizes.append(krnl // 2)
            else:
                self.conv_pad_sizes.append([i // 2 for i in krnl])

        # Determine where to use channel tokens vs patch tokens
        # Use accurate convolution output size formula: floor((L_in + 2*pad - kernel) / stride) + 1
        do_channel_token = [False] * n_stages
        feature_map_sizes = []
        feature_map_size = list(input_size)
        
        for s in range(n_stages):
            stride = strides[s] if isinstance(strides[s], int) else strides[s][0]
            kernel = kernel_sizes[s] if isinstance(kernel_sizes[s], int) else kernel_sizes[s][0]
            pad = self.conv_pad_sizes[s] if isinstance(self.conv_pad_sizes[s], int) else self.conv_pad_sizes[s][0]
            
            # Calculate output size using convolution formula
            new_size = []
            for i, size_in in enumerate(feature_map_size):
                size_out = (size_in + 2 * pad - kernel) // stride + 1
                new_size.append(size_out)
            
            feature_map_sizes.append(new_size)
            feature_map_size = new_size
            
            # Use channel tokens when spatial dimension is smaller than channel dimension
            if np.prod(feature_map_size) <= features_per_stage[s]:
                do_channel_token[s] = True

        # Stem: initial processing block
        stem_channels = stem_channels if stem_channels is not None else features_per_stage[0]
        self.stem = BasicResBlock(
            conv_op=conv_op,
            input_channels=input_channels,
            output_channels=stem_channels,
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            kernel_size=kernel_sizes[0],
            padding=self.conv_pad_sizes[0],
            stride=1,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
            use_1x1conv=True
        )

        input_channels = stem_channels

        # Build encoder stages
        stages = []
        mamba_layers = []
        
        for s in range(n_stages):
            # Each stage starts with a residual block (potentially with downsampling)
            stage_blocks = [
                BasicResBlock(
                    conv_op=conv_op,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    input_channels=input_channels,
                    output_channels=features_per_stage[s],
                    kernel_size=kernel_sizes[s],
                    padding=self.conv_pad_sizes[s],
                    stride=strides[s],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs
                )
            ]
            
            # Add additional residual blocks
            for _ in range(n_blocks_per_stage[s] - 1):
                stage_blocks.append(
                    BasicResBlock(
                        conv_op=conv_op,
                        input_channels=features_per_stage[s],
                        output_channels=features_per_stage[s],
                        kernel_size=kernel_sizes[s],
                        padding=self.conv_pad_sizes[s],
                        stride=1,
                        use_1x1conv=False,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                )
            
            stage = nn.Sequential(*stage_blocks)
            stages.append(stage)
            
            # Add Mamba layer at alternating stages (ensuring last stage has one)
            if bool(s % 2) ^ bool(n_stages % 2):
                mamba_layers.append(
                    MambaLayer(
                        dim=np.prod(feature_map_sizes[s]) if do_channel_token[s] else features_per_stage[s],
                        channel_token=do_channel_token[s]
                    )
                )
            else:
                mamba_layers.append(nn.Identity())

            input_channels = features_per_stage[s]

        self.stages = nn.ModuleList(stages)
        self.mamba_layers = nn.ModuleList(mamba_layers)
        self.output_channels = features_per_stage
        self.strides = strides
        self.return_skips = return_skips

        # Store architecture parameters
        self.conv_op = conv_op
        self.norm_op = norm_op
        self.norm_op_kwargs = norm_op_kwargs
        self.nonlin = nonlin
        self.nonlin_kwargs = nonlin_kwargs
        self.conv_bias = conv_bias
        self.kernel_sizes = kernel_sizes

    def forward(self, x):
        if self.stem is not None:
            x = self.stem(x)
        ret = []
        for s in range(len(self.stages)):
            x = self.stages[s](x)
            x = self.mamba_layers[s](x)
            ret.append(x)
        if self.return_skips:
            return ret
        else:
            return ret[-1]


class UMambaMag(WaveformModel):
    """
    UMamba architecture adapted for seismic magnitude regression.
    
    This model combines residual blocks with Mamba layers (state space models)
    for processing seismic waveforms. The architecture uses an encoder-only
    structure with global pooling for magnitude regression.
    
    Key features:
    - Residual blocks for local feature extraction
    - Mamba layers for long-range temporal modeling
    - Rooling for temporal aggregation
    - Regression head for magnitude prediction
    
    Args:
        in_channels: Number of input channels (default: 3 for 3-component seismogram)
        sampling_rate: Sampling rate in Hz (default: 100)
        norm: Normalization method - 'std' for standardization, 'peak' for peak normalization, 'none' for no normalization
        n_stages: Number of encoder stages (default: 4)
        features_per_stage: Number of features at each stage (default: [8, 16, 32, 64])
        kernel_size: Kernel size for convolutions (default: 7)
        strides: Downsampling strides for each stage (default: [2, 2, 2, 2])
        n_blocks_per_stage: Number of residual blocks per stage (default: 2)
        pooling_type: Type of pooling - 'avg', or 'max' (default: 'avg')
        hidden_dims: Hidden dimensions for regression head (default: [128, 64])
        dropout: Dropout rate for regression head (default: 0.3)
    """
    
    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)

    def __init__(
        self,
        in_channels: int = 3,
        sampling_rate: int = 100,
        norm: str = "std",
        n_stages: int = 4,
        features_per_stage: List[int] = None,
        kernel_size: int = 7,
        strides: List[int] = None,
        n_blocks_per_stage: int = 2,
        pooling_type: str = "avg",
        hidden_dims: List[int] = None,
        dropout: float = 0.3,
        **kwargs,
    ):
        citation = (
            "UMamba magnitude estimator adapted from: "
            "Ma, J., Li, F., & Wang, B. (2024). U-Mamba: Enhancing Long-range Dependency "
            "for Biomedical Image Segmentation. arXiv:2401.04722. "
            "Combined with Mamba: Gu, A., & Dao, T. (2023). Mamba: Linear-Time Sequence "
            "Modeling with Selective State Spaces. arXiv:2312.00752."
        )

        super().__init__(
            citation=citation,
            in_samples=3001,
            output_type="scalar",  # Fixed: V2 outputs scalar, not array
            pred_sample=(0, 3001),
            labels=["magnitude"],
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.norm = norm
        self.n_stages = n_stages
        self.pooling_type = pooling_type
        self.dropout = dropout

        # Default architecture parameters
        if features_per_stage is None:
            features_per_stage = [8 * (2 ** i) for i in range(n_stages)]
        if strides is None:
            strides = [2] * n_stages
        if hidden_dims is None:
            hidden_dims = [128, 64]

        self.features_per_stage = features_per_stage
        self.kernel_size = kernel_size
        self.strides = strides
        self.n_blocks_per_stage = n_blocks_per_stage
        self.hidden_dims = hidden_dims

        # Build encoder
        self.encoder = UMambaEncoder(
            input_size=(3001,),  # 1D input
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv1d,
            kernel_sizes=kernel_size,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            conv_bias=True,
            norm_op=nn.BatchNorm1d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
            return_skips=False,  # Don't need skips for regression
            stem_channels=features_per_stage[0],
        )

        # Global pooling layer
        encoder_output_channels = features_per_stage[-1]
        if pooling_type == "avg":
            self.pooling = nn.AdaptiveAvgPool1d(1)
        elif pooling_type == "max":
            self.pooling = nn.AdaptiveMaxPool1d(1)
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")

        # Build regression head
        regression_layers = []
        in_features = encoder_output_channels
        
        for hidden_dim in hidden_dims:
            regression_layers.extend([
                nn.Linear(in_features, hidden_dim),
                nn.LeakyReLU(inplace=True),
                nn.Dropout(dropout),
            ])
            in_features = hidden_dim
        
        # Final output layer
        regression_layers.append(nn.Linear(in_features, 1))
        
        self.regression_head = nn.Sequential(*regression_layers)

    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, samples)
            
        Returns:
            Magnitude predictions of shape (batch,)
        """
        # Encode
        features = self.encoder(x)  # (batch, channels, temporal)
        
        # Pool
        pooled = self.pooling(features).squeeze(-1)  # (batch, channels)
        
        # Regress
        magnitude = self.regression_head(pooled)  # (batch, 1)
        
        return magnitude.squeeze(-1)  # (batch,)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Preprocessing: center and optionally normalize the batch.
        
        Options:
        - 'std': Standardize by standard deviation
        - 'peak': Normalize by peak amplitude
        - 'none': Only center (remove mean)
        """
        batch = batch - batch.mean(axis=-1, keepdims=True)

        if self.norm == "std":
            batch = batch / (batch.std(axis=-1, keepdims=True) + 1e-10)
        elif self.norm == "peak":
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)
        elif self.norm == "none":
            pass  # Only centered
        else:
            raise ValueError(f"Unknown normalization: {self.norm}")

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Post-processing: ensure output is in correct shape.
        
        For scalar output, just return as is - shape (batch,)
        """
        return batch

    def get_model_args(self):
        """
        Get model arguments for saving/loading.
        
        Returns a dictionary of arguments needed to reconstruct the model.
        """
        model_args = super().get_model_args()
        
        # Remove base class arguments
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
        ]:
            if key in model_args:
                del model_args[key]

        # Add UMambaMag-specific arguments
        model_args["in_channels"] = self.in_channels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["n_stages"] = self.n_stages
        model_args["features_per_stage"] = self.features_per_stage
        model_args["kernel_size"] = self.kernel_size
        model_args["strides"] = self.strides
        model_args["n_blocks_per_stage"] = self.n_blocks_per_stage
        model_args["pooling_type"] = self.pooling_type
        model_args["hidden_dims"] = self.hidden_dims
        model_args["dropout"] = self.dropout

        return model_args