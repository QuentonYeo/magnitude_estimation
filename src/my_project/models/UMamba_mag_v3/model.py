"""
UMamba Magnitude Estimation V3 - Triple-Head Architecture with Multi-Scale Fusion

Encoder-only architecture with:
- Primary head: Multi-scale fusion + pooling → MLP → scalar magnitude
- Auxiliary head: 1x1 conv → per-timestep magnitude predictions  
- Uncertainty head: Pooling → Linear → log variance (confidence estimate)

Features:
- Multi-scale feature fusion from all encoder stages
- Configurable pooling (avg, max, or hybrid)
- Uncertainty-weighted loss (Kendall & Gal, 2017)
- Flexible inference: extract any subset of heads
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Union, List, Tuple, Type, Optional
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


class HybridPooling(nn.Module):
    """
    Learnable hybrid pooling: α * max_pool + (1-α) * avg_pool
    
    The model learns the optimal blend between max and average pooling.
    Max pooling captures peak amplitudes (important for magnitude).
    Average pooling captures overall energy.
    """
    def __init__(self):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, x):
        """
        Args:
            x: (batch, channels, temporal) tensor
        Returns:
            pooled: (batch, channels) tensor
        """
        avg_pool = x.mean(dim=-1)
        max_pool = x.max(dim=-1)[0]
        alpha = torch.sigmoid(self.alpha)  # Ensure 0 <= alpha <= 1
        return alpha * max_pool + (1 - alpha) * avg_pool


class MultiScaleScalarHead(nn.Module):
    """
    Multi-scale feature fusion for scalar magnitude prediction.
    
    Concatenates pooled features from ALL encoder stages to capture
    multi-scale information (high-freq P-waves + low-freq energy).
    
    Args:
        stage_channels: List of channel dimensions per stage [8, 16, 32, 64]
        hidden_dims: Hidden dimensions for MLP [192, 96]
        pooling_type: 'avg', 'max', or 'hybrid'
        dropout: Dropout rate (default 0.25)
    """
    def __init__(
        self,
        stage_channels: List[int],
        hidden_dims: List[int] = [192, 96],
        pooling_type: str = "max",
        dropout: float = 0.25,
    ):
        super().__init__()
        self.stage_channels = stage_channels
        self.pooling_type = pooling_type
        
        # Pooling layer
        if pooling_type == "avg":
            self.pool = lambda x: x.mean(dim=-1)
        elif pooling_type == "max":
            self.pool = lambda x: x.max(dim=-1)[0]
        elif pooling_type == "hybrid":
            self.pool = HybridPooling()
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # MLP for scalar prediction
        total_channels = sum(stage_channels)
        self.mlp = nn.Sequential(
            nn.Linear(total_channels, hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dims[1], 1),
        )
    
    def forward(self, stage_features: List[torch.Tensor]) -> torch.Tensor:
        """
        Args:
            stage_features: List of tensors [(B,C1,T1), (B,C2,T2), ..., (B,C4,T4)]
        
        Returns:
            scalar: (B,) scalar magnitude predictions
        """
        # Pool each stage independently
        pooled_features = []
        for feat in stage_features:
            pooled = self.pool(feat)  # (B, C_i)
            pooled_features.append(pooled)
        
        # Concatenate multi-scale features
        multi_scale = torch.cat(pooled_features, dim=1)  # (B, sum(C_i))
        
        # Predict scalar magnitude
        scalar = self.mlp(multi_scale).squeeze(-1)  # (B,)
        
        return scalar


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
        conv_op: Type[nn.Module],
        input_channels: int,
        output_channels: int,
        norm_op: Type[nn.Module],
        norm_op_kwargs: dict,
        kernel_size: Union[int, List[int], Tuple[int, ...]],
        padding: Union[int, List[int], Tuple[int, ...]],
        stride: Union[int, List[int], Tuple[int, ...]],
        use_1x1conv: bool = False,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = {'inplace': True},
    ):
        super().__init__()
        
        self.conv1 = conv_op(
            input_channels,
            output_channels,
            kernel_size,
            stride,
            padding,
            bias=False
        )
        self.norm1 = norm_op(output_channels, **norm_op_kwargs)
        self.act1 = nonlin(**nonlin_kwargs)
        
        self.conv2 = conv_op(
            output_channels,
            output_channels,
            kernel_size,
            1,
            padding,
            bias=False
        )
        self.norm2 = norm_op(output_channels, **norm_op_kwargs)
        self.act2 = nonlin(**nonlin_kwargs)
        
        if use_1x1conv:
            self.conv_skip = conv_op(input_channels, output_channels, 1, stride, bias=False)
        else:
            self.conv_skip = None

    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.act1(self.norm1(out))
        out = self.conv2(out)
        out = self.norm2(out)
        
        if self.conv_skip is not None:
            residual = self.conv_skip(x)
        
        out += residual
        out = self.act2(out)
        
        return out


class UMambaEncoder(nn.Module):
    """
    U-Mamba style encoder with alternating Mamba and residual blocks.
    """
    def __init__(
        self,
        input_channels: int,
        n_stages: int,
        features_per_stage: Union[int, List[int], Tuple[int, ...]],
        conv_op: Type[nn.Module],
        kernel_sizes: Union[int, List[int], Tuple[int, ...]],
        strides: Union[int, List[int], Tuple[int, ...]],
        n_blocks_per_stage: Union[int, List[int], Tuple[int, ...]],
        norm_op: Type[nn.Module] = nn.BatchNorm1d,
        norm_op_kwargs: dict = None,
        nonlin: Type[nn.Module] = nn.LeakyReLU,
        nonlin_kwargs: dict = None,
    ):
        super().__init__()
        
        if norm_op_kwargs is None:
            norm_op_kwargs = {}
        if nonlin_kwargs is None:
            nonlin_kwargs = {'inplace': True}
        
        if isinstance(features_per_stage, int):
            features_per_stage = [features_per_stage * (2 ** i) for i in range(n_stages)]
        if isinstance(kernel_sizes, int):
            kernel_sizes = [kernel_sizes] * n_stages
        if isinstance(strides, int):
            strides = [strides] * n_stages
        if isinstance(n_blocks_per_stage, int):
            n_blocks_per_stage = [n_blocks_per_stage] * n_stages
        
        self.conv_op = conv_op
        self.n_stages = n_stages
        self.features_per_stage = features_per_stage
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.n_blocks_per_stage = n_blocks_per_stage
        
        # Initial stem
        self.stem = BasicResBlock(
            conv_op=conv_op,
            input_channels=input_channels,
            output_channels=features_per_stage[0],
            norm_op=norm_op,
            norm_op_kwargs=norm_op_kwargs,
            kernel_size=kernel_sizes[0],
            padding=kernel_sizes[0] // 2,
            stride=1,
            use_1x1conv=True,
            nonlin=nonlin,
            nonlin_kwargs=nonlin_kwargs,
        )
        
        # Build stages
        self.stages = nn.ModuleList()
        for stage_idx in range(n_stages):
            stage_modules = []
            
            # Determine input channels for this stage
            if stage_idx == 0:
                input_channels_stage = features_per_stage[0]
            else:
                input_channels_stage = features_per_stage[stage_idx - 1]
            
            output_channels_stage = features_per_stage[stage_idx]
            
            # Downsampling block (first block of stage)
            stage_modules.append(
                BasicResBlock(
                    conv_op=conv_op,
                    input_channels=input_channels_stage,
                    output_channels=output_channels_stage,
                    norm_op=norm_op,
                    norm_op_kwargs=norm_op_kwargs,
                    kernel_size=kernel_sizes[stage_idx],
                    padding=kernel_sizes[stage_idx] // 2,
                    stride=strides[stage_idx],
                    use_1x1conv=True,
                    nonlin=nonlin,
                    nonlin_kwargs=nonlin_kwargs,
                )
            )
            
            # Additional residual blocks
            for _ in range(n_blocks_per_stage[stage_idx] - 1):
                stage_modules.append(
                    BasicResBlock(
                        conv_op=conv_op,
                        input_channels=output_channels_stage,
                        output_channels=output_channels_stage,
                        norm_op=norm_op,
                        norm_op_kwargs=norm_op_kwargs,
                        kernel_size=kernel_sizes[stage_idx],
                        padding=kernel_sizes[stage_idx] // 2,
                        stride=1,
                        use_1x1conv=False,
                        nonlin=nonlin,
                        nonlin_kwargs=nonlin_kwargs,
                    )
                )
            
            # Add Mamba layer at alternating stages
            if stage_idx % 2 == 1:  # Stages 1 and 3
                # Always use patch tokens (spatial positions as tokens)
                # This treats the temporal dimension as the sequence
                stage_modules.append(
                    MambaLayer(
                        dim=output_channels_stage,
                        d_state=16,
                        d_conv=4,
                        expand=2,
                        channel_token=False  # Use patch tokens (standard mode)
                    )
                )
            
            self.stages.append(nn.Sequential(*stage_modules))
    
    def forward(self, x, return_all_stages=False):
        """
        Forward pass through encoder.
        
        Args:
            x: Input tensor (batch, channels, samples)
            return_all_stages: If True, return list of all stage outputs
        
        Returns:
            If return_all_stages=False: Final encoded features
            If return_all_stages=True: List of features from all stages
        """
        x = self.stem(x)
        
        if return_all_stages:
            stage_features = []
            for stage in self.stages:
                x = stage(x)
                stage_features.append(x)
            return stage_features
        else:
            for stage in self.stages:
                x = stage(x)
            return x


class UMambaMag(WaveformModel):
    """
    UMamba V3: Triple-head magnitude estimation with multi-scale fusion.
    
    Architecture:
    - Encoder with Mamba layers (outputs all stage features)
    - Primary head: Multi-scale fusion → MLP → scalar magnitude
    - Auxiliary head: 1x1 conv → per-timestep magnitude (training only)
    - Uncertainty head: Pooling → Linear → log variance (confidence estimate)
    
    Features:
    - Multi-scale feature fusion from all encoder stages
    - Configurable pooling (avg, max, hybrid)
    - Uncertainty-weighted loss for better calibration
    - Flexible inference: extract any subset of heads
    
    Args:
        in_channels: Number of input channels (default: 3 for ENZ)
        sampling_rate: Sampling rate in Hz (default: 100)
        norm: Normalization method ('std' or 'peak')
        n_stages: Number of encoder stages
        features_per_stage: Number of features at each stage
        kernel_size: Convolution kernel size
        strides: Stride at each stage (for downsampling)
        n_blocks_per_stage: Number of residual blocks per stage
        pooling_type: Type of pooling ('avg', 'max', or 'hybrid')
        hidden_dims: Hidden dimensions for scalar head MLP
        dropout: Dropout rate
        scalar_weight: Weight for scalar loss (default 0.7)
        temporal_weight: Weight for temporal loss (default 0.25)
        use_uncertainty: Whether to use uncertainty head (default True)
    """
    
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
        pooling_type: str = "max",
        hidden_dims: List[int] = None,
        dropout: float = 0.25,
        scalar_weight: float = 0.7,
        temporal_weight: float = 0.25,
        use_uncertainty: bool = True,
        **kwargs,
    ):
        citation = (
            "UMamba V3 for magnitude estimation - triple-head architecture "
            "with multi-scale fusion, temporal predictions, and uncertainty estimates"
        )
        
        super().__init__(
            citation=citation,
            in_samples=3001,
            output_type="array",
            pred_sample=(0, 3001),
            labels=["magnitude"],
            sampling_rate=sampling_rate,
            **kwargs,
        )
        
        self.in_channels = in_channels
        self.norm = norm
        self.n_stages = n_stages
        self.pooling_type = pooling_type
        self.dropout_rate = dropout
        self.scalar_weight = scalar_weight
        self.temporal_weight = temporal_weight
        self.use_uncertainty = use_uncertainty
        
        # Default parameters
        if features_per_stage is None:
            features_per_stage = [8, 16, 32, 64]
        if strides is None:
            strides = [2, 2, 2, 2]
        if hidden_dims is None:
            hidden_dims = [192, 96]  # Larger for multi-scale fusion
        
        self.features_per_stage = features_per_stage
        self.strides = strides
        self.hidden_dims = hidden_dims
        
        # Encoder
        self.encoder = UMambaEncoder(
            input_channels=in_channels,
            n_stages=n_stages,
            features_per_stage=features_per_stage,
            conv_op=nn.Conv1d,
            kernel_sizes=kernel_size,
            strides=strides,
            n_blocks_per_stage=n_blocks_per_stage,
            norm_op=nn.BatchNorm1d,
            norm_op_kwargs={'eps': 1e-5, 'affine': True},
            nonlin=nn.LeakyReLU,
            nonlin_kwargs={'inplace': True},
        )
        
        # Primary head: Multi-scale scalar magnitude prediction
        self.scalar_head = MultiScaleScalarHead(
            stage_channels=features_per_stage,
            hidden_dims=hidden_dims,
            pooling_type=pooling_type,
            dropout=dropout,
        )
        
        # Auxiliary head: Temporal magnitude prediction
        self.temporal_head = nn.Conv1d(features_per_stage[-1], 1, kernel_size=1)
        
        # Uncertainty head: Global log variance
        if use_uncertainty:
            uncertainty_linear = nn.Linear(features_per_stage[-1], 1)
            # Initialize with positive bias to prevent collapse
            # Start with log_var ≈ 0, which gives precision = exp(0) = 1
            nn.init.constant_(uncertainty_linear.bias, 0.0)
            nn.init.normal_(uncertainty_linear.weight, mean=0.0, std=0.01)
            
            self.uncertainty_head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                uncertainty_linear,
            )
        else:
            self.uncertainty_head = None
    
    def forward(
        self, 
        x, 
        return_temporal=False, 
        return_uncertainty=False,
        return_all=False
    ):
        """
        Forward pass through the network with flexible output selection.
        
        Args:
            x: Input tensor of shape (batch, channels, samples)
            return_temporal: If True, include temporal predictions
            return_uncertainty: If True, include uncertainty estimates
            return_all: If True, return all three heads (overrides other flags)
        
        Returns:
            Training mode (self.training=True):
                - If use_uncertainty: (scalar, temporal, log_var)
                - Else: (scalar, temporal)
            
            Inference mode (self.training=False):
                - return_all=True: (scalar, temporal, log_var) or (scalar, temporal)
                - return_temporal=True, return_uncertainty=True: (scalar, temporal, log_var)
                - return_temporal=True: (scalar, temporal)
                - return_uncertainty=True: (scalar, log_var)
                - Default: scalar only
        
        Shapes:
            - scalar: (batch,) - single magnitude per waveform
            - temporal: (batch, samples) - magnitude at each timestep
            - log_var: (batch,) - log variance (uncertainty) per waveform
        """
        # Encode - get all stage features for multi-scale fusion
        stage_features = self.encoder(x, return_all_stages=True)  # List[(B,C1,T1), ..., (B,C4,T4)]
        final_features = stage_features[-1]  # (batch, channels, temporal_dim)
        
        # Primary prediction: Scalar magnitude (multi-scale fusion)
        magnitude_scalar = self.scalar_head(stage_features)  # (batch,)
        
        # Determine what to return
        if self.training or return_all or return_temporal or return_uncertainty:
            outputs = [magnitude_scalar]
            
            # Temporal predictions
            if self.training or return_all or return_temporal:
                magnitude_temporal = self.temporal_head(final_features).squeeze(1)  # (batch, temporal_dim)
                
                # Upsample to original length if needed
                if magnitude_temporal.shape[-1] != x.shape[-1]:
                    magnitude_temporal = F.interpolate(
                        magnitude_temporal.unsqueeze(1),
                        size=x.shape[-1],
                        mode='linear',
                        align_corners=False
                    ).squeeze(1)
                
                outputs.append(magnitude_temporal)
            
            # Uncertainty estimates
            if self.use_uncertainty and (self.training or return_all or return_uncertainty):
                log_var = self.uncertainty_head(final_features).squeeze(-1)  # (batch,)
                # Clamp log_var to prevent numerical instability
                # log_var in [-3, 3] gives precision in [exp(-3), exp(3)] ≈ [0.05, 20]
                # This is tighter than [-5, 3] to prevent immediate saturation
                log_var = torch.clamp(log_var, min=-3.0, max=3.0)
                outputs.append(log_var)
            
            return tuple(outputs) if len(outputs) > 1 else outputs[0]
        
        # Default inference: scalar only (fastest)
        return magnitude_scalar
    
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
        
        return batch
