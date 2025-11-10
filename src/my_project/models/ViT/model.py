import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Any

import seisbench.util as sbu
from seisbench.models.base import WaveformModel

# ====================================================================
# COMPLETE ViT MAGNITUDE ESTIMATION MODEL
# Based on Figure 1c from the paper
# Converted to SeisBench WaveformModel for 3001 samples
# ====================================================================


class SelfAttention(nn.Module):
    """Standard Self-Attention block (Figure 1h)"""

    def __init__(self, embed_dim, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.W_q = nn.Linear(embed_dim, embed_dim)
        self.W_k = nn.Linear(embed_dim, embed_dim)
        self.W_v = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(
            self.embed_dim
        )
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        attention_output = torch.matmul(attention_weights, V)

        return attention_output


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention (Figure 1g)"""

    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.attention_heads = nn.ModuleList(
            [SelfAttention(embed_dim, dropout) for _ in range(num_heads)]
        )

        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attention_outputs = [head(x) for head in self.attention_heads]
        output = torch.stack(attention_outputs, dim=0).mean(dim=0)
        output = self.out_proj(output)
        output = self.dropout(output)
        return output


class MLP(nn.Module):
    """MLP block with GELU activation"""

    def __init__(self, embed_dim, hidden_dim, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerEncoder(nn.Module):
    """Transformer Encoder Block (Figure 1f)"""

    def __init__(self, embed_dim, num_heads=4, mlp_hidden_dim=200, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.msa = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.mlp = MLP(embed_dim, mlp_hidden_dim, dropout)

    def forward(self, x):
        # Norm -> MSA -> Residual
        residual = x
        x = self.norm1(x)
        x = self.msa(x)
        x = x + residual

        # Norm -> MLP -> Residual
        residual = x
        x = self.norm2(x)
        x = self.mlp(x)
        x = x + residual

        return x


class ConvBlock(nn.Module):
    """Convolution block with linear activation to preserve amplitude"""

    def __init__(
        self, in_channels, out_channels, kernel_size=3, pool_size=2, dropout=0.1
    ):
        super().__init__()
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2
        )
        self.dropout = nn.Dropout(dropout)
        self.pool = nn.MaxPool1d(kernel_size=pool_size)

    def forward(self, x):
        x = self.conv(x)  # Linear activation (no non-linearity)
        x = self.dropout(x)
        x = self.pool(x)
        return x


class PatchEmbedding(nn.Module):
    """
    Patch embedding with positional encoding

    MPS-compatible implementation using reshape instead of unfold.
    The paper describes dividing input into non-overlapping patches of size L.
    """

    def __init__(self, input_dim=75, num_channels=32, patch_size=5, embed_dim=100):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = input_dim // patch_size
        patch_dim = num_channels * patch_size

        self.projection = nn.Linear(patch_dim, embed_dim)
        self.position_embedding = nn.Parameter(
            torch.randn(1, self.num_patches, embed_dim)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, channels, length) e.g., (batch, 32, 75)
        Returns:
            embedded: (batch, num_patches, embed_dim) e.g., (batch, 15, 100)
        """
        batch_size, channels, length = x.shape

        # MPS-compatible patching using reshape
        # (batch, channels, length) -> (batch, channels, num_patches, patch_size)
        x = x.reshape(batch_size, channels, self.num_patches, self.patch_size)

        # Rearrange to (batch, num_patches, channels, patch_size)
        x = x.permute(0, 2, 1, 3)

        # Flatten each patch: (batch, num_patches, channels * patch_size)
        x = x.reshape(batch_size, self.num_patches, -1)

        # Apply learnable embedding projection
        x = self.projection(x)  # (batch, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.position_embedding

        return x


class ViTMagnitudeEstimator(WaveformModel):
    """
    Complete ViT Magnitude Estimation Model (Figure 1c)
    Converted to SeisBench WaveformModel interface

    Architecture from paper:
    - Input: 30s, 3-component seismograms (3001 samples @ 100Hz)
    - 4 Convolution blocks with pooling rates [2, 2, 2, 5]
    - Patch size: 5, Num patches: 15
    - Projection dimension: 100
    - 4 Transformer encoders with 4 attention heads
    - Final MLP: [1000, 500] neurons
    - Output: Single magnitude value
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)

    def __init__(
        self,
        in_channels=3,
        sampling_rate=100,
        conv_channels=[64, 32, 32, 32],
        kernel_size=3,
        pool_sizes=[2, 2, 2, 5],
        patch_size=5,
        embed_dim=100,
        num_transformer_blocks=4,
        num_heads=4,
        transformer_mlp_dim=200,
        final_mlp_dims=[1000, 500],
        dropout=0.1,
        final_dropout=0.5,
        norm="std",
        **kwargs,
    ):
        citation = (
            "Vision Transformer for Seismic Magnitude Estimation. "
            "Based on Transformer architecture applied to seismic waveforms."
        )

        # Handle PickBlue options like other models
        for option in ("norm_amp_per_comp", "norm_detrend"):
            if option in kwargs:
                setattr(self, option, kwargs[option])
                del kwargs[option]
            else:
                setattr(self, option, False)

        super().__init__(
            citation=citation,
            in_samples=3001,  # 30 seconds at 100Hz
            output_type="scalar",  # Changed from "array" - this is magnitude regression
            pred_sample=(0, 3001),
            labels=["magnitude"],  # Single output for magnitude
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_size = kernel_size
        self.pool_sizes = pool_sizes
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_transformer_blocks = num_transformer_blocks
        self.num_heads = num_heads
        self.transformer_mlp_dim = transformer_mlp_dim
        self.final_mlp_dims = final_mlp_dims
        self.dropout = dropout
        self.final_dropout = final_dropout
        self.norm = norm

        # Calculate input length for conv output calculation
        input_length = 3001

        # ============================================================
        # CONVOLUTIONAL FEATURE EXTRACTION (Block A in Figure 1c)
        # ============================================================
        self.conv_blocks = nn.ModuleList()
        in_ch = in_channels

        for out_ch, pool in zip(conv_channels, pool_sizes):
            self.conv_blocks.append(
                ConvBlock(in_ch, out_ch, kernel_size, pool, dropout)
            )
            in_ch = out_ch

        # Calculate output dimension after convolutions
        # 3001 -> 1500 -> 750 -> 375 -> 75 (approximately)
        conv_output_dim = input_length
        for pool in pool_sizes:
            conv_output_dim = conv_output_dim // pool

        # ============================================================
        # PATCH EMBEDDING
        # ============================================================
        self.patch_embed = PatchEmbedding(
            input_dim=conv_output_dim,
            num_channels=conv_channels[-1],
            patch_size=patch_size,
            embed_dim=embed_dim,
        )

        # ============================================================
        # TRANSFORMER ENCODERS (repeated U times, U=4)
        # ============================================================
        self.transformer_encoders = nn.ModuleList(
            [
                TransformerEncoder(
                    embed_dim=embed_dim,
                    num_heads=num_heads,
                    mlp_hidden_dim=transformer_mlp_dim,
                    dropout=dropout,
                )
                for _ in range(num_transformer_blocks)
            ]
        )

        # ============================================================
        # FINAL MLP FOR MAGNITUDE PREDICTION
        # ============================================================
        num_patches = self.patch_embed.num_patches
        flatten_dim = num_patches * embed_dim

        mlp_layers = []
        in_dim = flatten_dim

        for hidden_dim in final_mlp_dims:
            mlp_layers.append(nn.Linear(in_dim, hidden_dim))
            mlp_layers.append(nn.GELU())
            mlp_layers.append(nn.Dropout(final_dropout))
            in_dim = hidden_dim

        self.final_mlp = nn.Sequential(*mlp_layers)

        # Output layer: single neuron with linear activation for magnitude
        self.output = nn.Linear(final_mlp_dims[-1], 1)

    def forward(self, x):
        """
        Args:
            x: (batch, 3, 3001) - 3-component seismograms
        Returns:
            magnitude: (batch, 3001, 1) - magnitude predictions at each time step
        """
        # Convolutional feature extraction
        for conv_block in self.conv_blocks:
            x = conv_block(x)
        # x is now (batch, 32, 75) approximately

        # Patch embedding
        x = self.patch_embed(x)
        # x is now (batch, 15, 100)

        # Transformer encoders
        for transformer in self.transformer_encoders:
            x = transformer(x)
        # x is still (batch, 15, 100)

        # Flatten for final MLP
        x = x.reshape(x.shape[0], -1)
        # x is now (batch, 1500)

        # Final MLP
        x = self.final_mlp(x)

        # Output magnitude
        magnitude = self.output(x)
        # magnitude is (batch, 1)
        
        # Return scalar magnitude (batch,) for proper regression training
        # No need to expand to (batch, 1, 3001) - that's for detection models
        return magnitude.squeeze(-1)  # (batch,)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """Preprocessing: center and normalize the batch."""
        batch = batch - batch.mean(axis=-1, keepdims=True)

        if self.norm == "std":
            batch = batch / (batch.std(axis=-1, keepdims=True) + 1e-10)
        elif self.norm == "peak":
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """Post-processing: return scalar predictions as-is."""
        # For scalar output (magnitude regression), no transposition needed
        # batch is already (batch,) shape
        return batch

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Aggregate magnitude predictions.

        :param annotations: Model predictions
        :param argdict: Additional arguments
        :return: Classification output with magnitude estimate
        """
        # For magnitude models, we typically don't need complex aggregation
        # This is a placeholder implementation similar to other magnitude models
        return sbu.ClassifyOutput(self.name, picks=sbu.PickList())

    def get_model_args(self):
        """Get model arguments for saving/loading."""
        model_args = super().get_model_args()
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

        model_args["in_channels"] = self.in_channels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["conv_channels"] = self.conv_channels
        model_args["kernel_size"] = self.kernel_size
        model_args["pool_sizes"] = self.pool_sizes
        model_args["patch_size"] = self.patch_size
        model_args["embed_dim"] = self.embed_dim
        model_args["num_transformer_blocks"] = self.num_transformer_blocks
        model_args["num_heads"] = self.num_heads
        model_args["transformer_mlp_dim"] = self.transformer_mlp_dim
        model_args["final_mlp_dims"] = self.final_mlp_dims
        model_args["dropout"] = self.dropout
        model_args["final_dropout"] = self.final_dropout
        model_args["norm"] = self.norm
        model_args["norm_amp_per_comp"] = self.norm_amp_per_comp
        model_args["norm_detrend"] = self.norm_detrend

        return model_args
