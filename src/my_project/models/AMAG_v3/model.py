import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import seisbench.util as sbu
from seisbench.models.base import WaveformModel


class MagnitudeNet(WaveformModel):
    """
    U-Net based magnitude prediction model with LSTM and attention bottleneck.
    Modified from PhaseNet architecture to predict earthquake magnitude from waveforms.

    :param in_channels: Number of input channels (default: 3 for ENZ components)
    :param filter_factor: Multiplier for number of filters in each layer
    :param lstm_hidden: Hidden size for LSTM layer
    :param lstm_layers: Number of LSTM layers
    :param dropout: Dropout rate for regularization
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)

    def __init__(
        self,
        in_channels=3,
        sampling_rate=100,
        filter_factor: int = 1,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        dropout: float = 0.2,
        **kwargs,
    ):
        citation = (
            "Modified from PhaseNet (Zhu & Beroza, 2019) for magnitude prediction"
        )

        super().__init__(
            citation=citation,
            in_samples=3000,
            output_type="array",
            pred_sample=(0, 3000),
            labels=["magnitude"],
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.filter_factor = filter_factor
        self.lstm_hidden = lstm_hidden
        self.lstm_layers = lstm_layers
        self.dropout_rate = dropout
        self.depth = 5
        self.kernel_size = 7
        self.stride = 2  # Changed from 4 to 2 for better temporal resolution
        self.filters_root = 8
        self.activation = torch.relu

        # Initial convolution
        self.inc = nn.Conv1d(
            self.in_channels,
            self.filters_root * filter_factor,
            self.kernel_size,
            padding="same",
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root * filter_factor, eps=1e-3)

        # Encoder branch
        self.down_branch = nn.ModuleList()
        last_filters = self.filters_root * filter_factor

        for i in range(self.depth):
            filters = int(2**i * self.filters_root) * filter_factor
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)

            if i == self.depth - 1:
                # Bottom of U-Net - no downsampling
                conv_down = None
                bn2 = None
            else:
                # Use stride=2 for all downsampling layers
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=self.kernel_size // 2,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        # Bottleneck: LSTM + Attention
        self.bottleneck_filters = (
            int(2 ** (self.depth - 1) * self.filters_root) * filter_factor
        )

        # LSTM expects (batch, seq, features), so we'll need to transpose
        self.lstm = nn.LSTM(
            input_size=self.bottleneck_filters,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            dropout=dropout if lstm_layers > 1 else 0,
            bidirectional=True,
        )

        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=lstm_hidden * 2,  # *2 for bidirectional
            num_heads=4,
            dropout=dropout,
            batch_first=True,
        )

        # Project back to bottleneck filter size
        self.bottleneck_proj = nn.Linear(lstm_hidden * 2, self.bottleneck_filters)
        self.bottleneck_bn = nn.BatchNorm1d(self.bottleneck_filters, eps=1e-3)
        self.dropout = nn.Dropout(dropout)

        # Scalar output head - Global Average Pooling + MLP
        # Pool the bottleneck features to get single magnitude prediction
        self.scalar_head = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Pool temporal dimension: (batch, filters, time) -> (batch, filters, 1)
            nn.Flatten(),  # (batch, filters, 1) -> (batch, filters)
            nn.Linear(self.bottleneck_filters, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1)  # Single scalar output
        )

    def forward(self, x, logits=False):
        # Initial convolution
        x = self.activation(self.in_bn(self.inc(x)))

        # Encoder
        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                x = self.activation(bn2(conv_down(x)))

        # Bottleneck: LSTM + Attention
        # x shape: (batch, channels, time)
        batch_size, channels, time_steps = x.shape

        # Transpose for LSTM: (batch, time, channels)
        x = x.transpose(1, 2)

        # LSTM
        x, _ = self.lstm(x)

        # Attention (self-attention)
        x, _ = self.attention(x, x, x)

        # Project back to bottleneck size and apply dropout
        x = self.bottleneck_proj(x)
        x = self.dropout(x)

        # Transpose back: (batch, channels, time)
        x = x.transpose(1, 2)
        x = self.activation(self.bottleneck_bn(x))

        # Scalar output head
        # x shape: (batch, bottleneck_filters, time)
        scalar_output = self.scalar_head(x)  # (batch, 1)
        scalar_output = scalar_output.squeeze(-1)  # (batch,)

        return scalar_output

    @staticmethod
    def _merge_skip(skip, x):
        """Merge skip connection with upsampled features."""
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]
        return torch.cat([skip, x_resize], dim=1)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """Normalize input waveforms."""
        # Remove mean
        batch = batch - batch.mean(axis=-1, keepdims=True)

        # Normalize by standard deviation
        std = batch.std(axis=-1, keepdims=True)
        batch = batch / (std + 1e-10)

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """Post-process predictions - no-op for scalar output."""
        return batch

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Aggregate magnitude predictions.

        :param annotations: Model predictions
        :param argdict: Additional arguments
        :return: Classification output with magnitude estimate
        """
        # Average magnitude over time or take max/median depending on use case
        # This is a simple placeholder - adjust based on your needs
        return sbu.ClassifyOutput(self.name, picks=sbu.PickList())

    def get_model_args(self):
        """Get model configuration arguments."""
        model_args = super().get_model_args()
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
        ]:
            del model_args[key]

        model_args["in_channels"] = self.in_channels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["filter_factor"] = self.filter_factor
        model_args["lstm_hidden"] = self.lstm_hidden
        model_args["lstm_layers"] = self.lstm_layers
        model_args["dropout"] = self.dropout_rate

        return model_args
