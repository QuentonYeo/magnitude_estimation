import torch
import torch.nn as nn
import numpy as np
from typing import Any

import seisbench.util as sbu
from seisbench.models.base import Conv1dSame, WaveformModel


class PhaseNetMagv2(WaveformModel):
    """
    PhaseNet architecture modified for scalar magnitude regression.

    This model uses the same U-Net architecture as PhaseNet but outputs a single
    scalar magnitude value per waveform instead of per-sample predictions.
    
    Architecture:
    - U-Net encoder-decoder for feature extraction
    - Global average pooling on final features
    - MLP head for scalar magnitude prediction (1 value per waveform)
    
    The model predicts a single magnitude value per waveform by taking the maximum
    of the labeled magnitude values (which are constant after P-arrival).
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)

    def __init__(
        self,
        in_channels=3,
        sampling_rate=100,
        norm="std",
        filter_factor: int = 1,
        **kwargs,
    ):
        citation = (
            "Modified from PhaseNet: Zhu, W., & Beroza, G. C. (2019). "
            "PhaseNet: a deep-neural-network-based seismic arrival-time picking method. "
            "Geophysical Journal International, 216(1), 261-273. "
            "https://doi.org/10.1093/gji/ggy423"
        )

        super().__init__(
            citation=citation,
            in_samples=3001,
            output_type="scalar",  # Changed from "array" to "scalar"
            pred_sample=(0, 3001),
            labels=["magnitude"],  # Single output for magnitude
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = 1  # Single scalar output for magnitude regression
        self.norm = norm
        self.filter_factor = filter_factor
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        # Input convolution
        self.inc = nn.Conv1d(
            self.in_channels,
            self.filters_root * filter_factor,
            self.kernel_size,
            padding="same",
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root * filter_factor, eps=1e-3)

        # Downsampling branch
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
                # No downsampling at the bottom
                conv_down = None
                bn2 = None
            else:
                conv_down = Conv1dSame(
                    filters,
                    filters,
                    self.kernel_size,
                    stride=self.stride,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        # Upsampling branch
        self.up_branch = nn.ModuleList()
        for i in range(self.depth - 1):
            filters = int(2 ** (3 - i) * self.filters_root) * filter_factor
            conv_up = nn.ConvTranspose1d(
                last_filters, filters, self.kernel_size, self.stride, bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            conv_same = nn.Conv1d(
                2 * filters, filters, self.kernel_size, padding="same", bias=False
            )
            bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.up_branch.append(nn.ModuleList([conv_up, bn1, conv_same, bn2]))

        # Scalar magnitude head: Global pooling + MLP
        # Similar to UMamba V3's scalar head approach
        self.global_pool = nn.AdaptiveAvgPool1d(1)  # Pool temporal dimension
        self.magnitude_head = nn.Sequential(
            nn.Linear(last_filters, last_filters // 2),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(last_filters // 2, 1),
        )

    def forward(self, x):
        """
        Forward pass through PhaseNet U-Net with scalar magnitude prediction.
        
        Args:
            x: Input tensor of shape (batch, channels, samples)
        
        Returns:
            magnitude_scalar: (batch,) scalar magnitude predictions
        """
        x = self.activation(self.in_bn(self.inc(x)))

        # Downsampling path
        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                x = self.activation(bn2(conv_down(x)))

        # Upsampling path
        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]  # Crop to match skip connection size

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        # Global pooling and scalar prediction
        x_pooled = self.global_pool(x).squeeze(-1)  # (batch, channels)
        magnitude_scalar = self.magnitude_head(x_pooled).squeeze(-1)  # (batch,)
        
        return magnitude_scalar

    @staticmethod
    def _merge_skip(skip, x):
        """Merge skip connection with upsampled feature map."""
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]
        return torch.cat([skip, x_resize], dim=1)

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
        """Post-processing: return scalar predictions as-is (no transposition needed)."""
        return batch

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
        model_args["norm"] = self.norm
        model_args["filter_factor"] = self.filter_factor

        return model_args
