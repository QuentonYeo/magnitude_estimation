import torch
import torch.nn as nn
import numpy as np
from typing import Any

import seisbench.util as sbu
from seisbench.models.base import Conv1dSame, WaveformModel


class PhaseNetMag(WaveformModel):
    """
    PhaseNet architecture modified for magnitude regression.

    This model uses the same U-Net architecture as PhaseNet but outputs a single channel
    for magnitude prediction instead of 3 channels for phase classification.

    The model predicts magnitude values at each time sample, where the magnitude
    is expected to be zero before the first P arrival and equal to the event
    magnitude after the first P arrival (as defined by MagnitudeLabellerPhaseNet).
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
            output_type="array",
            pred_sample=(0, 3001),
            labels=["magnitude"],  # Single output for magnitude
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = 1  # Single output for magnitude regression
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

        # Output layer for magnitude regression (no activation function)
        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")

    def forward(self, x):
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

        # Output magnitude predictions (no activation)
        x = self.out(x)
        return x

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
        """Post-processing: transpose predictions to correct shape."""
        # Transpose predictions to correct shape (batch, samples, channels)
        batch = torch.transpose(batch, -1, -2)
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
