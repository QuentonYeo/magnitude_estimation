import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from packaging import version

import seisbench.util as sbu

from seisbench.models.base import Conv1dSame, WaveformModel, _cache_migration_v0_v3


class PhaseNetLSTM(WaveformModel):
    """
    PhaseNet with LSTM at the bottleneck of the U-Net architecture.

    .. document_args:: seisbench.models PhaseNetLSTM

    :param filter_factor: Increase the number of filters used in each layer by this factor compared to the original
                          PhaseNet. Based on PhaseNetWC proposed by Naoi et al. (2024)
    :param lstm_hidden_size: Hidden size of the LSTM layer at the bottleneck (if None, uses the bottleneck filters)
    :param lstm_num_layers: Number of LSTM layers at the bottleneck
    :param lstm_bidirectional: Whether to use bidirectional LSTM
    """

    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (0, 0),
    )
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)

    _weight_warnings = [
        (
            "ethz|geofon|instance|iquique|lendb|neic|scedc|stead",
            "1",
            "The normalization for this weight version is incorrect and will lead to degraded performance. "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/seisbench/seisbench/pull/188 .",
        ),
        (
            "diting",
            "1",
            "This version of the Diting picker uses an incorrect sampling rate (100 Hz instead of 50 Hz). "
            "Run from_pretrained with update=True once to solve this issue. "
            "For details, see https://github.com/JUNZHU-SEIS/USTC-Pickers/issues/1 .",
        ),
    ]

    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases="NPS",
        sampling_rate=100,
        norm="std",
        filter_factor: int = 1,
        lstm_hidden_size: int = None,
        lstm_num_layers: int = 1,
        lstm_bidirectional: bool = True,
        **kwargs,
    ):
        citation = (
            "Zhu, W., & Beroza, G. C. (2019). "
            "PhaseNet: a deep-neural-network-based seismic arrival-time picking method. "
            "Geophysical Journal International, 216(1), 261-273. "
            "https://doi.org/10.1093/gji/ggy423"
        )

        # PickBlue options
        for option in ("norm_amp_per_comp", "norm_detrend"):
            if option in kwargs:
                setattr(self, option, kwargs[option])
                del kwargs[option]
            else:
                setattr(self, option, False)

        super().__init__(
            citation=citation,
            in_samples=3001,
            output_type="array",
            pred_sample=(0, 3001),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.filter_factor = filter_factor
        self.depth = 5
        self.kernel_size = 7
        self.stride = 4
        self.filters_root = 8
        self.activation = torch.relu

        # LSTM parameters
        self.lstm_num_layers = lstm_num_layers
        self.lstm_bidirectional = lstm_bidirectional
        self.bottleneck_filters = (
            int(2 ** (self.depth - 1) * self.filters_root) * filter_factor
        )
        self.lstm_hidden_size = (
            lstm_hidden_size
            if lstm_hidden_size is not None
            else self.bottleneck_filters
        )

        self.inc = nn.Conv1d(
            self.in_channels,
            self.filters_root * filter_factor,
            self.kernel_size,
            padding="same",
        )
        self.in_bn = nn.BatchNorm1d(self.filters_root * filter_factor, eps=1e-3)

        self.down_branch = nn.ModuleList()
        self.up_branch = nn.ModuleList()

        last_filters = self.filters_root * filter_factor
        for i in range(self.depth):
            filters = int(2**i * self.filters_root) * filter_factor
            conv_same = nn.Conv1d(
                last_filters, filters, self.kernel_size, padding="same", bias=False
            )
            last_filters = filters
            bn1 = nn.BatchNorm1d(filters, eps=1e-3)
            if i == self.depth - 1:
                conv_down = None
                bn2 = None
            else:
                if i in [1, 2, 3]:
                    padding = 0  # Pad manually
                else:
                    padding = self.kernel_size // 2
                conv_down = nn.Conv1d(
                    filters,
                    filters,
                    self.kernel_size,
                    self.stride,
                    padding=padding,
                    bias=False,
                )
                bn2 = nn.BatchNorm1d(filters, eps=1e-3)

            self.down_branch.append(nn.ModuleList([conv_same, bn1, conv_down, bn2]))

        # LSTM at the bottleneck
        self.lstm = nn.LSTM(
            input_size=self.bottleneck_filters,
            hidden_size=self.lstm_hidden_size,
            num_layers=lstm_num_layers,
            batch_first=True,
            bidirectional=lstm_bidirectional,
        )

        # Projection layer to match expected dimensions after LSTM
        lstm_output_size = self.lstm_hidden_size * (2 if lstm_bidirectional else 1)
        self.lstm_proj = nn.Conv1d(lstm_output_size, self.bottleneck_filters, 1)
        self.lstm_bn = nn.BatchNorm1d(self.bottleneck_filters, eps=1e-3)

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

        self.out = nn.Conv1d(last_filters, self.classes, 1, padding="same")
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x, logits=False):
        x = self.activation(self.in_bn(self.inc(x)))

        skips = []
        for i, (conv_same, bn1, conv_down, bn2) in enumerate(self.down_branch):
            x = self.activation(bn1(conv_same(x)))

            if conv_down is not None:
                skips.append(x)
                if i == 1:
                    x = F.pad(x, (2, 3), "constant", 0)
                elif i == 2:
                    x = F.pad(x, (1, 3), "constant", 0)
                elif i == 3:
                    x = F.pad(x, (2, 3), "constant", 0)

                x = self.activation(bn2(conv_down(x)))

        # Apply LSTM at the bottleneck
        # x shape: (batch, channels, time) -> need (batch, time, channels) for LSTM
        batch_size, channels, time_steps = x.shape
        x = x.permute(0, 2, 1)  # (batch, time, channels)
        x, _ = self.lstm(x)  # (batch, time, lstm_output_size)
        x = x.permute(0, 2, 1)  # (batch, lstm_output_size, time)

        # Project back to expected dimensions
        x = self.activation(self.lstm_bn(self.lstm_proj(x)))

        for i, ((conv_up, bn1, conv_same, bn2), skip) in enumerate(
            zip(self.up_branch, skips[::-1])
        ):
            x = self.activation(bn1(conv_up(x)))
            x = x[:, :, 1:-2]

            x = self._merge_skip(skip, x)
            x = self.activation(bn2(conv_same(x)))

        x = self.out(x)
        if logits:
            return x
        else:
            return self.softmax(x)

    @staticmethod
    def _merge_skip(skip, x):
        offset = (x.shape[-1] - skip.shape[-1]) // 2
        x_resize = x[:, :, offset : offset + skip.shape[-1]]

        return torch.cat([skip, x_resize], dim=1)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        batch = batch - batch.mean(axis=-1, keepdims=True)
        if self.norm_detrend:
            batch = sbu.torch_detrend(batch)
        if self.norm_amp_per_comp:
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)
        else:
            if self.norm == "std":
                std = batch.std(axis=-1, keepdims=True)
                batch = batch / (std + 1e-10)
            elif self.norm == "peak":
                peak = batch.abs().max(axis=-1, keepdims=True)[0]
                batch = batch / (peak + 1e-10)

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        # Transpose predictions to correct shape
        batch = torch.transpose(batch, -1, -2)
        prenan, postnan = argdict.get(
            "blinding", self._annotate_args.get("blinding")[1]
        )
        if prenan > 0:
            batch[:, :prenan] = np.nan
        if postnan > 0:
            batch[:, -postnan:] = np.nan
        return batch

    def classify_aggregate(self, annotations, argdict) -> sbu.ClassifyOutput:
        """
        Converts the annotations to discrete thresholds using
        :py:func:`~seisbench.models.base.WaveformModel.picks_from_annotations`.
        Trigger onset thresholds for picks are derived from the argdict at keys "[phase]_threshold".

        :param annotations: See description in superclass
        :param argdict: See description in superclass
        :return: List of picks
        """
        picks = sbu.PickList()
        for phase in self.labels:
            if phase == "N":
                # Don't pick noise
                continue

            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                argdict.get(
                    f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
                ),
                phase,
            )

        picks = sbu.PickList(sorted(picks))

        return sbu.ClassifyOutput(self.name, picks=picks)

    def get_model_args(self):
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
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["norm_amp_per_comp"] = self.norm_amp_per_comp
        model_args["norm_detrend"] = self.norm_detrend
        model_args["lstm_hidden_size"] = self.lstm_hidden_size
        model_args["lstm_num_layers"] = self.lstm_num_layers
        model_args["lstm_bidirectional"] = self.lstm_bidirectional

        return model_args
