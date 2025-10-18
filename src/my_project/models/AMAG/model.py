import json
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import seisbench.util as sbu
from seisbench.models.base import WaveformModel


class CustomAttention(nn.Module):
    """
    Custom attention mechanism following the Keras implementation with ReLU activation.

    The Keras code uses: SeqSelfAttention(attention_activation='relu')
    This is simpler than the original equation (5) - it just uses scaled dot-product attention
    with ReLU activation instead of sigmoid + softmax.
    """

    def __init__(self, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # Single linear layer for computing attention scores
        # This mimics the Keras SeqSelfAttention behavior
        self.attention = nn.Linear(hidden_size, hidden_size)

    def forward(self, h):
        """
        Args:
            h: Hidden states from LSTM, shape (batch, seq_len, hidden_size)

        Returns:
            output: Attended output, shape (batch, seq_len, hidden_size)
            attention_weights: Attention weights, shape (batch, seq_len, seq_len)
        """
        batch_size, seq_len, hidden_size = h.shape

        # Compute attention scores using scaled dot-product
        # Query = h, Key = h, Value = h (self-attention)
        q = self.attention(h)  # (batch, seq_len, hidden_size)
        k = h  # (batch, seq_len, hidden_size)
        v = h  # (batch, seq_len, hidden_size)

        # Scaled dot-product: Q @ K^T / sqrt(d)
        scores = torch.matmul(q, k.transpose(-2, -1)) / np.sqrt(hidden_size)

        # Apply ReLU activation (as per Keras code)
        scores = F.relu(scores)

        # Normalize with softmax
        attention_weights = F.softmax(scores, dim=-1)  # (batch, seq_len, seq_len)

        # Apply attention to values
        output = torch.matmul(attention_weights, v)  # (batch, seq_len, hidden_size)

        return output, attention_weights


class AMAG(WaveformModel):
    """
    .. document_args:: seisbench.models YourModelName

    Add your model description and parameter documentation here.

    :param in_channels: Number of input channels (typically 3 for Z, N, E components)
    :param classes: Number of output classes
    :param phases: String of phase labels (e.g., 'NPS' for Noise, P-wave, S-wave)
    :param sampling_rate: Sampling rate of the input waveforms in Hz
    """

    # Configure annotation arguments with defaults
    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["*_threshold"] = ("Detection threshold for the provided phase", 0.3)
    _annotate_args["blinding"] = (
        "Number of prediction samples to discard on each side of each window prediction",
        (0, 0),
    )
    # Add or modify other annotation arguments as needed

    # Weight version warnings (if applicable)
    _weight_warnings = [
        # Add warnings about specific weight versions if needed
        # Example: ("weight_name", "version", "Warning message")
    ]

    def __init__(
        self,
        in_channels=3,
        classes=3,
        phases="NPS",
        sampling_rate=100,
        norm="std",
        encoder_depth=4,
        kernel_size=5,
        leaky_relu_slope=0.01,
        **kwargs,
    ):
        # Citation for your model
        citation = (
            "Author(s), Year. "
            "Title of paper. "
            "Journal name, volume(issue), pages. "
            "DOI or URL"
        )

        # Handle any custom options from kwargs
        for option in ("custom_option1", "custom_option2"):
            if option in kwargs:
                setattr(self, option, kwargs[option])
                del kwargs[option]
            else:
                setattr(self, option, False)

        # Initialize parent class
        super().__init__(
            citation=citation,
            in_samples=600,  # 6 seconds at 100 Hz
            output_type="array",
            pred_sample=(0, 600),
            labels=phases,
            sampling_rate=sampling_rate,
            **kwargs,
        )

        # Store configuration parameters
        self.in_channels = in_channels
        self.classes = classes
        self.norm = norm
        self.encoder_depth = encoder_depth
        self.kernel_size = kernel_size
        self.leaky_relu_slope = leaky_relu_slope

        # Build encoder
        self.encoder = nn.ModuleList()

        # Initial CB1 layer: 3 → 8 channels
        in_ch = in_channels
        out_ch = 8
        self.encoder.append(self._make_CB1(in_ch, out_ch))

        # Subsequent encoder layers: CB1 -> CB2
        for d in range(1, encoder_depth):
            in_ch = out_ch
            out_ch = 8 * (2 ** (d - 1))  # nchannel = 8 * 2^(d-1)

            cb1 = self._make_CB1(in_ch, out_ch)
            cb2 = self._make_CB2(out_ch, out_ch)
            self.encoder.append(nn.ModuleList([cb1, cb2]))

        # Final CB1 layer at bottleneck
        in_ch = out_ch
        out_ch = 8 * (2 ** (encoder_depth - 1))
        self.bottleneck = self._make_CB1(in_ch, out_ch)

        # LSTM layer
        # Input: (batch, channels, sequence) -> needs (batch, sequence, features)
        # Hidden size equals number of channels to maintain dimensionality
        lstm_hidden_size = 8 * (2 ** (encoder_depth - 1))
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            batch_first=True,
            bidirectional=False,
        )

        # Attention mechanism
        # Custom attention with sigmoid and softmax
        self.attention = CustomAttention(lstm_hidden_size)

        # Build decoder
        self.decoder = nn.ModuleList()

        # Decoder mirrors encoder structure in reverse
        # After bottleneck we have 8 * 2^(encoder_depth-1) channels
        in_ch = 8 * (2 ** (encoder_depth - 1))

        for d in range(encoder_depth - 1, 0, -1):
            out_ch = 8 * (2 ** (d - 1))

            # Each decoder block: CB3 (upsample + conv) -> CCut (concat + conv)
            cb3 = self._make_CB3(in_ch, out_ch)
            # After concatenation with skip, we have out_ch (from CB3) + out_ch (from skip) = 2*out_ch
            ccut = self._make_CCut(out_ch * 2, out_ch)
            self.decoder.append(nn.ModuleList([cb3, ccut]))

            # Next layer's input is current layer's output
            in_ch = out_ch

        # Final output layer - 1x1 convolution to produce single magnitude output
        final_channels = 8  # After all decoder layers
        self.output_conv = nn.Conv1d(final_channels, 1, kernel_size=1)

    def _make_CB1(self, in_channels, out_channels):
        """
        CB1 block: Conv2d(stride=1) -> BN -> LeakyReLU
        """
        return nn.Sequential(
            nn.Conv1d(
                in_channels, out_channels, self.kernel_size, stride=1, padding="same"
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3),  # Paper uses eps=1e-3
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
        )

    def _make_CB2(self, in_channels, out_channels):
        """
        CB2 block: Conv2d(stride=2) -> BN -> LeakyReLU
        Downsampling through stride
        """
        return nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                self.kernel_size,
                stride=2,
                padding=self.kernel_size // 2,
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3),  # Paper uses eps=1e-3
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
        )

    def _make_CB3(self, in_channels, out_channels):
        """
        CB3 block: Upsampling2D -> Conv1d(stride=1) -> BN -> LeakyReLU
        Keras: UpSampling2D(size=(2,1)) then Conv2D
        PyTorch: Upsample then Conv1d
        """
        return nn.Sequential(
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.Conv1d(
                in_channels, out_channels, self.kernel_size, stride=1, padding="same"
            ),
            nn.BatchNorm1d(out_channels, eps=1e-3),
            nn.LeakyReLU(negative_slope=self.leaky_relu_slope),
        )

    def forward(self, x, logits=False):
        """
        Forward pass of the model.

        :param x: Input tensor of shape (batch_size, in_channels, samples)
        :param logits: If True, return raw logits; if False, return probabilities
        :return: Output tensor of shape (batch_size, classes, samples)
        """
        # Encoder forward pass
        encoder_outputs = []

        # Initial CB1 layer
        x = self.encoder[0](x)
        encoder_outputs.append(x)  # Store: 8 channels, 600 length

        # Subsequent encoder layers with CB1 -> CB2
        for i in range(1, self.encoder_depth):
            cb1, cb2 = self.encoder[i]
            x = cb1(x)  # Transform channels
            encoder_outputs.append(
                x
            )  # Store AFTER CB1, BEFORE CB2 (for skip connections)
            x = cb2(x)  # Downsample

        # Bottleneck
        x = self.bottleneck(x)

        # LSTM processing
        # Keras: Reshape to (batch, sequence*1, channels) for LSTM
        # PyTorch: (batch, channels, sequence) -> (batch, sequence, channels)
        batch_size = x.shape[0]
        x = x.transpose(1, 2)  # (batch, sequence, channels)

        # Pass through LSTM
        # Output shape: (batch, sequence, hidden_size)
        x, (h_n, c_n) = self.lstm(x)

        # Attention mechanism
        x, attention_weights = self.attention(x)
        # x is still (batch, 600/(2^(d-1)), 8*2^(d-1))

        # Reshape back to (batch, channels, sequence) for decoder
        # Keras: Reshape back to (batch, 1, sequence, channels)
        # PyTorch: (batch, sequence, channels) -> (batch, channels, sequence)
        x = x.transpose(1, 2)  # (batch, 8*2^(d-1), 600/(2^(d-1)))

        # Decoder WITHOUT skip connections (like Keras crop_and_cut which crops upsampled to match skip)
        # The Keras code crops net2 (upsampled) to match net1 (skip), then just uses net2
        # This means it's NOT actually concatenating - just cropping!
        for i, cb3 in enumerate(self.decoder):
            # Upsample / decode stage
            if isinstance(cb3, nn.ModuleList):
                for layer in cb3:
                    x = layer(x)
            else:
                x = cb3(x)

            # Get corresponding encoder output for potential cropping
            skip_idx = len(encoder_outputs) - 1 - i
            skip = encoder_outputs[skip_idx]

            # Crop upsampled feature to match skip dimensions (like Keras crop_and_cut)
            # But in Keras, after cropping they just use the cropped net2, not concatenation!
            if skip.shape[-1] != x.shape[-1]:
                # Crop x to match skip's size
                diff = x.shape[-1] - skip.shape[-1]
                if diff > 0:
                    x = x[:, :, diff // 2 : diff // 2 + skip.shape[-1]]

        # Final output convolution
        x = self.output_conv(x)  # (batch, 1, 600)

        # Output shape: (batch, 1, 600)

        if logits:
            return x
        else:
            return torch.softmax(x, dim=1)

    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Preprocessing applied to each batch before model inference.

        :param batch: Input batch tensor
        :param argdict: Dictionary of annotation arguments
        :return: Preprocessed batch tensor
        """
        # Standard preprocessing: remove mean
        batch = batch - batch.mean(axis=-1, keepdims=True)

        # Apply normalization based on self.norm
        if self.norm == "std":
            std = batch.std(axis=-1, keepdims=True)
            batch = batch / (std + 1e-10)
        elif self.norm == "peak":
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)
        # Add other normalization methods as needed

        return batch

    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Postprocessing applied to model outputs after inference.

        :param batch: Output batch tensor from model
        :param piggyback: Additional data carried through annotation
        :param argdict: Dictionary of annotation arguments
        :return: Postprocessed batch tensor
        """
        # Transpose predictions to correct shape if needed
        # (batch_size, classes, samples) -> (batch_size, samples, classes)
        batch = torch.transpose(batch, -1, -2)

        # Apply blinding (set edge samples to NaN)
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
        Converts the annotations to discrete predictions (e.g., picks).

        :param annotations: Annotations from the model
        :param argdict: Dictionary of annotation arguments
        :return: ClassifyOutput with picks or other discrete predictions
        """
        picks = sbu.PickList()

        # Extract picks for each phase
        for phase in self.labels:
            if phase == "N":  # Skip noise class
                continue

            # Get threshold for this phase
            threshold = argdict.get(
                f"{phase}_threshold", self._annotate_args.get("*_threshold")[1]
            )

            # Convert continuous predictions to discrete picks
            picks += self.picks_from_annotations(
                annotations.select(channel=f"{self.__class__.__name__}_{phase}"),
                threshold,
                phase,
            )

        # Sort picks by time
        picks = sbu.PickList(sorted(picks))

        return sbu.ClassifyOutput(self.name, picks=picks)

    def get_model_args(self):
        """
        Returns dictionary of model arguments for serialization.

        :return: Dictionary of model configuration parameters
        """
        model_args = super().get_model_args()

        # Remove keys that are set automatically
        for key in [
            "citation",
            "in_samples",
            "output_type",
            "default_args",
            "pred_sample",
            "labels",
        ]:
            del model_args[key]

        # Add model-specific configuration
        model_args["in_channels"] = self.in_channels
        model_args["classes"] = self.classes
        model_args["phases"] = self.labels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["encoder_depth"] = self.encoder_depth
        model_args["kernel_size"] = self.kernel_size
        model_args["leaky_relu_slope"] = self.leaky_relu_slope

        # Add any custom options
        # model_args["custom_option1"] = self.custom_option1

        return model_args
