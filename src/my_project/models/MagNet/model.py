"""
MagNet - BiLSTM-based Magnitude Estimation Model

A simple and effective model for earthquake magnitude estimation using 
convolutional layers followed by bidirectional LSTM.

Architecture:
    Input: 3-channel seismograms, 30 seconds (3000 samples) at 100 Hz
    
    1. Conv1D (64 kernels, size 3) → Dropout(0.2) → MaxPool(4)
    2. Conv1D (32 kernels, size 3) → Dropout(0.2) → MaxPool(4)  
    3. BiLSTM (100 units)
    4. Fully Connected (1 output, linear activation)
    
    Loss: MSE on scalar magnitude

Reference:
    Based on the architecture described in Mousavi et al. (2020)
    for magnitude estimation from seismic waveforms.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Any

import seisbench.util as sbu
from seisbench.models.base import WaveformModel


class MagNet(WaveformModel):
    """
    BiLSTM-based magnitude estimation model.
    
    Simple architecture with convolutional feature extraction followed by
    bidirectional LSTM for temporal modeling and a fully connected layer
    for magnitude prediction.
    
    Args:
        in_channels: Number of input channels (default: 3 for ENZ components)
        sampling_rate: Sampling rate in Hz (default: 100)
        norm: Normalization method ('std', 'peak', or 'none')
        lstm_hidden: Number of hidden units in BiLSTM layer (default: 100)
        dropout: Dropout rate for regularization (default: 0.2)
    """
    
    _annotate_args = WaveformModel._annotate_args.copy()
    _annotate_args["overlap"] = (_annotate_args["overlap"][0], 1500)
    
    def __init__(
        self,
        in_channels: int = 3,
        sampling_rate: int = 100,
        norm: str = "std",
        lstm_hidden: int = 100,
        dropout: float = 0.2,
        **kwargs,
    ):
        citation = (
            "MagNet: BiLSTM-based magnitude estimation model. "
            "Architecture inspired by Mousavi et al. (2020) for seismic waveform analysis."
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
        self.norm = norm
        self.lstm_hidden = lstm_hidden
        self.dropout_rate = dropout
        
        # First convolutional layer: 64 kernels, size 3
        self.conv1 = nn.Conv1d(
            in_channels=in_channels,
            out_channels=64,
            kernel_size=3,
            padding=1,  # 'same' padding to preserve dimensions
            bias=True
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout)
        self.maxpool1 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # Second convolutional layer: 32 kernels, size 3
        self.conv2 = nn.Conv1d(
            in_channels=64,
            out_channels=32,
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(dropout)
        self.maxpool2 = nn.MaxPool1d(kernel_size=4, stride=4)
        
        # After two maxpool layers with factor 4, the temporal dimension is reduced by 16
        # Input: 3000 samples -> After pool1: 750 -> After pool2: 187 (floor(750/4))
        # The LSTM input size is the number of channels from conv2 (32)
        
        # Bidirectional LSTM layer
        self.lstm = nn.LSTM(
            input_size=32,  # Number of features per timestep (from conv2 output channels)
            hidden_size=lstm_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            dropout=0  # No dropout in single-layer LSTM
        )
        
        # Fully connected layer for scalar magnitude prediction
        # Input: 2 * lstm_hidden (bidirectional concatenation)
        # We'll use the last timestep's output from the LSTM
        self.fc = nn.Linear(2 * lstm_hidden, 1)
        
        # Activation function for conv layers
        self.activation = nn.ReLU()
    
    def forward(self, x):
        """
        Forward pass through the network.
        
        Args:
            x: Input tensor of shape (batch, channels, samples)
               Expected shape: (batch, 3, 3000)
        
        Returns:
            magnitude: Scalar magnitude prediction of shape (batch, 1)
        """
        # First conv block: Conv -> BN -> ReLU -> Dropout -> MaxPool
        x = self.conv1(x)  # (batch, 64, 3000)
        x = self.bn1(x)
        x = self.activation(x)
        x = self.dropout1(x)
        x = self.maxpool1(x)  # (batch, 64, 750)
        
        # Second conv block: Conv -> BN -> ReLU -> Dropout -> MaxPool
        x = self.conv2(x)  # (batch, 32, 750)
        x = self.bn2(x)
        x = self.activation(x)
        x = self.dropout2(x)
        x = self.maxpool2(x)  # (batch, 32, 187)
        
        # Prepare for LSTM: transpose to (batch, seq_len, features)
        x = x.transpose(1, 2)  # (batch, 187, 32)
        
        # BiLSTM layer
        # lstm_out: (batch, seq_len, 2*hidden_size)
        # h_n: (2, batch, hidden_size) - final hidden states for forward and backward
        lstm_out, (h_n, c_n) = self.lstm(x)  # (batch, 187, 200)
        
        # Use the last timestep's output for magnitude prediction
        # Take the output at the last timestep
        last_output = lstm_out[:, -1, :]  # (batch, 200)
        
        # Fully connected layer for scalar magnitude
        magnitude = self.fc(last_output)  # (batch, 1)
        
        return magnitude
    
    def annotate_batch_pre(
        self, batch: torch.Tensor, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Preprocessing: center and optionally normalize the batch.
        
        Options:
        - 'std': Standardize by standard deviation (recommended)
        - 'peak': Normalize by peak amplitude
        - 'none': Only center (remove mean)
        """
        # Center the data (remove mean)
        batch = batch - batch.mean(axis=-1, keepdims=True)
        
        # Apply normalization
        if self.norm == "std":
            batch = batch / (batch.std(axis=-1, keepdims=True) + 1e-10)
        elif self.norm == "peak":
            peak = batch.abs().max(axis=-1, keepdims=True)[0]
            batch = batch / (peak + 1e-10)
        # 'none' means no normalization, just centering
        
        return batch
    
    def annotate_batch_post(
        self, batch: torch.Tensor, piggyback: Any, argdict: dict[str, Any]
    ) -> torch.Tensor:
        """
        Post-processing: reshape output if needed.
        
        The model outputs (batch, 1), which is already in the correct format.
        """
        return batch
    
    def get_model_args(self):
        """
        Get model arguments for saving/loading.
        
        Returns the configuration needed to reconstruct the model.
        """
        model_args = super().get_model_args()
        
        # Remove base class arguments that shouldn't be saved
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
        
        # Add model-specific arguments
        model_args["in_channels"] = self.in_channels
        model_args["sampling_rate"] = self.sampling_rate
        model_args["norm"] = self.norm
        model_args["lstm_hidden"] = self.lstm_hidden
        model_args["dropout"] = self.dropout_rate
        
        return model_args
