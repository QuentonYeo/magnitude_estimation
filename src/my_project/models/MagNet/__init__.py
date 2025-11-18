"""
MagNet - BiLSTM-based Magnitude Estimation Model

Simple and effective architecture for earthquake magnitude estimation
using convolutional feature extraction and bidirectional LSTM.
"""

from .model import MagNet
from .train import train_magnet

__all__ = ["MagNet", "train_magnet"]
