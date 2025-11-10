"""
UMamba Magnitude Estimation V3 - Dual-Head Architecture

This version combines:
1. Primary head: Scalar magnitude prediction (for evaluation)
2. Auxiliary head: Per-timestep magnitude prediction (for richer training signal)

The dual-head approach provides:
- Richer training signal from temporal supervision
- Better encoder representations
- Regularization through multi-task learning
- Clean scalar evaluation (1 value per waveform)
- Optional temporal analysis during inference
"""

from .model import UMambaMag

__all__ = ['UMambaMag']
