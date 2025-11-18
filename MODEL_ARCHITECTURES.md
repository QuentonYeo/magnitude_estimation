# Seismic Magnitude Estimation Model Architectures

This document provides detailed architectural descriptions of all magnitude estimation models in this repository. Each model processes 3-component seismograms (Z, N, E channels) at 100 Hz sampling rate with 30-second windows (3000-3001 samples).

---

## Table of Contents
1. [AMAG v3 - U-Net with LSTM Attention](#amag-v3)
2. [EQTransformer MagV2 - Transformer with Scalar Head](#eqtransformer-magv2)
3. [MagNet - BiLSTM Magnitude Estimator](#magnet)
4. [PhaseNet MagV2 - U-Net with Scalar Head](#phasenet-magv2)
5. [UMamba MagV2 - Encoder-Only State Space Model](#umamba-magv2)
6. [UMamba MagV3 - Triple-Head Multi-Scale Fusion](#umamba-magv3)
7. [ViT Magnitude - Vision Transformer](#vit-magnitude)
8. [Deprecated Models](#deprecated-models)
   - [AMAG v2 (Deprecated)](#amag-v2-deprecated---use-amag-v3)
   - [UMamba v1 (Deprecated)](#umamba-v1-deprecated---use-v2-or-v3)
   - [Version Comparisons](#comparison-deprecated-vs-current-models)

---

## AMAG v3

**Architecture Type:** U-Net Encoder-Decoder with LSTM Attention Bottleneck  
**Parameters:** ~300K (varies with filter_factor)  
**Key Innovation:** Combines spatial feature extraction with temporal attention

### Architecture Overview

```
Input: (batch, 3, 3000)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER BRANCH (5 stages, stride=2)                         │
│                                                              │
│  Conv1d(3→8) + BN → ReLU                                    │
│       ↓                                                      │
│  ┌─ Conv(same) + BN → ReLU ─┐                              │
│  │  Conv(down, s=2) + BN → ReLU (skip connection)          │
│  └─ Repeat for [8→16→32→64→128 filters]                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ LSTM + ATTENTION BOTTLENECK                                  │
│                                                              │
│  Transpose: (batch, 128, T) → (batch, T, 128)              │
│       ↓                                                      │
│  BiLSTM(128→128, 2 layers, bidirectional)                  │
│       ↓                                                      │
│  Multi-Head Attention (4 heads, self-attention)            │
│       ↓                                                      │
│  Linear Projection(256→128) + BN + Dropout                 │
│       ↓                                                      │
│  Transpose back: (batch, T, 128) → (batch, 128, T)        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ SCALAR OUTPUT HEAD                                           │
│                                                              │
│  AdaptiveAvgPool1d(1): (batch, 128, T) → (batch, 128, 1)  │
│       ↓                                                      │
│  Flatten: (batch, 128)                                      │
│       ↓                                                      │
│  Linear(128→128) → ReLU → Dropout                          │
│       ↓                                                      │
│  Linear(128→1)                                              │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch,) scalar magnitude
```

### Encoder Block Structure

```
┌────────────────────────────────────────────────┐
│ Stage i (input: C_in channels)                 │
│                                                 │
│  ┌──────────────────────────────────────────┐ │
│  │ Conv1d(C_in → C_out, k=7, same padding) │ │
│  │           ↓                               │ │
│  │      BatchNorm1d                         │ │
│  │           ↓                               │ │
│  │         ReLU                             │ │
│  └──────────────────────────────────────────┘ │
│           ↓                                    │
│  ┌──────────────────────────────────────────┐ │
│  │ Conv1d(C_out → C_out, k=7, stride=2)    │ │ (skip saved)
│  │           ↓                               │ │
│  │      BatchNorm1d                         │ │
│  │           ↓                               │ │
│  │         ReLU                             │ │
│  └──────────────────────────────────────────┘ │
└────────────────────────────────────────────────┘
```

### ML Techniques
- **U-Net Architecture**: Progressive downsampling with skip connections
- **Bidirectional LSTM**: Captures long-range temporal dependencies in both directions
- **Multi-Head Self-Attention**: Learns which parts of the signal are most relevant
- **Global Average Pooling**: Temporal aggregation before scalar prediction
- **Batch Normalization**: Stable training with normalization after each conv layer
- **Dropout Regularization**: Prevents overfitting (rate: 0.2)

### Input Preprocessing
- Center: Subtract mean
- Normalize: Divide by std (default) or peak

---

## EQTransformer MagV2

**Architecture Type:** Transformer Encoder-Decoder with Scalar Head  
**Parameters:** ~500K  
**Key Innovation:** Combines CNN, BiLSTM, and Transformer attention for magnitude prediction

### Architecture Overview

```
Input: (batch, 3, 3001)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER STACK (7 layers)                                     │
│                                                              │
│  Conv1d layers with increasing filters:                     │
│  [8, 16, 16, 32, 32, 64, 64]                               │
│  Kernel sizes: [11, 9, 7, 7, 5, 5, 3]                      │
│  MaxPool after each layer (stride=2)                        │
│                                                              │
│  Final shape: (batch, 64, ~46)                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ RESIDUAL CNN STACK (7 blocks)                               │
│                                                              │
│  Each block: Conv(k=3,2) → Dropout → Conv(k=3,2) → Add     │
│  Maintains 64 channels                                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ BiLSTM STACK (3 blocks)                                     │
│                                                              │
│  Each block:                                                │
│    BiLSTM(64→16, bidirectional) → Dropout                  │
│    Projects back to 16 channels                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ TRANSFORMER LAYERS (2 layers)                               │
│                                                              │
│  Each layer: SeqSelfAttention(16 channels) → LayerNorm     │
│             → FeedForward → LayerNorm                       │
│                                                              │
│  Output shape: (batch, 16, time)                           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ SCALAR MAGNITUDE HEAD                                        │
│                                                              │
│  AdaptiveAvgPool1d(1): (batch, 16, T) → (batch, 16, 1)    │
│       ↓                                                      │
│  Flatten: (batch, 16)                                       │
│       ↓                                                      │
│  Linear(16→64) → ReLU → Dropout(0.1)                       │
│       ↓                                                      │
│  Linear(64→32) → ReLU → Dropout(0.1)                       │
│       ↓                                                      │
│  Linear(32→1)                                               │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch,) scalar magnitude
```

### Transformer Block Structure

```
┌────────────────────────────────────────────────────┐
│ Transformer Encoder Block                          │
│                                                     │
│  Input: (batch, 16, time)                         │
│      ↓                                             │
│  ┌──────────────────────────────────────────────┐ │
│  │ SeqSelfAttention (Additive Attention)        │ │
│  │   • W_t projection                            │ │
│  │   • Attention weights via softmax             │ │
│  │   • Weighted sum of values                    │ │
│  └──────────────────────────────────────────────┘ │
│      ↓                                             │
│  LayerNorm + Residual                             │
│      ↓                                             │
│  ┌──────────────────────────────────────────────┐ │
│  │ FeedForward Network                           │ │
│  │   Linear(16→128) → ReLU → Dropout            │ │
│  │            ↓                                   │ │
│  │   Linear(128→16)                              │ │
│  └──────────────────────────────────────────────┘ │
│      ↓                                             │
│  LayerNorm + Residual                             │
└────────────────────────────────────────────────────┘
```

### ML Techniques
- **Multi-Scale CNN**: Varying kernel sizes capture different frequency content
- **Residual Connections**: Enables training of deep network
- **Bidirectional LSTM**: Models temporal dependencies in both directions
- **Additive Self-Attention**: Learns which time steps are most informative
- **Layer Normalization**: Stabilizes transformer training
- **Dropout**: 0.1 rate throughout network
- **Cosine Tapering**: 6 samples on each end to reduce edge effects

### Training Details
- **Learning Rate**: 1e-4 (conservative for transformer stability)
- **Optimizer**: AdamW
- **Warmup**: 5 epochs (linear ramp: 20%→100%)
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Early Stopping**: 15 epochs patience
- **Batch Size**: 64

---

## MagNet

**Architecture Type:** Convolutional BiLSTM  
**Parameters:** ~40K  
**Key Innovation:** Simple and efficient architecture with strong performance

### Architecture Overview

```
Input: (batch, 3, 3000)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ CONV BLOCK 1                                                 │
│                                                              │
│  Conv1d(3→64, k=3, padding=1) → BatchNorm → ReLU          │
│       ↓                                                      │
│  Dropout(0.2)                                               │
│       ↓                                                      │
│  MaxPool1d(k=4, stride=4)                                   │
│                                                              │
│  Shape: (batch, 64, 750)                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ CONV BLOCK 2                                                 │
│                                                              │
│  Conv1d(64→32, k=3, padding=1) → BatchNorm → ReLU         │
│       ↓                                                      │
│  Dropout(0.2)                                               │
│       ↓                                                      │
│  MaxPool1d(k=4, stride=4)                                   │
│                                                              │
│  Shape: (batch, 32, 187)                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
    Transpose: (batch, 32, 187) → (batch, 187, 32)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ BiLSTM LAYER                                                 │
│                                                              │
│  BiLSTM(input=32, hidden=100, bidirectional)               │
│                                                              │
│  Output: (batch, 187, 200)  [100*2 from bidirectional]    │
└─────────────────────────────────────────────────────────────┘
    ↓
    Take last timestep: (batch, 200)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ FULLY CONNECTED                                              │
│                                                              │
│  Linear(200→1)                                              │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch, 1) magnitude
```

### Conv Block Structure

```
┌──────────────────────────────────────┐
│ Convolutional Block                  │
│                                       │
│  Conv1d(C_in → C_out, k=3, same)    │
│         ↓                             │
│    BatchNorm1d(C_out)                │
│         ↓                             │
│       ReLU                           │
│         ↓                             │
│    Dropout(0.2)                      │
│         ↓                             │
│  MaxPool1d(k=4, stride=4)           │
│         ↓                             │
│  Output: 4x temporal reduction       │
└──────────────────────────────────────┘
```

### ML Techniques
- **Hierarchical Feature Extraction**: Two conv layers progressively abstract features
- **MaxPooling**: Aggressive downsampling (16x total) reduces sequence length
- **Bidirectional LSTM**: Captures both past and future context
- **Batch Normalization**: After each conv layer for stable training
- **Dropout Regularization**: 0.2 rate after conv layers

### Design Philosophy
- **Simplicity**: Minimal architecture with strong inductive biases
- **Efficiency**: Only ~40K parameters, fast training and inference
- **Inspired by**: Mousavi et al. (2020) seismic waveform analysis

---

## PhaseNet MagV2

**Architecture Type:** U-Net with Global Pooling Scalar Head  
**Parameters:** ~80K (filter_factor=1)  
**Key Innovation:** Adapts PhaseNet's proven U-Net architecture for scalar regression

### Architecture Overview

```
Input: (batch, 3, 3001)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER BRANCH (5 stages, stride=4)                         │
│                                                              │
│  Conv1d(3→8) + BN → ReLU                                    │
│       ↓                                                      │
│  Stage 1: Conv(same, 8→8)  + BN → ReLU ─┐ (skip)          │
│           Conv(down, s=4) + BN → ReLU    │                 │
│       ↓                                   │                 │
│  Stage 2: Conv(same, 8→16) + BN → ReLU ─┤ (skip)          │
│           Conv(down, s=4) + BN → ReLU    │                 │
│       ↓                                   │                 │
│  Stage 3: Conv(same, 16→32) + BN → ReLU ┤ (skip)          │
│           Conv(down, s=4) + BN → ReLU    │                 │
│       ↓                                   │                 │
│  Stage 4: Conv(same, 32→64) + BN → ReLU ┤ (skip)          │
│           Conv(down, s=4) + BN → ReLU    │                 │
│       ↓                                   │                 │
│  Stage 5: Conv(same, 64→128) + BN → ReLU│ (bottleneck)    │
└─────────────────────────────────────────┴──────────────────┘
    ↓                                      ↑
┌─────────────────────────────────────────┴──────────────────┐
│ DECODER BRANCH (4 stages, stride=4)                         │
│                                                              │
│  Stage 1: ConvTranspose(128→64, s=4) + BN → ReLU          │
│           Concat with skip[3]                               │
│           Conv(128→64, same) + BN → ReLU                   │
│       ↓                                                      │
│  Stage 2: ConvTranspose(64→32, s=4) + BN → ReLU           │
│           Concat with skip[2]                               │
│           Conv(64→32, same) + BN → ReLU                    │
│       ↓                                                      │
│  Stage 3: ConvTranspose(32→16, s=4) + BN → ReLU           │
│           Concat with skip[1]                               │
│           Conv(32→16, same) + BN → ReLU                    │
│       ↓                                                      │
│  Stage 4: ConvTranspose(16→8, s=4) + BN → ReLU            │
│           Concat with skip[0]                               │
│           Conv(16→8, same) + BN → ReLU                     │
│                                                              │
│  Final shape: (batch, 8, ~3001)                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ SCALAR HEAD                                                  │
│                                                              │
│  AdaptiveAvgPool1d(1): (batch, 8, T) → (batch, 8, 1)      │
│       ↓                                                      │
│  Flatten: (batch, 8)                                        │
│       ↓                                                      │
│  Linear(8→4) → ReLU → Dropout(0.25)                        │
│       ↓                                                      │
│  Linear(4→1)                                                │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch,) scalar magnitude
```

### U-Net Stage Structure

```
Encoder Stage:                    Decoder Stage:
┌─────────────────────┐          ┌──────────────────────┐
│ Conv(same, k=7)     │          │ ConvTranspose(k=7)   │
│       ↓             │          │       ↓              │
│   BatchNorm         │          │   BatchNorm          │
│       ↓             │          │       ↓              │
│     ReLU            │          │     ReLU             │
│       ↓             │  ┌───────┤       ↓              │
│  [Save Skip] ───────┼──┘       │  Concat(skip)        │
│       ↓             │          │       ↓              │
│ Conv(down, s=4)     │          │ Conv(same, k=7)      │
│       ↓             │          │       ↓              │
│   BatchNorm         │          │   BatchNorm          │
│       ↓             │          │       ↓              │
│     ReLU            │          │     ReLU             │
└─────────────────────┘          └──────────────────────┘
```

### ML Techniques
- **U-Net Architecture**: Encoder-decoder with skip connections preserves spatial information
- **Skip Connections**: Concatenates encoder features with decoder for multi-scale fusion
- **Global Average Pooling**: Aggregates spatial information into single vector
- **Batch Normalization**: After every convolution for stable training
- **Stride-4 Downsampling**: Aggressive downsampling (256x total) for efficiency
- **MLP Head**: Small network (8→4→1) for final magnitude prediction

### Differences from Original PhaseNet
- **Output**: Scalar magnitude instead of per-sample phase probabilities
- **Head**: Global pooling + MLP instead of 1x1 conv per sample
- **Training**: Extracts max from temporal labels for scalar supervision

---

## UMamba MagV2

**Architecture Type:** Encoder-Only with Mamba State Space Models  
**Parameters:** ~220K  
**Key Innovation:** Replaces attention/convolution with efficient state space models

### Architecture Overview

```
Input: (batch, 3, 3000)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER (4 stages with alternating Mamba layers)            │
│                                                              │
│  Stage 0: [8 channels]                                      │
│    ResBlock → ResBlock → (no Mamba)                        │
│    Downsample (stride=2) → (batch, 8, 1500)               │
│       ↓                                                      │
│  Stage 1: [16 channels]                                     │
│    ResBlock → ResBlock → Mamba Layer                       │
│    Downsample (stride=2) → (batch, 16, 750)               │
│       ↓                                                      │
│  Stage 2: [32 channels]                                     │
│    ResBlock → ResBlock → (no Mamba)                        │
│    Downsample (stride=2) → (batch, 32, 375)               │
│       ↓                                                      │
│  Stage 3: [64 channels]                                     │
│    ResBlock → ResBlock → Mamba Layer                       │
│    No downsample → (batch, 64, 375)                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ POOLING + REGRESSION HEAD                                    │
│                                                              │
│  Average Pooling: (batch, 64, 375) → (batch, 64)          │
│       ↓                                                      │
│  Linear(64→128) → ReLU → Dropout(0.3)                      │
│       ↓                                                      │
│  Linear(128→64) → ReLU → Dropout(0.3)                      │
│       ↓                                                      │
│  Linear(64→1)                                               │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch,) scalar magnitude
```

### Mamba Layer Structure

```
┌─────────────────────────────────────────────────────────────┐
│ Mamba State Space Layer                                      │
│                                                              │
│  Input: (batch, channels, temporal)                         │
│      ↓                                                       │
│  Reshape to tokens: (batch, temporal, channels)             │
│      ↓                                                       │
│  LayerNorm                                                  │
│      ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Mamba SSM Core                                      │    │
│  │                                                      │    │
│  │  • Selective scan mechanism                         │    │
│  │  • Input-dependent state transitions                │    │
│  │  • Linear time complexity O(L)                      │    │
│  │                                                      │    │
│  │  Parameters:                                         │    │
│  │    - d_state=16  (SSM state dimension)             │    │
│  │    - d_conv=4    (local conv width)                │    │
│  │    - expand=2    (expansion factor)                │    │
│  └────────────────────────────────────────────────────┘    │
│      ↓                                                       │
│  Reshape back: (batch, channels, temporal)                  │
│      ↓                                                       │
│  Residual Add                                               │
└─────────────────────────────────────────────────────────────┘
```

### Residual Block Structure

```
┌────────────────────────────────────┐
│ BasicResBlock                      │
│                                     │
│  Input ──────────────────┐         │
│    ↓                     │         │
│  Conv1d(k=3, p=1)       │         │
│    ↓                     │         │
│  BatchNorm              │         │
│    ↓                     │         │
│  LeakyReLU              │         │
│    ↓                     │         │
│  Conv1d(k=3, p=1)       │         │
│    ↓                     │         │
│  BatchNorm              │         │
│    ↓                     │         │
│  Add ←───────────────────┘         │
│    ↓                               │
│  LeakyReLU                         │
└────────────────────────────────────┘
```

### ML Techniques
- **State Space Models (Mamba)**: Linear-time sequence modeling vs quadratic attention
- **Selective Scan**: Input-dependent state transitions (key innovation over linear RNNs)
- **Residual Connections**: Enable deep network training
- **Alternating Mamba**: Mamba at stages 1 and 3 for long-range modeling
- **Progressive Downsampling**: Stride-2 reduces sequence length efficiently
- **Encoder-Only Design**: No decoder needed for scalar regression

### Advantages over V1
- **50% fewer parameters** (220K vs 560K)
- **Faster inference** (no decoder upsampling)
- **Simpler architecture** (direct pooling instead of temporal-to-scalar conversion)
- **Better efficiency** (linear complexity for long sequences)

---

## UMamba MagV3

**Architecture Type:** Encoder-Only with Triple-Head Multi-Scale Fusion  
**Parameters:** ~216K  
**Key Innovation:** Multi-scale feature fusion from all encoder stages + uncertainty estimation

### Architecture Overview

```
Input: (batch, 3, 3000)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER (4 stages, ALL features saved)                      │
│                                                              │
│  Stage 0: ResBlock×2 → (batch, 8, 1500)  ────┐             │
│  Stage 1: ResBlock×2 → Mamba → (batch, 16, 750) ──┤         │
│  Stage 2: ResBlock×2 → (batch, 32, 375)  ────┤    │         │
│  Stage 3: ResBlock×2 → Mamba → (batch, 64, 375)  ─┘    │    │
│                                                 │    │    │  │
└─────────────────────────────────────────────────┼────┼────┼──┘
                                                  ↓    ↓    ↓  ↓
┌─────────────────────────────────────────────────────────────┐
│ TRIPLE-HEAD PREDICTION                                       │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ SCALAR HEAD (Primary, weight=0.7)                    │  │
│  │                                                        │  │
│  │  Multi-Scale Fusion:                                  │  │
│  │    Pool stage 0 → (batch, 8)  ─┐                     │  │
│  │    Pool stage 1 → (batch, 16) ─┤                     │  │
│  │    Pool stage 2 → (batch, 32) ─┤ Concatenate         │  │
│  │    Pool stage 3 → (batch, 64) ─┘                     │  │
│  │         ↓                                             │  │
│  │    Fused: (batch, 120)  [8+16+32+64]                 │  │
│  │         ↓                                             │  │
│  │    Linear(120→192) → ReLU → Dropout(0.25)           │  │
│  │         ↓                                             │  │
│  │    Linear(192→96) → ReLU → Dropout(0.25)            │  │
│  │         ↓                                             │  │
│  │    Linear(96→1)                                       │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ TEMPORAL HEAD (Auxiliary, weight=0.25)               │  │
│  │                                                        │  │
│  │  Last stage features: (batch, 64, 375)               │  │
│  │         ↓                                             │  │
│  │    Conv1d(64→1, k=1) → per-timestep predictions     │  │
│  │         ↓                                             │  │
│  │    Output: (batch, 375)                               │  │
│  └──────────────────────────────────────────────────────┘  │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐  │
│  │ UNCERTAINTY HEAD (Optional, weight=0.05)             │  │
│  │                                                        │  │
│  │  Last stage features: (batch, 64, 375)               │  │
│  │         ↓                                             │  │
│  │    Average Pool → (batch, 64)                         │  │
│  │         ↓                                             │  │
│  │    Linear(64→1) → log variance                       │  │
│  │         ↓                                             │  │
│  │    Uncertainty-weighted loss (Kendall & Gal 2017)    │  │
│  └──────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch,) scalar magnitude [+ optional temporal & uncertainty]
```

### Multi-Scale Fusion Head

```
┌────────────────────────────────────────────────────────────┐
│ Multi-Scale Scalar Head (Concatenates ALL stages)          │
│                                                             │
│  Stage 0 features (batch, 8, 1500)   ─→ MaxPool ─→ (8,)   │
│  Stage 1 features (batch, 16, 750)   ─→ MaxPool ─→ (16,)  │
│  Stage 2 features (batch, 32, 375)   ─→ MaxPool ─→ (32,)  │
│  Stage 3 features (batch, 64, 375)   ─→ MaxPool ─→ (64,)  │
│                                           ↓                 │
│                                    Concatenate              │
│                                           ↓                 │
│                                   Fused (120,)              │
│                                           ↓                 │
│  ┌────────────────────────────────────────────────────┐   │
│  │ MLP Head                                            │   │
│  │   Linear(120→192) → ReLU → Dropout                │   │
│  │            ↓                                        │   │
│  │   Linear(192→96) → ReLU → Dropout                 │   │
│  │            ↓                                        │   │
│  │   Linear(96→1)                                     │   │
│  └────────────────────────────────────────────────────┘   │
│                           ↓                                 │
│                   Scalar magnitude                          │
└────────────────────────────────────────────────────────────┘
```

### Hybrid Pooling (Learnable)

```
┌──────────────────────────────────────┐
│ Hybrid Pooling                       │
│                                       │
│  Input: (batch, channels, temporal)  │
│     ↓                 ↓               │
│  MaxPool          AvgPool            │
│     ↓                 ↓               │
│  (batch, C)      (batch, C)          │
│     ↓                 ↓               │
│     └────── α ────────┘               │
│              ↓                        │
│  α·max + (1-α)·avg                   │
│                                       │
│  α = learnable parameter [0,1]       │
└──────────────────────────────────────┘
```

### ML Techniques
- **Multi-Scale Fusion**: Concatenates features from ALL encoder stages (captures both high-freq P-waves and low-freq energy)
- **Triple-Head Architecture**: 
  - Scalar head (primary): Direct magnitude prediction
  - Temporal head (auxiliary): Per-timestep predictions for richer gradients
  - Uncertainty head (optional): Confidence estimation via log-variance
- **Weighted Loss**: Configurable weights (default: scalar=0.7, temporal=0.25, uncertainty=0.05)
- **Hybrid Pooling**: Learnable blend of max and average pooling
- **Kendall & Gal Uncertainty**: Homoscedastic uncertainty for loss weighting
- **Mamba SSM**: Efficient long-range modeling at alternating stages

### Training Configuration
- **Learning Rate**: 1e-3
- **Optimizer**: AdamW
- **Warmup**: 5 epochs (20%→100%)
- **Gradient Clipping**: 1.0
- **Early Stopping**: 15 epochs patience
- **Batch Size**: 64

### Advantages
- **Best performance**: Multi-scale fusion captures comprehensive signal information
- **Uncertainty quantification**: Provides confidence estimates
- **Flexible inference**: Can extract scalar only or all three heads
- **Efficient**: ~216K parameters with linear time complexity

---

## ViT Magnitude

**Architecture Type:** Vision Transformer  
**Parameters:** ~500K  
**Key Innovation:** Applies transformer attention to seismic waveform patches

### Architecture Overview

```
Input: (batch, 3, 3001)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ CONVOLUTIONAL FEATURE EXTRACTION                             │
│                                                              │
│  Conv Block 1: Conv(3→64, k=3) → Dropout → MaxPool(2)      │
│                Shape: (batch, 64, 1500)                      │
│       ↓                                                      │
│  Conv Block 2: Conv(64→32, k=3) → Dropout → MaxPool(2)     │
│                Shape: (batch, 32, 750)                       │
│       ↓                                                      │
│  Conv Block 3: Conv(32→32, k=3) → Dropout → MaxPool(2)     │
│                Shape: (batch, 32, 375)                       │
│       ↓                                                      │
│  Conv Block 4: Conv(32→32, k=3) → Dropout → MaxPool(5)     │
│                Shape: (batch, 32, 75)                        │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ PATCH EMBEDDING                                              │
│                                                              │
│  Input: (batch, 32, 75)                                     │
│       ↓                                                      │
│  Divide into patches:                                       │
│    Patch size = 5                                           │
│    Num patches = 75/5 = 15                                  │
│       ↓                                                      │
│  Reshape: (batch, 15, 32*5) = (batch, 15, 160)            │
│       ↓                                                      │
│  Linear Projection: (batch, 15, 160) → (batch, 15, 100)   │
│       ↓                                                      │
│  Add Positional Embedding: + learned PE[15, 100]           │
│       ↓                                                      │
│  Output: (batch, 15, 100)  [15 tokens, 100-dim]           │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ TRANSFORMER ENCODER STACK (4 layers)                        │
│                                                              │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Transformer Layer 1                                 │    │
│  │   LayerNorm → Multi-Head Attention (4 heads)       │    │
│  │   Residual Add                                      │    │
│  │   LayerNorm → MLP(100→200→100→100)                │    │
│  │   Residual Add                                      │    │
│  └────────────────────────────────────────────────────┘    │
│       ↓                                                      │
│  [Repeat 3 more times]                                      │
│       ↓                                                      │
│  Output: (batch, 15, 100)                                   │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ GLOBAL AVERAGE POOLING                                       │
│                                                              │
│  Average over tokens: (batch, 15, 100) → (batch, 100)     │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ FINAL MLP HEAD                                               │
│                                                              │
│  Linear(100→1000) → GELU → Dropout(0.5)                    │
│       ↓                                                      │
│  Linear(1000→500) → GELU → Dropout(0.5)                    │
│       ↓                                                      │
│  Linear(500→1)                                              │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch,) scalar magnitude
```

### Transformer Encoder Block

```
┌────────────────────────────────────────────────────────────┐
│ Transformer Encoder Layer                                   │
│                                                              │
│  Input: (batch, 15, 100)                                    │
│      ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ Multi-Head Self-Attention (4 heads)                │    │
│  │                                                      │    │
│  │  For each head:                                     │    │
│  │    Q = Linear(100→100)(x)                          │    │
│  │    K = Linear(100→100)(x)                          │    │
│  │    V = Linear(100→100)(x)                          │    │
│  │         ↓                                           │    │
│  │    Attention = softmax(QK^T / √d) · V              │    │
│  │         ↓                                           │    │
│  │  Average across heads                              │    │
│  │         ↓                                           │    │
│  │  Output Projection → Dropout                       │    │
│  └────────────────────────────────────────────────────┘    │
│      ↓                                                       │
│  LayerNorm + Residual                                       │
│      ↓                                                       │
│  ┌────────────────────────────────────────────────────┐    │
│  │ MLP (Feed-Forward)                                  │    │
│  │                                                      │    │
│  │  Linear(100→200) → GELU → Dropout                 │    │
│  │        ↓                                            │    │
│  │  Linear(200→100) → GELU → Dropout                 │    │
│  │        ↓                                            │    │
│  │  Linear(100→100) → Dropout                         │    │
│  └────────────────────────────────────────────────────┘    │
│      ↓                                                       │
│  LayerNorm + Residual                                       │
└────────────────────────────────────────────────────────────┘
```

### Self-Attention Mechanism

```
┌──────────────────────────────────────────────────┐
│ Self-Attention (per head)                        │
│                                                   │
│  Input tokens: X = (batch, 15, 100)             │
│                                                   │
│  Query:  Q = W_q · X  (batch, 15, 100)          │
│  Key:    K = W_k · X  (batch, 15, 100)          │
│  Value:  V = W_v · X  (batch, 15, 100)          │
│                                                   │
│  Attention Scores = Q · K^T / √d                 │
│                     (batch, 15, 15)              │
│          ↓                                        │
│  Attention Weights = softmax(scores)             │
│          ↓                                        │
│  Output = Weights · V                            │
│           (batch, 15, 100)                       │
│                                                   │
│  Interpretation:                                  │
│    Each patch attends to all other patches       │
│    Learns which temporal regions are important   │
└──────────────────────────────────────────────────┘
```

### ML Techniques
- **Patch Embedding**: Divides signal into non-overlapping patches (5 samples each)
- **Positional Encoding**: Learnable position embeddings preserve temporal order
- **Multi-Head Self-Attention**: Each token attends to all others (global receptive field)
- **Layer Normalization**: Stabilizes transformer training (before each sub-layer)
- **Residual Connections**: Enables deep network training
- **GELU Activation**: Smooth non-linearity (better than ReLU for transformers)
- **Global Average Pooling**: Aggregates patch representations
- **Two-Stage Processing**: CNNs for local features → Transformers for global context

### Design Rationale
- **Convolutional preprocessing**: Reduces sequence length and extracts local features
- **Transformer attention**: Captures long-range dependencies between patches
- **No class token**: Uses average pooling instead (simpler for regression)
- **Linear activation in conv**: Preserves amplitude information

### Training Details
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW
- **Dropout**: 0.1 in attention/transformer, 0.5 in final MLP
- **Batch Size**: 64
- **Gradient Clipping**: 1.0

---

## Deprecated Models

The following models are included for historical reference but have been superseded by improved versions.

### AMAG v2 (Deprecated - Use AMAG v3)

**Architecture Type:** U-Net Encoder-Decoder with LSTM Attention  
**Parameters:** ~300K  
**Status:** ⚠️ Superseded by AMAG v3 (scalar head version)  
**Reason for Deprecation:** Outputs temporal predictions instead of scalar, requires post-processing

#### Architecture Overview

```
Input: (batch, 3, 3000)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER BRANCH (5 stages, stride=2)                         │
│  [Same as AMAG v3]                                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ LSTM + ATTENTION BOTTLENECK                                  │
│  [Same as AMAG v3]                                          │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ DECODER BRANCH (4 stages, stride=2)                         │
│                                                              │
│  Stage 1: ConvTranspose(128→64) + BN → ReLU                │
│           Concat with skip[3]                               │
│           Conv(128→64, same) + BN → ReLU                   │
│       ↓                                                      │
│  Stage 2: ConvTranspose(64→32) + BN → ReLU                 │
│           Concat with skip[2]                               │
│           Conv(64→32, same) + BN → ReLU                    │
│       ↓                                                      │
│  Stage 3: ConvTranspose(32→16) + BN → ReLU                 │
│           Concat with skip[1]                               │
│           Conv(32→16, same) + BN → ReLU                    │
│       ↓                                                      │
│  Stage 4: ConvTranspose(16→8) + BN → ReLU                  │
│           Concat with skip[0]                               │
│           Conv(16→8, same) + BN → ReLU                     │
│                                                              │
│  Final shape: (batch, 8, ~3000)                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ OUTPUT HEAD (Temporal)                                       │
│                                                              │
│  Conv1d(8→1, k=1): Per-timestep magnitude predictions      │
│                                                              │
│  Output: (batch, 1, 3000) → (batch, 3000)                  │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch, 3000) temporal magnitude predictions
```

#### Key Differences from v3
- **Output**: Temporal array (3000 values) vs scalar (1 value)
- **Decoder**: Full U-Net decoder with skip connections vs direct pooling
- **Head**: 1x1 conv for per-sample prediction vs global pool + MLP
- **Training**: Uses temporal labels directly vs extracts max for scalar
- **Inference**: Requires taking max/mean of output vs direct scalar

#### Why Superseded
1. **Inefficient**: Decoder adds ~150K parameters for temporal output that gets reduced to scalar
2. **Redundant computation**: Predicts 3000 values when only 1 is needed
3. **Post-processing required**: Must aggregate temporal predictions (max, mean, etc.)
4. **Slower inference**: Full U-Net decoder increases latency

#### When to Use
- **Don't use**: AMAG v3 is strictly better for magnitude estimation
- **Historical reference**: Understanding evolution of architecture

---

### UMamba v1 (Deprecated - Use v2 or v3)

**Architecture Type:** U-Net Encoder-Decoder with Mamba SSM  
**Parameters:** ~560K  
**Status:** ⚠️ Superseded by UMamba v2 (encoder-only) and v3 (multi-scale)  
**Reason for Deprecation:** Full U-Net decoder unnecessary for scalar output, 2.5x more parameters than v2

#### Architecture Overview

```
Input: (batch, 3, 3001)
    ↓
┌─────────────────────────────────────────────────────────────┐
│ ENCODER (4 stages with Mamba layers)                        │
│  [Same as UMamba v2]                                        │
│                                                              │
│  Stage 0: ResBlock×2 → (batch, 8, 1500)                    │
│  Stage 1: ResBlock×2 → Mamba → (batch, 16, 750)           │
│  Stage 2: ResBlock×2 → (batch, 32, 375)                    │
│  Stage 3: ResBlock×2 → Mamba → (batch, 64, 375)           │
└─────────────────────────────────────────────────────────────┘
    ↓                                      ↑
┌─────────────────────────────────────────┴──────────────────┐
│ DECODER (4 stages with skip connections)                    │
│                                                              │
│  Stage 1: Upsample(64→32) + Concat skip[2]                 │
│           ResBlock×2                                        │
│       ↓                                                      │
│  Stage 2: Upsample(32→16) + Concat skip[1]                 │
│           ResBlock×2                                        │
│       ↓                                                      │
│  Stage 3: Upsample(16→8) + Concat skip[0]                  │
│           ResBlock×2                                        │
│       ↓                                                      │
│  Stage 4: Upsample(8→8)                                     │
│           ResBlock×2                                        │
│                                                              │
│  Final shape: (batch, 8, ~3001)                            │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ TEMPORAL OUTPUT HEAD                                         │
│                                                              │
│  Conv1d(8→1, k=1): Per-timestep predictions                │
│       ↓                                                      │
│  Output: (batch, 1, 3001)                                   │
│       ↓                                                      │
│  Max pooling: max(dim=-1) → scalar                         │
└─────────────────────────────────────────────────────────────┘
    ↓
Output: (batch,) scalar magnitude (after max reduction)
```

#### Decoder Stage Structure

```
┌────────────────────────────────────┐
│ Decoder Stage                      │
│                                     │
│  Upsample (interpolate + conv)    │
│         ↓                           │
│  Concatenate with skip             │
│         ↓                           │
│  ResBlock (stride=1)               │
│         ↓                           │
│  ResBlock (stride=1)               │
│         ↓                           │
│  [Repeat for n_conv_per_stage]    │
└────────────────────────────────────┘
```

#### Key Differences from v2/v3
- **Architecture**: Full U-Net decoder vs encoder-only
- **Parameters**: ~560K vs ~220K (v2) vs ~216K (v3)
- **Output Path**: Encoder→Decoder→Temporal→MaxPool→Scalar vs direct pooling
- **Skip Connections**: Uses decoder skips (not needed for regression)
- **Efficiency**: 2.5x more parameters with similar performance

#### Performance Comparison
| Metric | v1 | v2 | v3 |
|--------|----|----|-----|
| Parameters | 560K | 220K | 216K |
| Inference Speed | Slow | Fast | Fast |
| Performance | Good | Good | **Best** |

#### Why Superseded
1. **Overparameterized**: Decoder adds 340K params with no benefit for scalar output
2. **Slower inference**: Full upsampling path increases latency
3. **Unnecessary complexity**: Skip connections designed for dense prediction, not regression
4. **Temporal intermediate**: Produces 3001 values then reduces to 1 scalar
5. **Memory intensive**: Decoder feature maps consume more VRAM

#### When to Use
- **Don't use**: v2 is faster with same performance, v3 is more accurate
- **Historical reference**: Understanding why encoder-only is sufficient for regression
- **Research comparison**: Benchmarking encoder-decoder vs encoder-only

---

## Comparison: Deprecated vs Current Models

### AMAG: v2 vs v3

| Aspect | v2 (Deprecated) | v3 (Current) |
|--------|----------------|--------------|
| **Output** | Temporal (3000 samples) | Scalar (1 value) |
| **Head Architecture** | 1x1 Conv | Global Pool + MLP |
| **Parameters** | ~300K | ~300K |
| **Training** | Temporal labels | Scalar labels (max) |
| **Post-processing** | Required (max/mean) | Not needed |
| **Inference Speed** | Slower (full decoder) | Faster (no decoder) |
| **Use Case** | None (deprecated) | ✅ Production ready |

**Migration Path:**
```python
# Old (v2)
predictions = model(x)  # (batch, 3000)
magnitude = predictions.max(dim=-1)[0]  # Post-process to scalar

# New (v3)
magnitude = model(x)  # (batch,) - direct scalar output
```

### UMamba: v1 vs v2 vs v3

| Aspect | v1 (Deprecated) | v2 (Current) | v3 (SOTA) |
|--------|----------------|--------------|-----------|
| **Architecture** | Encoder-Decoder | Encoder-Only | Encoder-Only |
| **Output Path** | Temporal→Max | Direct Scalar | Triple-Head |
| **Parameters** | 560K | 220K | 216K |
| **Skip Connections** | Decoder skips | None | None (saves stages) |
| **Feature Fusion** | Local (decoder) | Single-scale | Multi-scale |
| **Uncertainty** | No | No | Yes |
| **Inference Speed** | Slow | Fast | Fast |
| **Performance** | Good | Good | **Best** |

**Migration Path:**
```python
# Old (v1)
model = UMambaMag(n_stages=4, deep_supervision=False)  # Encoder-decoder
temporal_output = model(x)  # (batch, 3001)
magnitude = temporal_output.max(dim=-1)[0]  # Reduce to scalar

# New (v2) - Simpler
model = UMambaMag(n_stages=4)  # Encoder-only, no decoder args
magnitude = model(x)  # (batch,) - direct output

# New (v3) - Best performance
model = UMambaMag(n_stages=4, pooling_type="max")
magnitude = model(x)  # Primary head output
# Optional: magnitude, temporal, uncertainty = model(x, return_all=True)
```

### Why Encoder-Only Works Better

For **regression** tasks (single scalar output):
1. **No spatial reconstruction needed**: Unlike segmentation, we don't need pixel-wise outputs
2. **Global information sufficient**: Magnitude depends on overall signal properties
3. **Decoder is wasteful**: Upsampling back to full resolution wastes computation
4. **Pooling is optimal**: Directly aggregates features at bottleneck

For **detection/segmentation** tasks (dense outputs):
- Decoder is essential for spatial precision
- Skip connections preserve fine details
- Upsampling recovers resolution

**Key Insight:** Architecture should match task requirements. Scalar regression needs aggregation, not reconstruction.

---

## Summary Comparison Table

### Current Models (Production Ready)

| Model | Type | Parameters | Key Technique | Time Complexity | Status |
|-------|------|------------|---------------|-----------------|--------|
| **AMAG v3** | U-Net + LSTM | ~300K | LSTM attention bottleneck | O(n²) attention | ✅ Current |
| **EQTransformer V2** | CNN-LSTM-Transformer | ~500K | Multi-stage hybrid | O(n²) attention | ✅ Current |
| **MagNet** | CNN-BiLSTM | ~40K | Simple and efficient | O(n) LSTM | ✅ Current |
| **PhaseNet MagV2** | U-Net | ~80K | Skip connections | O(n) conv | ✅ Current |
| **UMamba MagV2** | Encoder + Mamba | ~220K | State space models | O(n) SSM | ✅ Current |
| **UMamba MagV3** | Encoder + Mamba | ~216K | Multi-scale fusion | O(n) SSM | ✅ **SOTA** |
| **ViT Magnitude** | Vision Transformer | ~500K | Patch-based attention | O(n²) attention | ✅ Current |

### Deprecated Models (Historical Reference Only)

| Model | Type | Parameters | Key Technique | Issue | Superseded By |
|-------|------|------------|---------------|-------|---------------|
| **AMAG v2** | U-Net + LSTM | ~300K | Temporal output | Inefficient decoder | AMAG v3 |
| **UMamba v1** | U-Net + Mamba | ~560K | Full encoder-decoder | 2.5x overparameterized | UMamba v2/v3 |

---

## Common Design Patterns

### Normalization Strategies
All models support standardization preprocessing:
```python
# Center: remove mean
batch = batch - batch.mean(axis=-1, keepdims=True)

# Normalize by std (most common)
batch = batch / (batch.std(axis=-1, keepdims=True) + 1e-10)

# OR normalize by peak (alternative)
peak = batch.abs().max(axis=-1, keepdims=True)[0]
batch = batch / (peak + 1e-10)
```

### Temporal-to-Scalar Conversion
Models use different strategies to produce scalar output:

1. **Global Pooling** (PhaseNet V2, UMamba V2, ViT)
   ```python
   x_pooled = x.mean(dim=-1)  # or max(dim=-1)[0]
   magnitude = mlp(x_pooled)
   ```

2. **Multi-Scale Fusion** (UMamba V3)
   ```python
   pooled_stages = [pool(stage) for stage in all_stages]
   fused = torch.cat(pooled_stages, dim=1)
   magnitude = mlp(fused)
   ```

3. **LSTM Last Output** (MagNet)
   ```python
   lstm_out, _ = lstm(x)
   magnitude = fc(lstm_out[:, -1, :])  # Last timestep
   ```

4. **Attention Pooling** (AMAG v3, EQTransformer V2)
   ```python
   lstm_out, _ = lstm(x)
   attended, _ = attention(lstm_out, lstm_out, lstm_out)
   pooled = adaptive_pool(attended)
   magnitude = mlp(pooled)
   ```

### Regularization Techniques
- **Dropout**: Rates vary (0.1-0.5) depending on model capacity
- **Batch Normalization**: After conv layers in CNN-based models
- **Layer Normalization**: In transformers and Mamba layers
- **Gradient Clipping**: 1.0 for transformer-based models
- **Weight Decay**: 1e-2 for AdamW optimizer

### Training Stability
- **Warmup**: 5 epochs linear ramp for large models
- **Scheduler**: ReduceLROnPlateau (patience=5, factor=0.5)
- **Early Stopping**: 10-15 epochs patience
- **Conservative LR**: 1e-4 for transformers, 1e-3 for others

---

## Model Selection Guide

### ⚠️ Avoid Deprecated Models
- **AMAG v2**: Use AMAG v3 instead (same performance, better architecture)
- **UMamba v1**: Use UMamba v2 (2.5x fewer parameters) or v3 (best performance)

### Choose **MagNet** if:
- Need minimal parameters (~40K) and fast inference
- Limited computational resources
- Want simple, interpretable architecture
- Prioritize efficiency over absolute performance

### Choose **PhaseNet MagV2** if:
- Familiar with PhaseNet architecture
- Need balance of performance and simplicity (~80K params)
- Want proven U-Net design with skip connections
- Need good performance without state-of-the-art complexity

### Choose **UMamba MagV2** if:
- Need efficient long-range modeling (linear time)
- Want modern state space architecture
- Have long sequences where attention is too expensive
- Prefer encoder-only design (~220K params)

### Choose **UMamba MagV3** if:
- **Want best performance** (current SOTA)
- Need uncertainty quantification
- Want multi-scale feature fusion
- Can afford slightly more computation (~216K params)
- Need confidence estimates for predictions

### Choose **ViT Magnitude** if:
- Want global receptive field from start
- Comfortable with transformer architectures
- Have sufficient data for transformer training
- Need interpretable attention maps

### Choose **EQTransformer V2** if:
- Need hybrid CNN-LSTM-Transformer approach
- Want multi-stage feature processing
- Familiar with EQTransformer architecture
- Have 30s windows (original design)

### Choose **AMAG v3** if:
- Want U-Net with skip connections
- Need LSTM attention bottleneck
- Prefer encoder-decoder architecture
- Want balance between U-Net and recurrent models

---

## References

1. **PhaseNet**: Zhu & Beroza (2019) - U-Net for phase picking
2. **EQTransformer**: Mousavi et al. (2020) - Transformer for earthquake detection
3. **Vision Transformer**: Dosovitskiy et al. (2021) - Patch-based transformers
4. **Mamba**: Gu & Dao (2023) - Selective state space models
5. **Uncertainty**: Kendall & Gal (2017) - What uncertainties do we need in Bayesian deep learning
6. **U-Net**: Ronneberger et al. (2015) - Convolutional networks for biomedical image segmentation

---

## Implementation Notes

### Data Format
All models expect:
- **Input shape**: `(batch, 3, 3000)` or `(batch, 3, 3001)`
- **Channels**: [Z, N, E] components
- **Sampling rate**: 100 Hz
- **Window**: 30 seconds

### Label Format
Training uses temporal labels:
```python
# Before P-arrival: 0
# After P-arrival: magnitude value
label[0:onset] = 0
label[onset:] = magnitude
```

For scalar heads, extract maximum:
```python
scalar_target = temporal_label.max(dim=1)[0]
```

### Framework
All models inherit from `seisbench.models.WaveformModel` for:
- Standardized preprocessing
- Consistent annotation interface
- Model saving/loading
- Integration with SeisBench datasets

---

*Last updated: November 2025*
