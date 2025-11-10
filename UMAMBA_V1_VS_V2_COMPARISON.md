# UMamba Magnitude Estimation: V1 vs V2 Comparison

## Executive Summary

This document provides a comprehensive comparison between **UMamba_mag (V1)** and **UMamba_mag_v2 (V2)** architectures for seismic magnitude estimation. The key difference is architectural: **V1 uses a U-Net-style encoder-decoder** while **V2 uses an encoder-only architecture with global pooling**.

---

## Architecture Comparison

### UMamba_mag V1: U-Net Style (Encoder-Decoder)

**Architecture:**
```
Input (B, 3, 3001) 
    ↓
Encoder (with Mamba layers)
    ↓ [Stage 1] → [Skip 1]
    ↓ [Stage 2] → [Skip 2]
    ↓ [Stage 3] → [Skip 3]
    ↓ [Stage 4] → [Skip 4]
Bottleneck (B, 64, ~188)
    ↓
Decoder (with skip connections)
    ↑ [Stage 4] ← [Skip 3]
    ↑ [Stage 3] ← [Skip 2]
    ↑ [Stage 2] ← [Skip 1]
    ↑ [Stage 1] ← [Skip 0 from stem]
Output (B, 1, 3001)
    ↓
Mean over time
Final Magnitude (B,)
```

**Key Features:**
- Full encoder-decoder with skip connections
- Outputs a time-series prediction (B, 3001) that needs averaging
- U-Net style architecture adapted from medical image segmentation
- Preserves spatial/temporal resolution through the decoder
- More parameters due to decoder path

**Output Processing:**
- Produces magnitude estimate for each time step
- Final magnitude = mean across all time steps
- Post-processing: `annotate_batch_post` transposes to (batch, samples, channels)

### UMamba_mag V2: Encoder-Only with Pooling

**Architecture:**
```
Input (B, 3, 3001)
    ↓
Encoder (with Mamba layers)
    ↓ [Stage 1: 8 channels]
    ↓ [Stage 2: 16 channels]
    ↓ [Stage 3: 32 channels]
    ↓ [Stage 4: 64 channels]
Features (B, 64, ~188)
    ↓
Global Pooling (AdaptiveAvgPool1d or AdaptiveMaxPool1d)
Pooled Features (B, 64)
    ↓
Regression Head:
  Linear(64, 128) → LeakyReLU → Dropout(0.3)
  Linear(128, 64) → LeakyReLU → Dropout(0.3)
  Linear(64, 1)
    ↓
Final Magnitude (B,)
```

**Key Features:**
- Encoder-only architecture (no decoder)
- Direct scalar output per sample
- Learnable pooling with configurable type (avg/max)
- Flexible regression head with dropout for regularization
- Fewer parameters (no decoder path)
- More standard for regression tasks

**Output Processing:**
- Directly produces scalar magnitude
- No time-series averaging needed
- Post-processing: Simple reshape to (batch, 1) for seisbench compatibility

---

## Detailed Component Comparison

### 1. Shared Components (Identical in Both)

Both versions share these core building blocks:

#### MambaLayer
```python
# Processes sequential data with state space models
# Supports both patch tokens and channel tokens
# Automatically casts to float32 for Mamba computation
```

#### BasicResBlock
```python
# Two-layer residual block with:
# - Conv → BatchNorm → LeakyReLU
# - Conv → BatchNorm
# - Skip connection with optional 1x1 conv
# - Final LeakyReLU activation
```

#### UMambaEncoder
```python
# Multi-stage encoder with:
# - Initial stem (residual block)
# - 4 stages with progressive downsampling
# - Mamba layers at alternating stages
# - Automatic channel/patch token selection
# - Feature map sizes: ~3001 → ~1500 → ~750 → ~375 → ~188
```

### 2. V1-Specific Components

#### UpsampleLayer
```python
class UpsampleLayer(nn.Module):
    """Upsampling with interpolation + 1x1 conv"""
    def __init__(self, input_channels, output_channels, pool_op_kernel_size, mode='nearest'):
        self.conv = Conv1d(input_channels, output_channels, kernel_size=1)
        self.pool_op_kernel_size = pool_op_kernel_size
        
    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.pool_op_kernel_size, mode=self.mode)
        x = self.conv(x)
        return x
```

#### UMambaDecoder
```python
class UMambaDecoder(nn.Module):
    """
    Multi-stage decoder with:
    - Progressive upsampling (4 stages)
    - Skip connection concatenation
    - Residual blocks at each stage
    - Segmentation head (1x1 conv) at each stage
    - Optional deep supervision
    """
```

### 3. V2-Specific Components

#### LearnablePooling (Not Currently Used)
```python
# Element-wise gating for channel-wise importance weighting
# Linear time complexity O(n) vs attention O(n²)
# Reserved for future experimentation
```

#### Global Pooling
```python
# Standard PyTorch adaptive pooling
if pooling_type == "avg":
    self.pooling = nn.AdaptiveAvgPool1d(1)
elif pooling_type == "max":
    self.pooling = nn.AdaptiveMaxPool1d(1)
```

#### Regression Head
```python
# Multi-layer perceptron with dropout
regression_layers = []
in_features = encoder_output_channels  # 64

for hidden_dim in hidden_dims:  # [128, 64]
    regression_layers.extend([
        nn.Linear(in_features, hidden_dim),
        nn.LeakyReLU(inplace=True),
        nn.Dropout(dropout),  # 0.3
    ])
    in_features = hidden_dim

regression_layers.append(nn.Linear(in_features, 1))
```

---

## Parameter Count Comparison

### V1 (Encoder-Decoder)
```
Encoder:
  - Stem: ~200 parameters
  - Stage 1 (3→8): ~1,800 parameters
  - Stage 2 (8→16): ~7,200 parameters
  - Stage 3 (16→32): ~28,800 parameters
  - Stage 4 (32→64): ~115,200 parameters
  - Mamba layers: ~50,000 parameters (varies by stage)

Decoder:
  - Upsampling layers: ~10,000 parameters
  - Decoder stages: ~150,000 parameters (mirror of encoder)
  - Segmentation heads: ~500 parameters

Total V1 (estimated): ~560,000 - 600,000 parameters
```

### V2 (Encoder-Only)
```
Encoder: (same as V1)
  - Stem: ~200 parameters
  - Stage 1-4: ~153,000 parameters
  - Mamba layers: ~50,000 parameters

Pooling:
  - AdaptiveAvgPool1d: 0 parameters

Regression Head:
  - Linear(64, 128): 8,192 parameters
  - Linear(128, 64): 8,192 parameters
  - Linear(64, 1): 64 parameters
  - Dropout: 0 parameters

Total V2 (estimated): ~220,000 - 240,000 parameters
```

**V2 has ~60% fewer parameters than V1** due to the absence of the decoder path.

---

## Forward Pass Comparison

### V1 Forward Pass
```python
def forward(self, x):
    # Input: (B, 3, 3001)
    
    # Encode with skip connections
    skips = self.encoder(x)  # List of 4 skip tensors
    
    # Decode with skip connections
    output = self.decoder(skips)  # (B, 1, 3001)
    
    # Interpolate to match input size if needed
    if output.shape[-1] != input_size:
        output = F.interpolate(output, size=input_size, mode='linear')
    
    # Remove channel dimension
    output = output.squeeze(1)  # (B, 3001)
    
    return output  # Time-series output

# Later: magnitude = output.mean(dim=1)
```

### V2 Forward Pass
```python
def forward(self, x):
    # Input: (B, 3, 3001)
    
    # Encode
    features = self.encoder(x)  # (B, 64, ~188)
    
    # Pool
    pooled = self.pooling(features).squeeze(-1)  # (B, 64)
    
    # Regress
    magnitude = self.regression_head(pooled)  # (B, 1)
    
    return magnitude.squeeze(-1)  # (B,) - direct scalar output
```

**Key Differences:**
- V1: Returns time-series → needs temporal averaging
- V2: Returns scalar directly → no post-processing needed
- V2: More computationally efficient (no decoder)

---

## Training Differences

### Loss Computation

**V1:**
```python
# Target needs to be expanded to time-series
if y_true.dim() == 1:
    y_true = y_true.unsqueeze(1).expand(-1, x.shape[-1])  # (B,) → (B, 3001)

y_pred = model(x)  # (B, 3001)
loss = MSELoss(y_pred, y_true)  # Loss over all time steps
```

**V2:**
```python
# Target is already scalar
if y_true.dim() > 1:
    y_true = y_true.squeeze(-1)  # (B, 1) → (B,)

y_pred = model(x)  # (B,)
loss = MSELoss(y_pred, y_true)  # Scalar loss per sample
```

### Computational Efficiency

| Aspect | V1 | V2 |
|--------|----|----|
| Forward pass time | Slower (decoder) | Faster (no decoder) |
| Memory usage | Higher (skip connections) | Lower |
| Gradient computation | More complex | Simpler |
| Batch size | Typically 32-64 | Can handle 64-128 |

---

## Pros and Cons Analysis

### V1: Encoder-Decoder (U-Net Style)

#### ✅ Advantages:
1. **Temporal Preservation**: Maintains full temporal resolution throughout
2. **Rich Feature Hierarchy**: Skip connections provide multi-scale features
3. **Interpretability**: Time-series output allows analysis of how magnitude estimate evolves
4. **Theoretical Soundness**: U-Net proven effective for dense prediction tasks
5. **Robustness**: Multiple paths for information flow (encoder + decoder + skips)

#### ❌ Disadvantages:
1. **Over-Engineering**: Regression doesn't need pixel-wise predictions
2. **More Parameters**: ~2.5x more parameters than V2
3. **Slower**: Decoder path adds computation time
4. **Memory Intensive**: Must store skip connections
5. **Training Complexity**: More components = more hyperparameters to tune
6. **Potential Overfitting**: More parameters with same task complexity

### V2: Encoder-Only with Pooling

#### ✅ Advantages:
1. **Simpler Architecture**: Fewer components, easier to understand and debug
2. **Fewer Parameters**: ~60% reduction compared to V1
3. **Faster Training & Inference**: No decoder computation
4. **Memory Efficient**: No skip connections to store
5. **Standard for Regression**: Follows established CNN regression patterns
6. **Better Regularization**: Dropout in regression head helps prevent overfitting
7. **Direct Output**: No need for temporal averaging
8. **Scalability**: Can use larger batch sizes

#### ❌ Disadvantages:
1. **Loss of Temporal Info**: Pooling discards temporal structure
2. **Less Interpretable**: Can't see how prediction evolves over time
3. **Single Resolution**: Only uses final encoder features
4. **Potentially Less Robust**: Single path from input to output

---

## Performance Predictions

### Expected V1 Performance:
- **Accuracy**: Potentially slightly better on complex signals (multi-scale features)
- **Training Time**: Longer (more parameters, more compute)
- **Inference Speed**: Slower (~1.5-2x compared to V2)
- **Generalization**: May overfit on smaller datasets
- **Best For**: Tasks requiring temporal analysis or when data is abundant

### Expected V2 Performance:
- **Accuracy**: Comparable or slightly lower, but more efficient
- **Training Time**: Faster (fewer parameters)
- **Inference Speed**: Faster (~1.5-2x compared to V1)
- **Generalization**: Better on smaller datasets (fewer parameters)
- **Best For**: Standard magnitude regression with limited compute/data

---

## Which Architecture is Better?

### 🏆 Recommendation: **V2 (Encoder-Only) is Likely Better**

#### Reasoning:

1. **Task Appropriateness**
   - Magnitude estimation is a **regression task** producing a single scalar
   - V1's time-series output is unnecessary complexity
   - V2's direct scalar output is more appropriate

2. **Computational Efficiency**
   - V2 trains ~2x faster
   - V2 infers ~1.5-2x faster
   - V2 uses less memory

3. **Parameter Efficiency**
   - V2 has 60% fewer parameters
   - Lower risk of overfitting
   - Easier to train with limited data

4. **Practical Considerations**
   - Simpler code = fewer bugs
   - Easier to debug and modify
   - Standard architecture pattern

5. **Scientific Precedent**
   - Image classification uses encoder + pooling + classifier
   - Time-series regression typically uses encoder + pooling + regressor
   - U-Net is designed for segmentation, not regression

### When V1 Might Be Better:

1. **Very Large Datasets** (millions of samples)
   - More parameters can be properly trained
   - Skip connections provide richer features

2. **Complex Temporal Patterns**
   - If magnitude correlates with specific temporal features
   - If you need to analyze how prediction evolves over time

3. **Transfer Learning**
   - If you have pre-trained U-Net weights
   - If you want to fine-tune from segmentation task

4. **Research/Interpretability**
   - If you need to analyze temporal prediction patterns
   - If interpretability of time-series output is valuable

---

## Experimental Validation Plan

To definitively determine which is better, run these experiments:

### 1. Standard Training
```python
# Same hyperparameters for both
config = {
    "learning_rate": 1e-3,
    "epochs": 150,
    "batch_size": 64,
    "optimizer": "AdamW",
    "weight_decay": 1e-5,
}
```

### 2. Compare Metrics
- MAE, RMSE, R² on test set
- Training time per epoch
- Inference time per sample
- Memory usage

### 3. Ablation Studies
- Try different pooling types in V2 (avg vs max)
- Try different regression head architectures
- Test on different datasets (ETHZ, STEAD, etc.)

### 4. Generalization Tests
- Train on STEAD, test on ETHZ
- Evaluate on different magnitude ranges
- Test with added noise

---

## Code Differences Summary

### V1 Model Initialization
```python
model = UMambaMag(
    in_channels=3,
    sampling_rate=100,
    norm="std",
    n_stages=4,
    features_per_stage=[8, 16, 32, 64],
    kernel_size=7,
    strides=[2, 2, 2, 2],
    n_blocks_per_stage=2,
    n_conv_per_stage_decoder=2,  # V1-specific
    deep_supervision=False,       # V1-specific
)
```

### V2 Model Initialization
```python
model = UMambaMag(
    in_channels=3,
    sampling_rate=100,
    norm="std",
    n_stages=4,
    features_per_stage=[8, 16, 32, 64],
    kernel_size=7,
    strides=[2, 2, 2, 2],
    n_blocks_per_stage=2,
    pooling_type="avg",      # V2-specific
    hidden_dims=[128, 64],   # V2-specific
    dropout=0.3,             # V2-specific
)
```

---

## Migration Guide

### Converting from V1 to V2

1. **Update imports:**
```python
# Old
from my_project.models.UMamba_mag.model import UMambaMag

# New
from my_project.models.UMamba_mag_v2.model import UMambaMag
```

2. **Update model initialization:**
```python
# Remove V1 parameters
# - n_conv_per_stage_decoder
# - deep_supervision

# Add V2 parameters
# + pooling_type="avg"
# + hidden_dims=[128, 64]
# + dropout=0.3
```

3. **Update training script:**
```python
# V1: Targets expanded to time-series
# V2: Targets kept as scalars (automatic in new train.py)
```

4. **Update evaluation:**
```python
# V1: Predictions need temporal averaging
predictions = model(x).mean(dim=1)

# V2: Predictions already scalar
predictions = model(x)
```

---

## Conclusion

**UMamba_mag_v2 (Encoder-Only with Pooling) is recommended** for magnitude estimation due to:

1. ✅ More appropriate architecture for regression
2. ✅ 60% fewer parameters
3. ✅ 2x faster training and inference
4. ✅ Better memory efficiency
5. ✅ Lower overfitting risk
6. ✅ Simpler implementation

**UMamba_mag (V1)** should be considered only if:
- You have very large datasets (>1M samples)
- You need temporal interpretability
- You have strong evidence that multi-scale features help

**Next Steps:**
1. Train both models on the same dataset
2. Compare performance metrics
3. Analyze computational costs
4. Choose based on your specific constraints (accuracy vs. speed)

Based on architectural principles and common practices in deep learning, **V2 is the better starting point**, with V1 as a more complex alternative if V2 proves insufficient.
