# UMamba V3 - Dual-Head Magnitude Estimation

## Overview

UMamba V3 is an encoder-only magnitude estimation model with a **dual-head architecture** that combines:

1. **Primary Head (Scalar)**: Global pooling → MLP → single magnitude value per waveform
2. **Auxiliary Head (Temporal)**: 1×1 conv → magnitude prediction at each timestep

### Key Features

- ✅ **Multi-task learning** with weighted loss combination
- ✅ **Richer training signal** from temporal supervision
- ✅ **Clean evaluation** using only scalar predictions (1 per waveform)
- ✅ **Temporal analysis** available during inference (optional)
- ✅ **Better regularization** through auxiliary task

## Architecture

```
Input (batch, 3, 3001)
    ↓
Encoder (Mamba-based)
    ↓ Stage 1 (8 channels)
    ↓ Stage 2 (16 channels) + Mamba
    ↓ Stage 3 (32 channels)
    ↓ Stage 4 (64 channels) + Mamba
    ↓
Features (batch, 64, ~188)
    ├─────────────────┬─────────────────┐
    │                 │                 │
    v                 v                 v
Primary Head    Auxiliary Head    (training only)
    │                 │
Global Pool      Conv1d(1x1)
    │                 │
   MLP           Upsamplea
    │                 │
Scalar (B,)    Temporal (B, 3001)
    │                 │
    v                 v
Used for       Training signal
evaluation     (not evaluated)
```

## Training

### Loss Function

```python
loss = (1 - α) * loss_scalar + α * loss_temporal
```

Where:
- `α` = temporal_weight (default: 0.3)
- `loss_scalar` = MSE on scalar predictions (primary task)
- `loss_temporal` = MSE on temporal predictions (auxiliary task)

### Default Hyperparameters

```python
learning_rate = 1e-3
batch_size = 64
temporal_weight = 0.3  # 70% scalar, 30% temporal
optimizer = "AdamW"
weight_decay = 1e-5
scheduler_patience = 5
scheduler_factor = 0.5
gradient_clip = 1.0
warmup_epochs = 5
early_stopping_patience = 15
```

### Training Command

```python
from my_project.models.UMamba_mag_v3.model import UMambaMag
from my_project.models.UMamba_mag_v3.train import train_umamba_mag_v3

# Create model
model = UMambaMag(
    in_channels=3,
    sampling_rate=100,
    norm="std",
    n_stages=4,
    features_per_stage=[8, 16, 32, 64],
    kernel_size=7,
    strides=[2, 2, 2, 2],
    n_blocks_per_stage=2,
    pooling_type="avg",
    hidden_dims=[128, 64],
    dropout=0.3,
    temporal_weight=0.3,  # Weight for auxiliary task
)

# Train
history = train_umamba_mag_v3(
    model_name="UMambaMag_v3_STEAD",
    model=model,
    data=stead_dataset,
    learning_rate=1e-3,
    epochs=100,
    batch_size=64,
    temporal_weight=0.3,
    early_stopping_patience=15,
)
```

## Evaluation

### Scalar Evaluation (Official Metrics)

Evaluation uses **ONLY scalar predictions** (1 per waveform):

```python
from my_project.models.UMamba_mag_v3.evaluate import evaluate_umamba_mag_v3

results = evaluate_umamba_mag_v3(
    model=model,
    model_path="path/to/best_model.pt",
    data=test_dataset,
    batch_size=64,
    plot_examples=True,
    save_temporal=False,  # Set True to analyze temporal predictions
)

print(f"MSE: {results['mse']:.4f}")
print(f"RMSE: {results['rmse']:.4f}")
print(f"MAE: {results['mae']:.4f}")
print(f"R²: {results['r2']:.4f}")
```

### Temporal Analysis (Optional)

To analyze temporal predictions:

```python
results = evaluate_umamba_mag_v3(
    model=model,
    model_path="path/to/best_model.pt",
    data=test_dataset,
    batch_size=64,
    save_temporal=True,  # Enable temporal analysis
    plot_examples=True,
    num_examples=10,
)

# Access temporal predictions
temporal_preds = results['temporal_predictions']  # (n_samples, 3001)
temporal_targets = results['temporal_targets']    # (n_samples, 3001)
waveforms = results['waveforms']                  # (n_samples, 3, 3001)
```

## Inference

### Scalar Prediction Only

```python
model.eval()
with torch.no_grad():
    x = torch.from_numpy(waveform).float().to(device)
    x_preproc = model.annotate_batch_pre(x.unsqueeze(0), {})
    
    # Get scalar prediction (default)
    magnitude_scalar = model(x_preproc)  # (1,)
    
print(f"Predicted magnitude: {magnitude_scalar.item():.2f}")
```

### Both Scalar and Temporal Predictions

```python
model.eval()
with torch.no_grad():
    x = torch.from_numpy(waveform).float().to(device)
    x_preproc = model.annotate_batch_pre(x.unsqueeze(0), {})
    
    # Get both predictions
    magnitude_scalar, magnitude_temporal = model(x_preproc, return_temporal=True)
    
print(f"Scalar prediction: {magnitude_scalar.item():.2f}")
print(f"Temporal shape: {magnitude_temporal.shape}")  # (1, 3001)
print(f"Temporal mean: {magnitude_temporal.mean().item():.2f}")
print(f"Temporal std: {magnitude_temporal.std().item():.3f}")
```

## Key Differences from V1 and V2

| Feature | V1 | V2 | V3 |
|---------|----|----|-----|
| Architecture | U-Net (encoder-decoder) | Encoder-only | Encoder-only |
| Training | Single head (averaged) | Single head (scalar) | **Dual-head** |
| Output (training) | Temporal → averaged | Scalar only | **Scalar + Temporal** |
| Output (inference) | Scalar | Scalar | Scalar (+ optional temporal) |
| Training signal | Averaged loss | Scalar loss | **Multi-task loss** |
| Temporal analysis | No | No | **Yes (optional)** |

## Advantages of Dual-Head Design

### 1. Richer Training Signal
- Temporal head provides gradient signal at every timestep
- Forces encoder to learn fine-grained temporal features
- Better than single-point supervision

### 2. Regularization
- Acts as implicit regularization (similar to deep supervision)
- Prevents overfitting by providing additional constraints
- Encoder must learn features useful for both tasks

### 3. Better Representations
- Encoder learns features that work for both:
  - Global magnitude estimation (scalar head)
  - Temporal magnitude tracking (temporal head)
- More robust and generalizable representations

### 4. Temporal Analysis
- Can visualize temporal predictions during inference
- Understand model behavior over time
- Use temporal variance as uncertainty estimate

### 5. Clean Evaluation
- Official metrics use only scalar output
- Fair comparison with other models
- No confusion about per-sample vs per-waveform

## Hyperparameter Tuning

### Temporal Weight (α)

Controls the balance between primary and auxiliary tasks:

```python
# More emphasis on scalar task (recommended for final model)
temporal_weight = 0.2  # 80% scalar, 20% temporal

# Balanced (good starting point)
temporal_weight = 0.3  # 70% scalar, 30% temporal

# More emphasis on temporal task (if temporal predictions are important)
temporal_weight = 0.4  # 60% scalar, 40% temporal
```

**Recommendation**: Start with 0.3, then tune based on validation performance.

### Architectural Parameters

```python
# Default (good for most cases)
features_per_stage = [8, 16, 32, 64]
hidden_dims = [128, 64]
dropout = 0.3

# Larger model (more capacity)
features_per_stage = [16, 32, 64, 128]
hidden_dims = [256, 128, 64]
dropout = 0.3

# Smaller model (faster, less memory)
features_per_stage = [8, 16, 32, 48]
hidden_dims = [96, 48]
dropout = 0.2
```

## Model Integration

To integrate into the main training pipeline, add to `main.py`:

```python
from my_project.models.UMamba_mag_v3.model import UMambaMag as UMambaMagV3

# In create_magnitude_model():
elif model_type == "umamba_mag_v3":
    return UMambaMagV3(
        in_channels=in_channels,
        sampling_rate=sampling_rate,
        norm=norm,
        temporal_weight=kwargs.get("temporal_weight", 0.3),
        # ... other params
    )

# In train_magnitude_unified():
elif isinstance(model, UMambaMagV3):
    results = train_umamba_mag_v3(...)

# In evaluate_magnitude_unified():
elif isinstance(model, UMambaMagV3):
    results = evaluate_umamba_mag_v3(...)
```

## Expected Performance

Based on the dual-head design, expected improvements over V2:

- **Better training stability** due to multi-task learning
- **Lower validation loss** from richer training signal
- **Better generalization** from implicit regularization
- **Similar inference speed** (temporal head not used by default)
- **Same evaluation methodology** (scalar predictions only)

## Citation

```
UMamba V3: Dual-Head Magnitude Estimation
Encoder-only architecture with multi-task learning
Primary head: Scalar magnitude (for evaluation)
Auxiliary head: Temporal magnitude (for training signal)
```

## Notes

1. **Temporal predictions are NOT used for evaluation** - only for training and optional analysis
2. **Official metrics use scalar predictions** (1 per waveform) for fair comparison
3. **Temporal weight** should be tuned based on validation performance
4. **Inference is fast** because temporal head is not computed by default
5. **Compatible with SeisBench** WaveformModel structure
