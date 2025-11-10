# Magnitude Model Outputs Analysis

**Date:** November 9, 2025  
**Analysis Focus:** Model architecture differences, output formats, training/evaluation consistency

---

## Executive Summary

This analysis examines **6 magnitude estimation models** implemented in the codebase:
1. **PhaseNetMag** - U-Net style per-sample predictions
2. **AMAG_v2 (MagnitudeNet)** - U-Net with LSTM+Attention bottleneck, per-sample predictions
3. **EQTransformerMag** - Transformer-based encoder-decoder, per-sample predictions
4. **ViTMag** - Vision Transformer, scalar predictions
5. **UMambaMag (V1)** - U-Net style with Mamba layers, per-sample predictions averaged
6. **UMambaMag_v2** - Encoder-only with pooling, direct scalar predictions

### Key Findings

1. **Output Format Inconsistency:** Models produce different output shapes
   - **Per-sample models:** Output shape `(batch, samples)` - magnitude at each timestep
   - **Scalar models:** Output shape `(batch,)` - single magnitude per waveform
   
2. **Evaluation Methodology:** Two distinct approaches
   - **Flattened evaluation:** All timestep predictions treated as independent samples
   - **Scalar evaluation:** One prediction per waveform
   
3. **Target Labeling:** Consistent across all models
   - Zero before first P-arrival
   - Event magnitude after first P-arrival
   - Labels shape: `(batch, samples)` for all models

4. **Critical Issue Identified:** Per-sample models are evaluated inconsistently
   - Training: MSE loss computed on all timesteps
   - Evaluation: Some scripts compute metrics on flattened arrays (inflating sample count)
   - True performance: Should be 1 value per waveform

---

## Model-by-Model Analysis

### 1. PhaseNetMag

**Architecture:** U-Net with encoder-decoder  
**Input:** `(batch, 3, 3001)` - 3 channels, 30 seconds @ 100Hz  
**Output:** `(batch, 1, 3001)` after forward, squeezed to `(batch, 3001)`

```python
def forward(self, x):
    # U-Net encoder
    x = self.activation(self.in_bn(self.inc(x)))
    skips = []
    for conv_same, bn1, conv_down, bn2 in self.down_branch:
        x = self.activation(bn1(conv_same(x)))
        if conv_down is not None:
            skips.append(x)
            x = self.activation(bn2(conv_down(x)))
    
    # U-Net decoder
    for (conv_up, bn1, conv_same, bn2), skip in zip(self.up_branch, skips[::-1]):
        x = self.activation(bn1(conv_up(x)))
        x = x[:, :, 1:-2]  # Crop to match skip
        x = self._merge_skip(skip, x)
        x = self.activation(bn2(conv_same(x)))
    
    # Output layer (no activation)
    x = self.out(x)  # (batch, 1, 3001)
    return x
```

**Evaluation Logic:**
```python
# From evaluate.py
y_pred = model(x_preproc)
y_pred = y_pred.squeeze(1)  # (batch, 3001)

# Metrics calculated on FLATTENED arrays
y_true_flat = all_targets.flatten()  # Shape: (batch * 3001,)
y_pred_flat = all_predictions.flatten()  # Shape: (batch * 3001,)

mse = mean_squared_error(y_true_clean, y_pred_clean)
```

**Issue:** Evaluation treats each timestep as independent sample, inflating the effective sample count by 3001x.

**Training:** Uses MSE loss on per-sample predictions, which is correct for this architecture.

---

### 2. AMAG_v2 (MagnitudeNet)

**Architecture:** U-Net with LSTM+Attention bottleneck  
**Input:** `(batch, 3, 3000)` - 3 channels, 30 seconds @ 100Hz  
**Output:** `(batch, 3000)` after forward

```python
def forward(self, x, logits=False):
    # Encoder
    x = self.activation(self.in_bn(self.inc(x)))
    skips = []
    for conv_same, bn1, conv_down, bn2 in self.down_branch:
        x = self.activation(bn1(conv_same(x)))
        if conv_down is not None:
            skips.append(x)
            x = self.activation(bn2(conv_down(x)))
    
    # LSTM + Attention bottleneck
    x = x.transpose(1, 2)  # (batch, time, channels)
    x, _ = self.lstm(x)
    x, _ = self.attention(x, x, x)
    x = self.bottleneck_proj(x)
    x = self.dropout(x)
    x = x.transpose(1, 2)  # (batch, channels, time)
    
    # Decoder
    for (conv_up, bn1, conv_same, bn2), skip in zip(self.up_branch, skips[::-1]):
        x = self.activation(bn1(conv_up(x)))
        # Crop to match skip
        x = self._merge_skip(skip, x)
        x = self.activation(bn2(conv_same(x)))
    
    # Output
    x = self.out(x)  # (batch, 1, 3000)
    x = torch.squeeze(x, dim=1)  # (batch, 3000)
    return x
```

**Evaluation Logic:**
```python
# From evaluate.py
predictions = model(x_preproc)  # (batch, 3000)
all_predictions.append(predictions.cpu().numpy())

# Concatenate and flatten
all_predictions = np.concatenate(all_predictions, axis=0).flatten()
all_targets = np.concatenate(all_targets, axis=0).flatten()

mse = mean_squared_error(all_targets, all_predictions)
```

**Issue:** Same as PhaseNetMag - treats each timestep as independent sample.

**Special Note:** Uses `MagnitudeLabellerAMAG` which adds +1 to magnitude labels (equation 11 from paper).

---

### 3. EQTransformerMag

**Architecture:** Transformer encoder-decoder  
**Input:** `(batch, 3, 3001)` - 3 channels, 30 seconds @ 100Hz  
**Output:** `(batch, 3001)` after forward

```python
def forward(self, x, logits=False):
    assert x.shape[1:] == (self.in_channels, self.in_samples)
    
    # Shared encoder
    x = self.encoder(x)
    x = self.res_cnn_stack(x)
    x = self.bi_lstm_stack(x)
    x, _ = self.transformer_1(x)
    x, _ = self.transformer_2(x)
    
    # Magnitude prediction decoder
    magnitude = self.decoder_mag(x)
    magnitude = self.conv_mag(magnitude)  # (batch, 1, 3001)
    magnitude = torch.squeeze(magnitude, dim=1)  # (batch, 3001)
    
    return magnitude
```

**Post-processing:**
```python
def annotate_batch_post(self, batch, piggyback, argdict):
    # Add channel dimension: (batch, samples) -> (batch, samples, 1)
    batch = torch.unsqueeze(batch, dim=-1)
    return batch
```

**Evaluation Logic:**
```python
# From evaluate.py
y_pred = model(x_preproc)  # (batch, 3001)
y_pred = y_pred.squeeze(1)  # Remove any extra dims

# Metrics calculated on FLATTENED arrays
y_true_flat = all_targets.flatten()
y_pred_flat = all_predictions.flatten()

mse = mean_squared_error(y_true_clean, y_pred_clean)
```

**Issue:** Same flattening approach - treats each timestep independently.

---

### 4. ViTMag (Vision Transformer)

**Architecture:** Convolutional feature extraction + Transformer + MLP  
**Input:** `(batch, 3, 3001)` - 3 channels, 30 seconds @ 100Hz  
**Output:** `(batch,)` - **SCALAR per waveform**

```python
def forward(self, x):
    # Convolutional feature extraction
    for conv_block in self.conv_blocks:
        x = conv_block(x)
    # x: (batch, 32, 75) approximately
    
    # Patch embedding
    x = self.patch_embed(x)
    # x: (batch, 15, 100)
    
    # Transformer encoders
    for transformer in self.transformer_encoders:
        x = transformer(x)
    # x: (batch, 15, 100)
    
    # Flatten for final MLP
    x = x.reshape(x.shape[0], -1)
    # x: (batch, 1500)
    
    # Final MLP
    x = self.final_mlp(x)
    
    # Output magnitude
    magnitude = self.output(x)  # (batch, 1)
    
    return magnitude.squeeze(-1)  # (batch,) - SCALAR
```

**Evaluation Logic:**
```python
# From evaluate.py
y_pred = model(x_preproc)  # (batch,) - already scalar
y_pred = y_pred.squeeze()

# Handle target shape - should be (batch,) for scalar regression
if y_true.dim() == 2:
    y_true = y_true.mean(dim=1)  # Average the per-sample labels
y_true = y_true.squeeze()

# Concatenate all batches
all_predictions = np.concatenate(all_predictions, axis=0)  # (total_samples,)
all_targets = np.concatenate(all_targets, axis=0)  # (total_samples,)

# Metrics calculated on scalar predictions
mse = mean_squared_error(all_targets, all_predictions)
```

**Correct:** Evaluation matches the model output - one prediction per waveform.

**Target Handling:** Averages the per-sample labels `(batch, 3001)` → `(batch,)` to match scalar prediction.

---

### 5. UMambaMag (V1)

**Architecture:** U-Net style with Mamba layers, encoder-decoder  
**Input:** `(batch, 3, 3001)` - 3 channels, 30 seconds @ 100Hz  
**Output:** `(batch,)` - **SCALAR per waveform** (after global pooling)

```python
def forward(self, x):
    # Encoder forward pass (returns list of skip connections)
    skips = self.encoder(x)
    
    # Use the final (deepest) encoder output
    features = skips[-1]  # (batch, channels, reduced_samples)
    
    # Global pooling
    pooled = self.global_pool(features)  # (batch, channels, 1)
    pooled = pooled.squeeze(-1)  # (batch, channels)
    
    # Regression head
    output = self.regression_head(pooled)  # (batch, 1)
    output = output.squeeze(-1)  # (batch,) - SCALAR
    
    return output
```

**Evaluation Logic:**
```python
# From evaluate.py
y_pred = model(x_preproc)  # (batch,) - scalar predictions

# Store predictions - already scalar per sample
pred_magnitudes = y_pred.cpu().numpy()
true_magnitudes = y_true.cpu().numpy()

# Handle target shape
if true_magnitudes.ndim > 1:
    # If shape is (batch, samples), take mean
    true_magnitudes = true_magnitudes.mean(axis=1)

all_predictions.append(pred_magnitudes)
all_targets.append(true_magnitudes)

# Use predictions directly (already scalar per sample)
pred_final = all_predictions  # (total_samples,)
target_final = all_targets  # (total_samples,)

# Compute metrics
mse = mean_squared_error(target_final, pred_final)
```

**Correct:** Scalar output with proper target averaging. V1 uses global pooling despite having decoder architecture.

**Note:** The decoder branch exists but isn't used for final prediction - only the encoder output is pooled.

---

### 6. UMambaMag_v2

**Architecture:** Encoder-only with global pooling (no decoder)  
**Input:** `(batch, 3, 3001)` - 3 channels, 30 seconds @ 100Hz  
**Output:** `(batch,)` - **SCALAR per waveform**

```python
def forward(self, x):
    # Encode
    features = self.encoder(x)  # (batch, channels, temporal)
    
    # Pool
    pooled = self.pooling(features).squeeze(-1)  # (batch, channels)
    
    # Regress
    magnitude = self.regression_head(pooled)  # (batch, 1)
    
    return magnitude.squeeze(-1)  # (batch,) - SCALAR
```

**Evaluation Logic:**
```python
# From evaluate.py
y_pred = model(x_preproc)  # (batch,) - scalar predictions

# Store predictions - already scalar per sample
pred_magnitudes = y_pred.cpu().numpy()
true_magnitudes = y_true.cpu().numpy()

# Handle target shape
if true_magnitudes.ndim > 1:
    # If shape is (batch, samples), take mean
    true_magnitudes = true_magnitudes.mean(axis=1)

all_predictions.append(pred_magnitudes)
all_targets.append(true_magnitudes)

# V2 directly outputs scalar predictions
pred_final = all_predictions  # (total_samples,)
target_final = all_targets  # (total_samples,)

# Compute metrics
mse = mean_squared_error(target_final, pred_final)
```

**Correct:** Direct scalar output with proper target averaging.

**Improvement over V1:** Removed unnecessary decoder, more efficient architecture.

---

## Comparison Table

| Model | Architecture | Input Shape | Output Shape | Target Processing | Evaluation | Correct? |
|-------|-------------|-------------|--------------|-------------------|------------|----------|
| **PhaseNetMag** | U-Net | (B,3,3001) | (B,3001) | Per-sample labels | Flattened | ❌ No |
| **AMAG_v2** | U-Net+LSTM | (B,3,3000) | (B,3000) | Per-sample labels +1 | Flattened | ❌ No |
| **EQTransformerMag** | Transformer | (B,3,3001) | (B,3001) | Per-sample labels | Flattened | ❌ No |
| **ViTMag** | ViT | (B,3,3001) | (B,) | Averaged labels | Scalar | ✅ Yes |
| **UMambaMag V1** | U-Net+Mamba | (B,3,3001) | (B,) | Averaged labels | Scalar | ✅ Yes |
| **UMambaMag V2** | Encoder+Pool | (B,3,3001) | (B,) | Averaged labels | Scalar | ✅ Yes |

**Legend:**
- B = Batch size
- ✅ = Evaluation methodology matches model output format
- ❌ = Evaluation methodology does not match model output format

---

## Training Methodology Analysis

### Loss Functions

All models use **MSE (Mean Squared Error)** loss during training:

```python
loss = F.mse_loss(predictions, targets)
```

**Per-sample models (PhaseNetMag, AMAG_v2, EQTransformerMag):**
- Predictions: `(batch, samples)`
- Targets: `(batch, samples)`
- Loss computed on all timesteps
- Backpropagation learns to predict magnitude at each time point

**Scalar models (ViTMag, UMambaMag V1/V2):**
- Predictions: `(batch,)`
- Targets: `(batch, samples)` → averaged to `(batch,)`
- Loss computed on single value per waveform
- Backpropagation learns to predict single magnitude value

### Optimizer Settings

**PhaseNetMag:**
```python
learning_rate = 1e-4 (default)
batch_size = 256 (default)
optimizer = Adam
```

**AMAG_v2:**
```python
learning_rate = 1e-3 (default)
batch_size = 256 (default)
optimizer = AdamW
scheduler_factor = 0.5
gradient_clip = 1.0
```

**EQTransformerMag:**
```python
learning_rate = 1e-4 (default)
batch_size = 64 (smaller due to memory)
optimizer = AdamW
scheduler_factor = 0.5
gradient_clip = 1.0
warmup_epochs = 5
```

**ViTMag:**
```python
learning_rate = 1e-4 (default)
batch_size = 64 (smaller due to memory)
optimizer = AdamW
weight_decay = 1e-2 (higher for transformers)
scheduler_factor = 0.5
gradient_clip = 1.0
warmup_epochs = 5
```

**UMambaMag V1/V2:**
```python
learning_rate = 1e-3 (default)
batch_size = 64 (smaller due to Mamba complexity)
optimizer = AdamW
scheduler_factor = 0.5
gradient_clip = 1.0
warmup_epochs = 5
early_stopping_patience = 15 (V2), 10 (V1)
```

---

## Target Labeling Strategy

All models use the **same labeling strategy** via `MagnitudeLabeller`:

```python
class MagnitudeLabeller(SupervisedLabeller):
    def label(self, X, metadata):
        length = X.shape[-1]  # 3001 or 3000
        mag = metadata.get(self.magnitude_column, 0.0)
        
        # Find earliest P arrival
        valid_pick_times = []
        for pick_key in self.phase_dict.keys():
            pick = metadata.get(pick_key, np.nan)
            if not np.isnan(pick) and pick >= 0 and pick < length:
                valid_pick_times.append(pick)
        
        # Initialize label array with zeros
        label = np.zeros(length, dtype=np.float32)
        
        # Set magnitude after first P arrival
        has_valid_picks = len(valid_pick_times) > 0
        has_valid_magnitude = mag > 0
        
        if has_valid_picks and has_valid_magnitude:
            onset = int(min(valid_pick_times))
            label[onset:] = mag  # All samples after P-arrival = magnitude
        
        return label  # Shape: (3001,) or (3000,)
```

**Result:** Labels are `(batch, samples)` with:
- `0.0` before first P-arrival
- `magnitude` value after first P-arrival

**For scalar models:** Labels are averaged during evaluation:
```python
if y_true.dim() == 2:
    y_true = y_true.mean(dim=1)  # (batch, samples) → (batch,)
```

---

## Evaluation Metrics

All models report the same metrics:
- **MSE** (Mean Squared Error)
- **RMSE** (Root Mean Squared Error)  
- **MAE** (Mean Absolute Error)
- **R²** (R-squared score)

### Problem: Inflated Sample Counts

**Per-sample models (PhaseNetMag, AMAG_v2, EQTransformerMag):**

```python
# Example from PhaseNetMag evaluation
print(f"Number of test samples: {len(test_generator)}")  # e.g., 10,000
print(f"Valid predictions: {len(y_true_clean)} / {len(y_true_flat)}")  
# Output: "Valid predictions: 30,010,000 / 30,010,000"
```

**Issue:** The flattened arrays contain `samples * timesteps` predictions, making it appear like there are 3000x more test samples than actually exist.

**True sample count:** Should be number of waveforms, not number of timesteps × waveforms.

### Correct Evaluation

**Scalar models (ViTMag, UMambaMag V1/V2):**

```python
print(f"Number of test samples: {len(pred_final)}")  # e.g., 10,000
# Correct: One prediction per waveform
```

---

## Critical Inconsistencies Found

### 1. Output Format Mismatch

**Per-sample models predict time-series but are supposed to predict scalar magnitudes:**
- PhaseNetMag outputs `(batch, 3001)` - magnitude at every timestep
- Target is `(batch, 3001)` with same value repeated after P-arrival
- This creates redundant predictions that should be averaged

### 2. Evaluation Methodology

**Flattened evaluation treats timesteps as independent samples:**
```python
# Current approach (INCORRECT for scalar magnitude task)
y_pred_flat = predictions.flatten()  # Shape: (batch * 3001,)
y_true_flat = targets.flatten()  # Shape: (batch * 3001,)
mse = mean_squared_error(y_true_flat, y_pred_flat)
```

**This gives each waveform 3001 votes in the metric calculation**, heavily weighting samples with more non-zero timesteps.

**Correct approach (used by scalar models):**
```python
# Average predictions over time dimension
y_pred_scalar = predictions.mean(axis=1)  # Shape: (batch,)
y_true_scalar = targets.mean(axis=1)  # Shape: (batch,)
mse = mean_squared_error(y_true_scalar, y_pred_scalar)
```

### 3. Training vs. Evaluation Inconsistency

**Training:**
- Per-sample models: MSE computed on `(batch, samples)` - correct
- Scalar models: MSE computed on `(batch,)` - correct

**Evaluation:**
- Per-sample models: Metrics on flattened arrays `(batch * samples,)` - **inconsistent**
- Scalar models: Metrics on `(batch,)` - **consistent**

---

## Recommendations

### For Per-Sample Models (PhaseNetMag, AMAG_v2, EQTransformerMag)

**Option 1: Correct Evaluation (Recommended)**
```python
# Average predictions over time to get scalar magnitude
y_pred_scalar = all_predictions.mean(axis=1)  # (batch,) 
y_true_scalar = all_targets.mean(axis=1)  # (batch,)

# Compute metrics on scalar values
mse = mean_squared_error(y_true_scalar, y_pred_scalar)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_true_scalar, y_pred_scalar)
r2 = r2_score(y_true_scalar, y_pred_scalar)
```

**Option 2: Restructure Models (Future Work)**
- Add global pooling layer after encoder/decoder
- Output scalar predictions like UMambaMag V2
- More efficient and cleaner architecture

### For Scalar Models (ViTMag, UMambaMag V1/V2)

**Current implementation is correct** - no changes needed.

### Unified Evaluation Function

**Create a consistent evaluation pipeline:**
```python
def evaluate_magnitude_model_consistent(model, data, batch_size):
    """
    Evaluate magnitude model with consistent methodology.
    Handles both per-sample and scalar output models.
    """
    predictions = []
    targets = []
    
    for batch in data_loader:
        x = batch['X'].to(device)
        y_true = batch['magnitude'].to(device)
        
        # Forward pass
        y_pred = model(x)
        
        # Convert to scalar if needed
        if y_pred.ndim > 1:  # Per-sample model
            y_pred = y_pred.mean(dim=1)  # Average over time
        if y_true.ndim > 1:
            y_true = y_true.mean(dim=1)  # Average over time
        
        predictions.append(y_pred.cpu().numpy())
        targets.append(y_true.cpu().numpy())
    
    # Concatenate
    pred_final = np.concatenate(predictions)  # (total_waveforms,)
    target_final = np.concatenate(targets)  # (total_waveforms,)
    
    # Compute metrics
    mse = mean_squared_error(target_final, pred_final)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_final, pred_final)
    r2 = r2_score(target_final, pred_final)
    
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_samples': len(pred_final),  # True number of waveforms
        'predictions': pred_final,
        'targets': target_final
    }
```

---

## Performance Implications

### Current Reported Metrics May Be Misleading

**Per-sample models report metrics on flattened arrays:**
- Appears to have 3000x more samples
- Metrics are artificially "smoothed" by redundant predictions
- Hard to compare with scalar models fairly

**True comparison requires:**
- Evaluating per-sample models with averaged predictions
- Reporting metrics on same number of waveforms
- Consistent sample counting

### Expected Impact of Correction

**When per-sample models are re-evaluated with averaging:**
- **MSE/RMSE may increase** - fewer predictions to average out errors
- **MAE may change** - depends on temporal consistency of predictions
- **R² may change** - reflects true per-waveform prediction quality
- **Sample count will decrease 3000x** - showing true test set size

---

## Conclusion

### Summary of Findings

1. **Three models (PhaseNetMag, AMAG_v2, EQTransformerMag) output per-sample predictions** but should predict scalar magnitudes
   - These are evaluated incorrectly (flattened)
   - Metrics are computed on 3000x inflated sample counts

2. **Three models (ViTMag, UMambaMag V1/V2) output scalar predictions**
   - These are evaluated correctly
   - Metrics computed on actual waveform count

3. **All models use same target labeling** (zero before P, magnitude after P)
   - Scalar models average targets during evaluation
   - Per-sample models should do the same but don't

4. **Evaluation scripts are not consistent** across model types
   - Need unified evaluation methodology
   - Current comparison between models is not fair

### Recommendations Priority

**High Priority:**
1. Fix evaluation scripts for per-sample models (average predictions over time)
2. Update reported metrics with correct sample counts
3. Re-run evaluations to get fair comparison

**Medium Priority:**
4. Create unified evaluation function for all magnitude models
5. Document evaluation methodology clearly in code

**Low Priority (Future Work):**
6. Consider restructuring per-sample models to scalar output
7. Benchmark different pooling strategies for scalar models

### Model Comparison (Fair)

Once evaluation is corrected, models should be compared on:
- **Same test set**
- **Same number of waveforms** (not timesteps)
- **Scalar predictions** (averaged for per-sample models)
- **Consistent metrics** (MSE, RMSE, MAE, R² on scalar values)

---

## Appendix: Model Output Examples

### Per-Sample Model Output

```python
# Input: (32, 3, 3001) - batch of 32 waveforms
output = phasenet_mag(input)
# Output: (32, 3001) - magnitude at each of 3001 timesteps

# Example for one waveform:
# output[0] = [0, 0, 0, ..., 0, 3.5, 3.5, 3.5, ..., 3.5]
#             ↑___ before P ___↑  ↑_____ after P _____↑
```

### Scalar Model Output

```python
# Input: (32, 3, 3001) - batch of 32 waveforms
output = umamba_mag_v2(input)
# Output: (32,) - single magnitude per waveform

# Example:
# output = [3.5, 2.1, 4.2, ...]
```

### Target Format

```python
# All models receive targets shaped: (batch, samples)
targets = batch['magnitude']  # Shape: (32, 3001)

# Example for one waveform with magnitude 3.5, P-arrival at sample 1000:
# targets[0] = [0, 0, ..., 0, 3.5, 3.5, ..., 3.5]
#              ↑____ 1000 ___↑  ↑_____ 2001 _____↑
```

---

**End of Analysis**
