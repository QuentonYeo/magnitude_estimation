# Training History Plotting Guide

## Overview

The `plot_training_history()` function in `src/my_project/utils/utils.py` supports two different history formats:

### 1. Simple Format (Default)
**Used by:** PhaseNet, EQTransformer, UMamba V1, UMamba V2, AMAG, etc.

**History Keys:**
- `train_losses`: List of training losses per epoch
- `val_losses`: List of validation losses per epoch
- `best_val_loss`: Best validation loss achieved
- `train_maes`: (optional) Training MAE per epoch
- `val_maes`: (optional) Validation MAE per epoch
- `learning_rates`: (optional) Learning rate schedule

**Plots Generated:**
- Loss curves (train vs val)
- Log-scale progress plot

### 2. Detailed Format
**Used by:** UMamba V3 (multi-head architecture)

**History Keys:**
- `train_loss`, `val_loss`: Combined weighted loss
- `train_loss_scalar`, `val_loss_scalar`: Scalar head loss (global magnitude)
- `train_loss_temporal`, `val_loss_temporal`: Temporal head loss (per-timestep)
- `train_uncertainty`, `val_uncertainty`: Log variance (if uncertainty head enabled)
- `learning_rates`: Learning rate schedule

**Plots Generated (with `detailed_metrics=True`):**
- Combined loss (scalar + temporal weighted)
- Scalar head loss
- Temporal head loss
- RMSE comparison
- Learning rate schedule
- Uncertainty evolution OR loss component breakdown

## Usage Examples

### Command Line

```bash
# Simple plotting (works for all models)
python plot_history_example.py src/trained_weights/PhaseNetMag_STEAD_20251020_104830/training_history.pt

# Detailed plotting for UMamba V3
python plot_history_example.py src/trained_weights/UMambaMag_v3_STEAD_20251112_123456/training_history.pt --detailed

# Show plot interactively
python plot_history_example.py path/to/history.pt --show
```

### Python Script

```python
from my_project.utils.utils import plot_training_history

# Simple view (backward compatible, works for all models)
history = plot_training_history(
    "src/trained_weights/model_name/training_history.pt"
)

# Detailed view for UMamba V3
history = plot_training_history(
    "src/trained_weights/UMambaMag_v3_STEAD_20251112_123456/training_history.pt",
    detailed_metrics=True,
    show_plot=False  # Set to True to display interactively
)
```

## Key Differences: UMamba V3 vs Other Models

| Feature | Other Models | UMamba V3 |
|---------|-------------|-----------|
| **Architecture** | Single magnitude output | Triple-head (scalar + temporal + uncertainty) |
| **Loss Function** | Simple MSE | Uncertainty-weighted MSE (Kendall & Gal 2017) |
| **History Keys** | `train_losses`, `val_losses` | `train_loss`, `train_loss_scalar`, `train_loss_temporal` |
| **Plotting Mode** | `detailed_metrics=False` (default) | `detailed_metrics=True` (recommended) |
| **Output Saved** | `training_history_{timestamp}.pt` | `training_history.pt` |

## Understanding UMamba V3 Metrics

### Scalar Loss
- **What it measures:** Error in predicting the single global magnitude value
- **Primary metric:** This is the main task performance indicator
- **Lower is better:** Direct magnitude prediction accuracy

### Temporal Loss
- **What it measures:** Error in predicting magnitude at each time step
- **Auxiliary task:** Helps learn temporal patterns in seismic signals
- **Benefit:** Improves feature representations for the scalar head

### Uncertainty (log_var)
- **What it measures:** Learned confidence/uncertainty for each sample
- **Interpretation:** `σ = sqrt(exp(log_var))` gives prediction standard deviation
- **Adaptive weighting:** Automatically down-weights difficult/noisy samples
- **Negative values:** Higher confidence (lower variance)
- **Positive values:** Lower confidence (higher variance)

### Combined Loss
- **Formula:** `w_s * (precision * MSE_scalar + log_var) + w_t * (precision * MSE_temporal + log_var)`
- **Default weights:** `w_s=0.7` (scalar), `w_t=0.25` (temporal)
- **Optimization target:** What the model actually minimizes during training

## Backward Compatibility

The plotting function is **fully backward compatible**:

- `detailed_metrics=False` (default) works with all models
- If `detailed_metrics=True` but history doesn't have detailed metrics, it automatically falls back to simple mode
- Existing scripts using `plot_training_history()` without the new parameter continue to work unchanged

## Analysis Features

Both plotting modes provide automatic training analysis:

✓ **Overfitting detection:** Checks if validation loss is increasing  
✓ **Convergence analysis:** Evaluates if training has stabilized  
✓ **Train-val gap:** Measures generalization performance  

Detailed mode adds:

✓ **Head comparison:** Analyzes which head is learning faster  
✓ **Uncertainty analysis:** Interprets model confidence changes  
✓ **Component breakdown:** Shows contribution of each loss term  

## Output Files

Plots are automatically saved as high-resolution PNG files:

- Simple mode: `training_history_{timestamp}_plot.png`
- Detailed mode: `training_history_plot_detailed.png`
