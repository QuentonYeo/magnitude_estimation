# Training Scheduler Analysis - UMamba V3

## Why Are There Steps in the Training Curves?

Looking at your training plots, the distinct "steps" or sudden drops in the loss curves are caused by the **ReduceLROnPlateau** scheduler reducing the learning rate during training.

## Learning Rate Schedule Breakdown

### Configuration (Default Parameters)

```python
# From train.py
warmup_epochs = 5          # Linear warmup from 10% to 100% of base LR
scheduler_patience = 5     # Wait 5 epochs before reducing LR
scheduler_factor = 0.5     # Reduce LR by 50% when triggered
learning_rate = 1e-3       # Base learning rate (typically)
```

### Three-Phase Training Strategy

#### Phase 1: Warmup (Epochs 1-5)
```
Epoch 1: LR = 1e-3 × (1/5) = 2e-4
Epoch 2: LR = 1e-3 × (2/5) = 4e-4
Epoch 3: LR = 1e-3 × (3/5) = 6e-4
Epoch 4: LR = 1e-3 × (4/5) = 8e-4
Epoch 5: LR = 1e-3 × (5/5) = 1e-3 ✓
```

**Purpose:** 
- Prevents instability in early training
- Allows model to gradually adjust to the optimization landscape
- Particularly important for Mamba architecture which can be sensitive to initial LR

**Code:**
```python
if epoch <= warmup_epochs:
    warmup_factor = epoch / warmup_epochs
    for param_group in optimizer.param_groups:
        param_group['lr'] = learning_rate * warmup_factor
```

#### Phase 2: Constant Learning Rate (Epochs 6+)
```
LR = 1e-3 (unchanged)
```

**What happens:** Model trains at full learning rate until validation loss stops improving.

**Code:**
```python
if epoch > warmup_epochs:
    scheduler.step(val_loss)  # Monitor validation loss
```

#### Phase 3: Learning Rate Reductions (Triggered by Plateaus)

**ReduceLROnPlateau Logic:**
1. Monitor validation loss after each epoch (starting from epoch 6)
2. If validation loss doesn't improve for `patience=5` consecutive epochs
3. Reduce learning rate: `new_lr = current_lr × factor` (factor=0.5)
4. Reset patience counter
5. Repeat

**Example LR Schedule from Your Plot:**

Looking at the steps in your training curves, here's what likely happened:

```
Epochs 1-5:   Warmup phase (smooth increase from 2e-4 → 1e-3)
Epochs 6-~35: LR = 1e-3 (large initial improvement)
Epoch ~36:    Plateau detected → LR reduced to 5e-4 ← FIRST STEP
Epochs 36-~50: LR = 5e-4 (continued improvement)
Epoch ~51:    Plateau detected → LR reduced to 2.5e-4 ← SECOND STEP
Epochs 51-~70: LR = 2.5e-4 (slower improvement)
Epoch ~71:    Plateau detected → LR reduced to 1.25e-4 ← THIRD STEP
Epochs 71-~90: LR = 1.25e-4 (fine-tuning)
Epoch ~91:    Plateau detected → LR reduced to 6.25e-5 ← FOURTH STEP
Epochs 91+:   LR = 6.25e-5 (very fine adjustments)
```

## Why This Creates "Steps" in the Loss Curves

### Immediate Effect of LR Reduction

When learning rate is reduced:

1. **Smaller gradient updates** → Model makes more conservative weight changes
2. **Finer exploration** → Can escape shallow local minima and find better solutions
3. **Sudden improvement** → Often see a brief acceleration in loss reduction (the "step")
4. **New plateau** → Eventually reaches another plateau at the new LR

### Visual Pattern in Your Plots

```
Loss
 |
 |  ╲
 |   ╲_______________  ← Plateau at LR=1e-3
 |                   ╲
 |                    ╲_________  ← Plateau at LR=5e-4
 |                              ╲
 |                               ╲____  ← Plateau at LR=2.5e-4
 |                                    ╲___
 |                                        ╲__  ← Continuing...
 |__________________________________________|____
                                          Epochs
```

Each horizontal segment = training at constant LR until patience exhausted
Each steep drop = moment when LR is reduced

## Why Log Scale Shows This More Clearly

In your **log scale plot** (right panel):
- The steps are more visible because log scale compresses the y-axis
- Each LR reduction creates a distinct "bend" in the curve
- The validation loss (red) shows these steps more prominently than training loss (blue)

## Comparing to Training Loss

**Observation:** Training loss (blue) is smoother than validation loss (red)

**Why:**
- Training loss is computed during optimization (model sees gradients)
- Validation loss is computed without gradients (true generalization)
- Scheduler only monitors validation loss → steps align with validation plateaus
- Training loss adjusts smoothly to each new LR

## Is This Good or Bad?

### ✅ **Good Signs in Your Plot:**

1. **Each LR reduction leads to improvement** → Scheduler is working as intended
2. **Validation tracks training** → No severe overfitting
3. **Gap between train/val is small** → Good generalization
4. **Both curves decreasing** → Model is still learning even at low LR

### ⚠️ **Potential Concerns:**

1. **Multiple plateaus needed** → Might benefit from different initial LR or schedule
2. **Very low final LR (6.25e-5)** → Close to diminishing returns
3. **Small improvements after epoch 80** → Could consider early stopping earlier

## Alternative Schedulers to Consider

### 1. Cosine Annealing
```python
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs, eta_min=1e-6
)
```
**Result:** Smooth decay, no steps, more predictable

### 2. Exponential Decay
```python
scheduler = torch.optim.lr_scheduler.ExponentialLR(
    optimizer, gamma=0.95  # 5% reduction per epoch
)
```
**Result:** Gradual exponential decrease

### 3. Custom Step Decay
```python
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[30, 60, 90], gamma=0.5
)
```
**Result:** Predetermined steps at specific epochs

### 4. One Cycle Policy
```python
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, epochs=epochs, 
    steps_per_epoch=len(train_loader)
)
```
**Result:** Warmup + smooth peak + decay, used in fast.ai

## Current Scheduler Pros/Cons

### ✅ Pros of ReduceLROnPlateau:
- **Adaptive:** Responds to actual training dynamics
- **No hyperparameter tuning:** Don't need to guess when to reduce LR
- **Prevents premature stopping:** Gives model chances to improve at lower LRs
- **Widely used:** Standard approach for many tasks

### ❌ Cons of ReduceLROnPlateau:
- **Creates step artifacts:** As seen in your plots
- **Patience is tricky:** Too short = too many reductions, too long = wasted epochs
- **Non-deterministic:** Different runs may have steps at different points
- **Delayed response:** Waits `patience` epochs before acting

## Recommendations for Your Model

Based on your training curves:

### 1. **Current setup is working well** ✓
- You achieved best val loss of 0.0805
- Model is converging smoothly
- No signs of overfitting

### 2. **Consider tuning patience**
```python
scheduler_patience = 7  # Current: 5
# Pros: Fewer LR reductions, smoother curves
# Cons: Might waste epochs at plateaus
```

### 3. **Consider early stopping earlier**
```python
early_stopping_patience = 10  # Current: 15
# Your model seems to plateau hard around epoch 80-90
# Could save training time without sacrificing performance
```

### 4. **Try Cosine Annealing for smoother curves**
```python
# Replace ReduceLROnPlateau with:
warmup_scheduler = # keep your warmup
main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=epochs-warmup_epochs, eta_min=1e-6
)
# Use warmup_scheduler for first 5 epochs, then main_scheduler
```

This would give you smooth curves like: `LR(t) = eta_min + 0.5(eta_max - eta_min)(1 + cos(πt/T))`

## Summary

The **steps in your training curves are caused by the ReduceLROnPlateau scheduler** reducing the learning rate by 50% every time validation loss plateaus for 5 consecutive epochs.

This is **completely normal and expected behavior** for this scheduler type. Each step represents:
1. Model hits a plateau at current LR
2. Patience counter (5 epochs) expires
3. LR is halved
4. Model explores with smaller steps
5. Often finds improvements → new descent
6. Eventually plateaus again → cycle repeats

Your training is healthy - the scheduler is doing its job by adaptively finding the right learning rate for each stage of training!
