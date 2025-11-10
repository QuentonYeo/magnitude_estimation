# UMamba Ablation Study

This document provides a comprehensive ablation study design for the UMamba magnitude estimation model, examining how different architectural parameters affect model performance.

## Table 1: Single-Parameter Ablation Study

| **Parameter** | **Default** | **Ablation Values** | **Expected Impact** | **Priority** | **CLI Command Suffix** |
|--------------|-------------|---------------------|---------------------|--------------|----------------------|
| **`features_per_stage`** | `8,16,32,64` | `[6,12,24,48]`<br>`[8,16,32,64]` (baseline)<br>`[12,24,48,96]`<br>`[16,32,64,128]`<br>`[16,32,64,128,256]`* | Width controls capacity<br>• Narrower: ↓ params, ↑ speed, ↓ accuracy<br>• Wider: ↑ params, ↓ speed, ↑ accuracy | **HIGH** | `--features_per_stage 12,24,48,96` |
| **`n_stages`** | `4` | `3`<br>`4` (baseline)<br>`5`* | Depth controls receptive field<br>• Fewer: ↓ params, ↑ speed, ↓ context<br>• More: ↑ params, ↓ speed, ↑ context | **HIGH** | `--n_stages 3 --features_per_stage 8,16,32 --strides 4,2,2` |
| **`strides`** | `2,2,2,2` | `[1,2,2,2]`<br>`[2,2,2,2]` (baseline)<br>`[4,2,2,2]`<br>`[2,2,2,1]`<br>`[4,4,2,2]` | Controls temporal downsampling<br>• Larger early: ↑ speed, ↓ fine detail<br>• Smaller late: ↑ refinement | **HIGH** | `--strides 4,2,2,2` |
| **`n_blocks_per_stage`** | `2` | `1`<br>`2` (baseline)<br>`3`<br>`[1,2,3,2]`* | Per-stage depth<br>• Fewer: ↓ params, ↑ speed, ↓ accuracy<br>• More: ↑ params, ↓ speed, ↑ accuracy | **MEDIUM** | `--n_blocks_per_stage 3` |
| **`kernel_size`** | `7` | `3`<br>`5`<br>`7` (baseline)<br>`9`<br>`11` | Receptive field per layer<br>• Smaller: ↑ speed, ↓ context<br>• Larger: ↓ speed, ↑ context | **MEDIUM** | `--kernel_size 9` |
| **`hidden_dims`** | `128,64` | `[64,32]`<br>`[128,64]` (baseline)<br>`[256,128,64]`<br>`[512,256,128]` | Regression head capacity<br>• Smaller: ↓ params, ↑ speed<br>• Larger: ↑ capacity, ↓ speed | **LOW** | `--hidden_dims 256,128,64` |
| **`dropout`** | `0.3` | `0.0`<br>`0.1`<br>`0.2`<br>`0.3` (baseline)<br>`0.4`<br>`0.5` | Regularization strength<br>• Lower: faster convergence, risk overfit<br>• Higher: slower convergence, better generalization | **MEDIUM** | `--dropout 0.2` |
| **`pooling_type`** | `avg` | `avg` (baseline)<br>`max` | Temporal aggregation method<br>• avg: smoother features<br>• max: peak-sensitive | **LOW** | `--pooling_type max` |
| **`norm`** | `std` | `none`<br>`std` (baseline)<br>`peak` | Input normalization<br>• none: raw amplitudes<br>• std: zero mean, unit variance<br>• peak: [-1, 1] range | **MEDIUM** | `--norm peak` |

*Requires modifying corresponding parameters (e.g., `n_stages=5` needs 5 values in other lists)

---

## Table 2: Recommended Ablation Experiments

| **Exp ID** | **Hypothesis** | **Parameter Changes** | **Full Command** | **Expected Result** |
|-----------|---------------|----------------------|------------------|---------------------|
| **E0** | Baseline | Default parameters | `--features_per_stage 8,16,32,64 --n_stages 4 --strides 2,2,2,2 --n_blocks_per_stage 2 --kernel_size 7 --hidden_dims 128,64 --dropout 0.3 --batch_size 32` | 227K params, baseline MAE/RMSE |
| **E1** | Width matters most | Increase width only | `--features_per_stage 16,32,64,128 --batch_size 16` | 4x params (908K), expect MAE ↓ 15-25% |
| **E2** | Depth matters most | Increase blocks only | `--n_blocks_per_stage 3` | 1.5x params (340K), expect MAE ↓ 5-10% |
| **E3** | Large kernels help | Larger kernels only | `--kernel_size 11` | 1.2x params (273K), expect MAE ↓ 3-8% |
| **E4** | Fast downsampling | Aggressive strides | `--strides 4,4,2,2` | Inference ↑ 40%, MAE ↑ 5-10% |
| **E5** | Shallow & wide | 3 stages, more features | `--n_stages 3 --features_per_stage 16,32,64 --strides 4,2,2` | 180K params, balanced trade-off |
| **E6** | Deep & narrow | 5 stages, fewer features | `--n_stages 5 --features_per_stage 6,12,24,48,96 --strides 2,2,2,2,2` | 300K params, better context |
| **E7** | Minimal model | All reductions | `--features_per_stage 6,12,24 --n_stages 3 --strides 4,2,2 --n_blocks_per_stage 1 --kernel_size 5 --hidden_dims 64,32 --batch_size 128` | 60K params, train 4x faster |
| **E8** | Max capacity | All increases | `--features_per_stage 16,32,64,128 --n_blocks_per_stage 3 --kernel_size 11 --hidden_dims 256,128,64 --dropout 0.2 --batch_size 16` | 1.1M params, best accuracy |
| **E9** | Pooling comparison | Max pooling | `--pooling_type max` | Same params, test sensitivity to peaks |
| **E10** | No dropout | Remove regularization | `--dropout 0.0` | Faster convergence, check overfit |
| **E11** | Peak normalization | Change normalization | `--norm peak` | Test if amplitude range matters |
| **E12** | Progressive depth | Variable blocks | `--n_blocks_per_stage 1,2,3,2` | 280K params, focus on middle stages |

---

## Usage

### Run Full Experiment Suite

```bash
#!/bin/bash
# Run full ablation study

MODEL_TYPE="umamba_mag_v2"
DATASET="STEAD"
BASE_CMD="uv run python -m src.my_project.main --mode train_mag --model_type $MODEL_TYPE --dataset $DATASET"

# Create logs directory
mkdir -p logs

# E0: Baseline
$BASE_CMD --features_per_stage 8,16,32,64 --n_stages 4 --strides 2,2,2,2 \
  --n_blocks_per_stage 2 --kernel_size 7 --hidden_dims 128,64 --dropout 0.3 \
  --batch_size 32 --epochs 100 --cuda 0 2>&1 | tee logs/E0_baseline.log

# E1: Width ablation
$BASE_CMD --features_per_stage 16,32,64,128 --batch_size 16 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E1_wide.log

# E2: Depth ablation
$BASE_CMD --n_blocks_per_stage 3 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E2_deep.log

# E3: Kernel ablation
$BASE_CMD --kernel_size 11 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E3_large_kernel.log

# E4: Stride ablation
$BASE_CMD --strides 4,4,2,2 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E4_fast_downsample.log

# E5: Shallow & wide
$BASE_CMD --n_stages 3 --features_per_stage 16,32,64 --strides 4,2,2 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E5_shallow_wide.log

# E6: Deep & narrow
$BASE_CMD --n_stages 5 --features_per_stage 6,12,24,48,96 --strides 2,2,2,2,2 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E6_deep_narrow.log

# E7: Minimal model
$BASE_CMD --features_per_stage 6,12,24 --n_stages 3 --strides 4,2,2 \
  --n_blocks_per_stage 1 --kernel_size 5 --hidden_dims 64,32 --batch_size 128 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E7_minimal.log

# E8: Max capacity
$BASE_CMD --features_per_stage 16,32,64,128 --n_blocks_per_stage 3 \
  --kernel_size 11 --hidden_dims 256,128,64 --dropout 0.2 --batch_size 16 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E8_max_capacity.log

# E9: Pooling ablation
$BASE_CMD --pooling_type max \
  --epochs 100 --cuda 0 2>&1 | tee logs/E9_max_pool.log

# E10: No dropout
$BASE_CMD --dropout 0.0 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E10_no_dropout.log

# E11: Peak normalization
$BASE_CMD --norm peak \
  --epochs 100 --cuda 0 2>&1 | tee logs/E11_peak_norm.log

# E12: Progressive depth
$BASE_CMD --n_blocks_per_stage 1,2,3,2 \
  --epochs 100 --cuda 0 2>&1 | tee logs/E12_progressive_depth.log
```

### Run Single Experiment

```bash
# Example: Run E1 (width ablation)
uv run python -m src.my_project.main --mode train_mag \
  --model_type umamba_mag_v2 --dataset STEAD \
  --features_per_stage 16,32,64,128 \
  --batch_size 16 --epochs 100 --cuda 0
```

### Evaluate All Experiments

```bash
# After training, evaluate each experiment
for exp in E0 E1 E2 E3 E4 E5 E6 E7 E8 E9 E10 E11 E12; do
    MODEL_PATH=$(find src/trained_weights -name "${exp}_*" -type d | head -1)/model_best.pt
    if [ -f "$MODEL_PATH" ]; then
        echo "Evaluating $exp..."
        uv run python -m src.my_project.main --mode eval_mag \
          --model_type umamba_mag_v2 --dataset STEAD \
          --model_path $MODEL_PATH --batch_size 64 \
          2>&1 | tee logs/${exp}_eval.log
    fi
done
```

---

## Metrics to Track

| **Metric** | **Collection Method** | **Analysis Goal** |
|-----------|----------------------|-------------------|
| **Parameters** | Model summary | Measure model size |
| **Training Time/Epoch** | Training logs | Efficiency comparison |
| **Inference Time** | Evaluation output | Production readiness |
| **MAE** | Evaluation metrics | Primary accuracy metric |
| **RMSE** | Evaluation metrics | Error magnitude |
| **R²** | Evaluation metrics | Goodness of fit |
| **Train Loss** | Training history | Convergence behavior |
| **Val Loss** | Training history | Overfitting detection |
| **Epochs to Converge** | Training history | Training efficiency |
| **Memory Usage** | GPU monitoring | Hardware requirements |

---

## Analysis Strategy

1. **Run all experiments** - Use the bash script above
2. **Collect metrics** - Parse logs into CSV format
3. **Normalize by baseline** - Calculate `(metric_exp - metric_E0) / metric_E0 * 100`
4. **Identify best performers** - Sort by MAE ↓, RMSE ↓, R² ↑
5. **Analyze trade-offs** - Plot accuracy vs params, accuracy vs speed
6. **Statistical significance** - T-test between top 3 and baseline
7. **Combine best features** - Create optimal configuration from insights

---

## Expected Outcomes

### High-Priority Parameters (Table 1)
- **`features_per_stage`**: Most impactful for accuracy; wider networks consistently improve performance
- **`n_stages`**: Controls model depth and receptive field; balance between context and efficiency
- **`strides`**: Critical for inference speed; aggressive downsampling trades accuracy for speed

### Recommended Next Steps
1. Start with **E0** (baseline) to establish metrics
2. Run **E1**, **E2**, **E3** to test individual parameter impacts
3. Combine insights from top performers into final configuration
4. Test final configuration with extended training (200+ epochs)

---

## Notes

- All experiments assume STEAD dataset with 100Hz sampling rate
- Adjust `--batch_size` based on GPU memory availability
- Monitor GPU memory usage for experiments E1, E8 (large models)
- Consider using mixed precision training (`--amp`) for faster training
- Save checkpoints every 5-10 epochs for analysis
