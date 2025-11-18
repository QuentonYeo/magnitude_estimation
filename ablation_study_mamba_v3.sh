#!/bin/bash
set -e  # Exit on error

# ============================================================================
# UMamba V3 Ablation Study - Sequential Execution with Proper GPU Management
# ============================================================================
# Tests UMamba V3 triple-head architecture with:
# - Multi-scale feature fusion (scalar head)
# - Temporal magnitude predictions (auxiliary task)
# - Optional uncertainty weighting (Kendall & Gal 2017)
# - Flexible pooling strategies (avg/max/hybrid)
# ============================================================================

MODEL_TYPE="umamba_mag_v3"
DATASET="STEAD"
BASE_CMD="uv run python -m src.my_project.main --mode train_mag --model_type $MODEL_TYPE --dataset $DATASET"
EPOCHS=150
MAX_PARALLEL=1  # Maximum parallel jobs (one per GPU)

# Create directories
mkdir -p logs results checkpoints
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
SUMMARY_FILE="results/ablation_summary_${TIMESTAMP}.txt"

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

# Count currently running background jobs
count_jobs() {
    jobs -r | wc -l
}

# Wait until we have fewer than MAX_PARALLEL jobs running
wait_for_slot() {
    while [ $(count_jobs) -ge $MAX_PARALLEL ]; do
        sleep 5
    done
}

# Get next available GPU
get_next_gpu() {
    local gpu0_count=$(ps aux | grep "cuda 0" | grep -v grep | wc -l)
    local gpu1_count=$(ps aux | grep "cuda 1" | grep -v grep | wc -l)
    
    if [ $gpu0_count -le $gpu1_count ]; then
        echo 0
    else
        echo 1
    fi
}

# Log experiment details
log_experiment() {
    local exp_id=$1
    local exp_name=$2
    local hypothesis=$3
    
    echo "=========================================="
    echo "Experiment: $exp_id - $exp_name"
    echo "Hypothesis: $hypothesis"
    echo "Started: $(date)"
    echo "=========================================="
}

# Run experiment with automatic GPU allocation
run_experiment() {
    local exp_id=$1
    local exp_name=$2
    local hypothesis=$3
    shift 3
    local args="$@"
    
    # Wait for available slot
    wait_for_slot
    
    # Get GPU
    local gpu=2
    
    log_experiment "$exp_id" "$exp_name" "$hypothesis"
    
    # Run in background
    (
        echo "[$exp_id] Running on GPU $gpu" | tee -a "$SUMMARY_FILE"
        
        $BASE_CMD $args --cuda $gpu --epochs $EPOCHS --quiet --early_stopping_patience 5 \
            2>&1 | tee logs/${exp_id}_${exp_name}.log
        
        # Extract metrics
        local exit_code=$?
        if [ $exit_code -eq 0 ]; then
            echo "[$exp_id] SUCCESS on GPU $gpu at $(date)" | tee -a "$SUMMARY_FILE"
            grep -E "MAE|RMSE|R2|Parameters|Training time" logs/${exp_id}_${exp_name}.log >> results/${exp_id}_metrics.txt 2>/dev/null || true
        else
            echo "[$exp_id] FAILED on GPU $gpu at $(date)" | tee -a "$SUMMARY_FILE"
        fi
    ) &
    
    sleep 2  # Small delay to prevent race conditions
}

# Wait for all background jobs to complete
wait_all() {
    echo "Waiting for all experiments to complete..."
    wait
    echo "All experiments finished at $(date)" | tee -a "$SUMMARY_FILE"
}

# ============================================================================
# ABLATION EXPERIMENTS
# ============================================================================

echo "Ablation study started at $(date)" | tee "$SUMMARY_FILE"
echo "Max parallel jobs: $MAX_PARALLEL" | tee -a "$SUMMARY_FILE"
echo "" | tee -a "$SUMMARY_FILE"

# ============================================================================
# BASELINE - UMamba V3 Default Configuration (Dual-Head)
# ============================================================================
# NOTE: This is dual-head mode (scalar + temporal) WITHOUT uncertainty head.
# The default V3 configuration does NOT include --use_uncertainty flag.

# run_experiment "E0" "baseline_v3_dual_head" \
#     "Baseline V3 dual-head with multi-scale fusion (scalar + temporal, NO uncertainty)" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 192,96 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm std \
#     --scalar_weight 0.7 \
#     --temporal_weight 0.25 \
#     --batch_size 32

# ============================================================================
# EXPERIMENT 1 - Triple-Head with Uncertainty Weighting
# ============================================================================
# Add uncertainty head for automatic sample weighting (Kendall & Gal 2017)
# DISABLED: The uncertainty-weighted loss formulation causes negative loss values
# The Kendall & Gal formulation: 0.5 * precision * error + 0.5 * log_var
# produces negative losses when log_var < 0 (which happens when variance < 1)
# This breaks training, early stopping, and learning rate scheduling.

# run_experiment "E1" "triple_head_with_uncertainty" \
#     "V3 triple-head: add uncertainty head for automatic sample weighting" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 192,96 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm std \
#     --scalar_weight 0.7 \
#     --temporal_weight 0.25 \
#     --use_uncertainty \
#     --batch_size 32

# ============================================================================
# KERNEL SIZE ABLATIONS (Dual-Head)
# ============================================================================

run_experiment "E2a" "small_kernel" \
    "Smaller kernel: local features only" \
    --features_per_stage 8,16,32,64 \
    --n_stages 4 \
    --strides 2,2,2,2 \
    --n_blocks_per_stage 2 \
    --kernel_size 3 \
    --hidden_dims 192,96 \
    --dropout 0.3 \
    --pooling_type avg \
    --norm std \
    --scalar_weight 0.7 \
    --temporal_weight 0.25 \
    --batch_size 32

run_experiment "E2b" "medium_kernel" \
    "Medium kernel: balanced receptive field" \
    --features_per_stage 8,16,32,64 \
    --n_stages 4 \
    --strides 2,2,2,2 \
    --n_blocks_per_stage 2 \
    --kernel_size 5 \
    --hidden_dims 192,96 \
    --dropout 0.3 \
    --pooling_type avg \
    --norm std \
    --scalar_weight 0.7 \
    --temporal_weight 0.25 \
    --batch_size 32

run_experiment "E2c" "large_kernel" \
    "Larger kernel: more context per layer" \
    --features_per_stage 8,16,32,64 \
    --n_stages 4 \
    --strides 2,2,2,2 \
    --n_blocks_per_stage 2 \
    --kernel_size 11 \
    --hidden_dims 192,96 \
    --dropout 0.3 \
    --pooling_type avg \
    --norm std \
    --scalar_weight 0.7 \
    --temporal_weight 0.25 \
    --batch_size 32

# ============================================================================
# WIDTH ABLATIONS (features_per_stage) - Dual-Head
# ============================================================================

run_experiment "E3a" "wide" \
    "Wider network: more capacity, better accuracy" \
    --features_per_stage 16,32,64,128 \
    --n_stages 4 \
    --strides 2,2,2,2 \
    --n_blocks_per_stage 2 \
    --kernel_size 7 \
    --hidden_dims 192,96 \
    --dropout 0.3 \
    --pooling_type avg \
    --norm std \
    --scalar_weight 0.7 \
    --temporal_weight 0.25 \
    --batch_size 16


# ============================================================================
# WAIT FOR ALL EXPERIMENTS
# ============================================================================

wait_all

# ============================================================================
# EXTRACT METRICS
# ============================================================================

echo "" | tee -a "$SUMMARY_FILE"
echo "=========================================="
echo "EXTRACTING METRICS"
echo "=========================================="

# Create CSV header
echo "ExpID,ExpName,MAE,RMSE,R2,Parameters,TrainingTime" > results/metrics_${TIMESTAMP}.csv

# Parse each experiment's results
for log_file in logs/E*.log; do
    if [ -f "$log_file" ]; then
        exp_name=$(basename "$log_file" .log)
        mae=$(grep -oP "MAE: \K[0-9.]+" "$log_file" | tail -1 || echo "N/A")
        rmse=$(grep -oP "RMSE: \K[0-9.]+" "$log_file" | tail -1 || echo "N/A")
        r2=$(grep -oP "R2: \K[0-9.]+" "$log_file" | tail -1 || echo "N/A")
        params=$(grep -oP "Parameters: \K[0-9,]+" "$log_file" | tail -1 || echo "N/A")
        time=$(grep -oP "Training time: \K[0-9.]+" "$log_file" | tail -1 || echo "N/A")
        
        echo "$exp_name,$mae,$rmse,$r2,$params,$time" >> results/metrics_${TIMESTAMP}.csv
    fi
done

echo "Metrics extracted to results/metrics_${TIMESTAMP}.csv" | tee -a "$SUMMARY_FILE"
echo "Ablation study completed successfully!" | tee -a "$SUMMARY_FILE"