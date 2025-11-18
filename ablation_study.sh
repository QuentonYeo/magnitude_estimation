#!/bin/bash
set -e  # Exit on error

# ============================================================================
# UMamba Ablation Study - Sequential Execution with Proper GPU Management
# ============================================================================

MODEL_TYPE="umamba_mag_v2"
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
    local gpu=3
    
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
# BASELINE
# ============================================================================

# COMMENTED OUT - Running only E4a to E5c
# run_experiment "E0" "baseline" \
#     "Baseline configuration for comparison" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# ============================================================================
# WIDTH ABLATIONS (features_per_stage)
# ============================================================================

# COMMENTED OUT - Running only E4a to E5c
# # COMMENTED OUT - Running only E4a to E5c
# run_experiment "E1a" "narrow" \
#     "Narrower network: fewer parameters, faster training" \
#     --features_per_stage 6,12,24,48 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 128,64 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm std \
#     --batch_size 32

# run_experiment "E1b" "wide" \
#     "Wider network: more capacity, better accuracy" \
    # --features_per_stage 16,32,64,128 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 16

# run_experiment "E1c" "extra_wide" \
#     "Extra wide network: maximum width impact" \
    # --features_per_stage 12,24,48,96 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 24

# ============================================================================
# DEPTH ABLATIONS (n_blocks_per_stage)
# ============================================================================

# COMMENTED OUT - Running only E4a to E5c
# run_experiment "E2a" "shallow_blocks" \
#     "Fewer blocks per stage: faster, less capacity" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 1 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# run_experiment "E2b" "deep_blocks" \
#     "More blocks per stage: deeper processing" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 3 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# Note: E2c removed - n_blocks_per_stage doesn't support comma-separated values

# ============================================================================
# STAGE ABLATIONS (n_stages)
# ============================================================================

# COMMENTED OUT - Running only E4a to E5c
# run_experiment "E3a" "shallow_stages" \
#     "Fewer stages: less hierarchical processing" \
    # --features_per_stage 8,16,32 \
    # --n_stages 3 \
    # --strides 4,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# run_experiment "E3b" "deep_stages" \
#     "More stages: deeper hierarchy, larger context" \
    # --features_per_stage 6,12,24,48,96 \
    # --n_stages 5 \
    # --strides 2,2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# ============================================================================
# KERNEL SIZE ABLATIONS
# ============================================================================

# COMMENTED OUT - Running only E6a to E11d
# run_experiment "E4a" "small_kernel" \
#     "Smaller kernel: local features only" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 3 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# run_experiment "E4b" "medium_kernel" \
#     "Medium kernel: balanced receptive field" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 5 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# run_experiment "E4c" "large_kernel" \
#     "Larger kernel: more context per layer" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 11 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# ============================================================================
# STRIDE ABLATIONS
# ============================================================================

# COMMENTED OUT - Running only E6a to E11d
# run_experiment "E5a" "aggressive_early_stride" \
#     "Fast early downsampling: speed over detail" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 4,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# run_experiment "E5b" "aggressive_all_stride" \
#     "Very fast downsampling: maximum speed" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 4,4,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# run_experiment "E5c" "gentle_late_stride" \
#     "Preserve late-stage resolution" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,1 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.3 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# ============================================================================
# DROPOUT ABLATIONS
# ============================================================================

# run_experiment "E6a" "no_dropout" \
#     "No regularization: test overfitting" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.0 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# run_experiment "E6b" "low_dropout" \
#     "Light regularization" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.1 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# run_experiment "E6c" "high_dropout" \
#     "Heavy regularization" \
    # --features_per_stage 8,16,32,64 \
    # --n_stages 4 \
    # --strides 2,2,2,2 \
    # --n_blocks_per_stage 2 \
    # --kernel_size 7 \
    # --hidden_dims 128,64 \
    # --dropout 0.5 \
    # --pooling_type avg \
    # --norm std \
    # --batch_size 32

# ============================================================================
# REGRESSION HEAD ABLATIONS (hidden_dims)
# ============================================================================

# run_experiment "E7a" "small_head" \
#     "Smaller regression head: faster, less capacity" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 64,32 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm std \
#     --batch_size 32

# run_experiment "E7b" "large_head" \
#     "Larger regression head: more capacity" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 256,128,64 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm std \
#     --batch_size 32

# run_experiment "E7c" "deep_head" \
#     "Deeper regression head: more layers" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 512,256,128,64 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm std \
#     --batch_size 32

# ============================================================================
# NORMALIZATION ABLATIONS
# ============================================================================

# Note: E8a removed - 'none' is not a valid option for --norm (only 'std' or 'peak')

# run_experiment "E8b" "peak_norm" \
#     "Peak normalization: [-1, 1] range" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 128,64 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm peak \
#     --batch_size 32

# ============================================================================
# POOLING ABLATIONS
# ============================================================================

# run_experiment "E9a" "max_pooling" \
#     "Max pooling: peak-sensitive aggregation" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 128,64 \
#     --dropout 0.3 \
#     --pooling_type max \
#     --norm std \
#     --batch_size 32

# ============================================================================
# BATCH SIZE ABLATIONS
# ============================================================================

# run_experiment "E10a" "small_batch" \
#     "Smaller batches: noisier gradients, possible better generalization" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 128,64 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm std \
#     --batch_size 16

# run_experiment "E10b" "large_batch" \
#     "Larger batches: stabler gradients, faster training" \
#     --features_per_stage 8,16,32,64 \
#     --n_stages 4 \
#     --strides 2,2,2,2 \
#     --n_blocks_per_stage 2 \
#     --kernel_size 7 \
#     --hidden_dims 128,64 \
#     --dropout 0.3 \
#     --pooling_type avg \
#     --norm std \
#     --batch_size 64

# ============================================================================
# COMBINED CONFIGURATIONS
# ============================================================================

run_experiment "E11a" "minimal" \
    "Minimal model: fast training and inference" \
    --features_per_stage 6,12,24 \
    --n_stages 3 \
    --strides 4,2,2 \
    --n_blocks_per_stage 1 \
    --kernel_size 5 \
    --hidden_dims 64,32 \
    --dropout 0.3 \
    --pooling_type avg \
    --norm std \
    --batch_size 128

run_experiment "E11b" "maximal" \
    "Maximum capacity: best accuracy target" \
    --features_per_stage 16,32,64,128 \
    --n_stages 4 \
    --strides 2,2,2,2 \
    --n_blocks_per_stage 3 \
    --kernel_size 11 \
    --hidden_dims 256,128,64 \
    --dropout 0.2 \
    --pooling_type avg \
    --norm std \
    --batch_size 32

run_experiment "E11c" "shallow_wide" \
    "Shallow & wide: balanced trade-off" \
    --features_per_stage 16,32,64 \
    --n_stages 3 \
    --strides 4,2,2 \
    --n_blocks_per_stage 2 \
    --kernel_size 7 \
    --hidden_dims 128,64 \
    --dropout 0.3 \
    --pooling_type avg \
    --norm std \
    --batch_size 24

run_experiment "E11d" "deep_narrow" \
    "Deep & narrow: large context, efficient" \
    --features_per_stage 6,12,24,48,96 \
    --n_stages 5 \
    --strides 2,2,2,2,2 \
    --n_blocks_per_stage 2 \
    --kernel_size 7 \
    --hidden_dims 128,64 \
    --dropout 0.3 \
    --pooling_type avg \
    --norm std \
    --batch_size 32

# ============================================================================
# HYBRID COMBINATIONS
# ============================================================================

run_experiment "E12a" "hybrid_wide_shallow_nodropout" \
    "Hybrid: Wide (E1b) + Shallow (E2a) + No Dropout (E6a)" \
    --features_per_stage 16,32,64,128 \
    --n_stages 4 \
    --strides 2,2,2,2 \
    --n_blocks_per_stage 1 \
    --kernel_size 7 \
    --hidden_dims 128,64 \
    --dropout 0.0 \
    --pooling_type avg \
    --norm std \
    --batch_size 32

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