# Copilot Instructions - Seismic Magnitude Estimation

This repository implements a unified framework for **seismic phase detection and magnitude estimation** using deep learning. Multiple model architectures (PhaseNet variants, transformers, and Mamba-based models) share common infrastructure for training, evaluation, and data loading.

## Architecture Overview

### Unified Entry Point
- **All workflows** run through `src/my_project/main.py` with `--mode` flags: `train_phase`, `train_mag`, `eval_phase`, `eval_mag`
- **Model selection** via `--model_type`: `phasenet`, `phasenet_mag`, `eqtransformer_mag`, `vit_mag`, `amag_v2`, `amag_v3`, `umamba_mag`, `umamba_mag_v2`, `umamba_mag_v3`, `magnet`
- **Datasets** supported: `ETHZ`, `STEAD`, `GEOFON`, `MLAAPDE` (via SeisBench library)

### Core Components
```
src/my_project/
├── main.py                          # CLI entry point, model factory functions
├── models/                          # Model implementations (each in subdirectory)
│   ├── phasenet_mag/               # PhaseNet for magnitude regression
│   ├── AMAG_v2/, AMAG_v3/          # Advanced magnitude models
│   ├── EQTransformer/              # Transformer-based (30s windows)
│   ├── ViT/                        # Vision Transformer
│   ├── UMamba_mag/, UMamba_mag_v2/ # Mamba state-space models
│   └── UMamba_mag_v3/              # Triple-head (scalar+temporal+uncertainty)
├── loaders/
│   ├── data_loader.py              # Augmentation pipelines
│   └── magnitude_labellers.py      # MagnitudeLabeller, MagnitudeLabellerAMAG
└── utils/
    ├── unified_training.py         # Train/eval routing to model-specific functions
    └── utils.py                    # Plotting, SNR analysis
```

## Critical Patterns

### Model Architecture Pattern
All models inherit from `seisbench.models.base.WaveformModel` with standard structure:
```python
class YourModel(WaveformModel):
    def __init__(self, in_channels=3, sampling_rate=100, norm="std", ...):
        super().__init__()
        # norm: "std" (zero mean, unit variance) or "peak" ([-1,1])
        self.in_channels = in_channels
        self.norm = norm
        # Build layers...
    
    def forward(self, x):
        # Input: (batch, 3, time_steps) - Z, N, E channels
        # Output: scalar magnitude or (batch, time_steps) temporal
```

### Training Function Pattern
Each model has `train.py` and `evaluate.py` with consistent signatures:
```python
def train_your_model(
    data: BenchmarkDataset,
    epochs: int = 50,
    learning_rate: float = 1e-3,
    batch_size: int = 64,
    warmup_epochs: int = 5,  # Linear LR warmup (10%→100%)
    scheduler_factor: float = 0.5,
    early_stopping_patience: int = 15,
    cuda: int = 0,
    quiet: bool = False,
    **kwargs
) -> WaveformModel:
    # 1. Create model instance
    # 2. Setup augmentations via data_loader.get_magnitude_augmentation()
    # 3. Configure optimizer (Adam/AdamW) + ReduceLROnPlateau scheduler
    # 4. Training loop with warmup: lr = base_lr * min(1.0, epoch / warmup_epochs)
    # 5. Save best model to src/trained_weights/{model_name}_{dataset}_{timestamp}/
```

### Data Augmentation System
**Magnitude labeling** differs by model type:
- **Standard models** (`MagnitudeLabeller`): `label[0:onset] = 0`, `label[onset:] = magnitude`
- **AMAG models** (`MagnitudeLabellerAMAG`): `label[0:onset] = 0`, `label[onset:] = magnitude + 1` (subtract 1 at inference)

Augmentation pipeline (from `data_loader.py`):
```python
augmentations = [
    sbg.WindowAroundSample(phase_dict.keys(), samples_before=3001, ...),
    sbg.RandomWindow(windowlen=3001, strategy="pad"),
    sbg.ChangeDtype(np.float32),
    MagnitudeLabeller(phase_dict=phase_dict)  # or MagnitudeLabellerAMAG
]
```

### Model-Specific Defaults
Transformers use lower LR and smaller batches:
- **EQTransformer/ViT**: `learning_rate=1e-4`, `batch_size=64`, `optimizer=AdamW`
- **Mamba models**: `learning_rate=1e-3`, `batch_size=64`, `optimizer=AdamW`, `warmup_epochs=5`
- **PhaseNetMag**: `learning_rate=1e-4`, `batch_size=256`, `optimizer=Adam`

## Development Workflows

### Run Experiments
```bash
# Standard usage with uv package manager
uv run python -m src.my_project.main --mode train_mag --model_type umamba_mag_v3 \
    --dataset STEAD --epochs 150 --cuda 0 --quiet

# Evaluation with plots (3 examples by default)
uv run python -m src.my_project.main --mode eval_mag --model_type umamba_mag_v3 \
    --dataset STEAD --model_path path/to/model_best.pt --plot --num_examples 5
```

### Ablation Studies
Use `ablation_study_mamba_v3.sh` as template:
- GPU allocation via `--cuda` flag (0, 1, or 2)
- Sequential execution with `wait_for_slot()` to limit parallel jobs
- Metrics extracted to `results/E{id}_metrics.txt`
- Logs to `logs/E{id}_{name}.log`

### Add New Models
1. Create `src/my_project/models/YourModel/` with `model.py`, `train.py`, `evaluate.py`
2. Add factory functions in `main.py`:
   - `extract_model_params()` - parse CLI args
   - `create_magnitude_model()` - instantiate model
   - `get_model_name()` - generate checkpoint name
3. Import train/eval functions in `utils/unified_training.py`
4. Add model type to `train_magnitude_unified()` and `evaluate_magnitude_unified()`

### Parameter Extraction Pattern
Models with list-like parameters (e.g., UMamba `features_per_stage`):
```python
if hasattr(args, "features_per_stage") and args.features_per_stage:
    try:
        params["features_per_stage"] = [int(x.strip()) for x in args.features_per_stage.split(",")]
    except:
        print(f"Warning: Could not parse features_per_stage, using default")
```

## Project-Specific Conventions

### Naming Conventions
- **Model checkpoints**: `{ModelName}_{Dataset}_{YYYYMMDD_HHMMSS}/model_best.pt`
- **Results files**: `results/E{exp_id}_metrics.txt` for experiments
- **Log files**: `logs/E{exp_id}_{description}.log`

### UMamba Model Versions
- **V1** (`umamba_mag`): U-Net encoder-decoder, ~560K params, temporal output → scalar
- **V2** (`umamba_mag_v2`): Encoder-only + pooling, ~220K params, direct scalar output
- **V3** (`umamba_mag_v3`): Triple-head architecture (~216K params):
  - **Scalar head**: Multi-scale fusion (concatenates ALL stage features) + MLP
  - **Temporal head**: 1x1 conv per-timestep predictions
  - **Uncertainty head** (optional): Learns log-variance for Kendall & Gal uncertainty weighting
  - **Loss weights**: `--scalar_weight 0.7 --temporal_weight 0.25` (default)

### Multi-Scale Fusion (V3 Specific)
V3's scalar head concatenates pooled features from **all** encoder stages:
```python
# From UMamba_mag_v3/model.py
fused = torch.cat([pool(f) for f in encoder_features], dim=1)
# e.g., [8, 16, 32, 64] → 120 channels → MLP → scalar
```

### CUDA Mamba Dependencies
- **Critical**: `causal-conv1d` and `mamba-ssm` must match PyTorch+CUDA versions
- Pre-built wheels in `pyproject.toml` for `torch==2.4.0` + CUDA 12.2
- Fallback: `pyproject-cuda.txt` for manual dependency resolution

## Common Tasks

### Debug Training Issues
1. Check augmentation pipeline in `loaders/data_loader.py` - ensure correct labeller
2. Verify input normalization: `norm="std"` (most models) vs `norm="peak"` (rare)
3. Inspect first batch: `MagnitudeLabeller(debug=True)` logs onset/magnitude
4. For AMAG models: subtract 1 from predictions (`label = mag + 1` during training)

### Modify Loss Functions
Training loops in `models/{model}/train.py`:
```python
def train_loop(dataloader, epoch):
    for batch in dataloader:
        # Custom loss weighting example (V3):
        loss = scalar_weight * scalar_loss + temporal_weight * temporal_loss
        if use_uncertainty:
            loss += uncertainty_regularization
```

### Add CLI Parameters
1. Add to `argparse` in `main.py` (around line 800+)
2. Extract in `extract_model_params()` with default fallback
3. Pass to model constructor in `create_magnitude_model()`

### Plot Training History
```bash
uv run python -m src.my_project.main --mode plot_history \
    --model_path path/to/model_directory --plot
```
Uses `utils/utils.py:plot_training_history()` to visualize loss curves.

## Documentation References
- **Ablation methodology**: `UMAMBA_ABLATION_STUDY.md` - parameter grid, experiment design
- **Training analysis**: `TRAINING_SCHEDULER_ANALYSIS.md` - warmup/scheduler rationale
- **Model comparison**: `UMAMBA_V1_VS_V2_COMPARISON.md` - architecture evolution
- **Evaluation metrics**: `EVALUATION_METHODOLOGY_COMPARISON.md` - MAE, RMSE, R² definitions

## Key Insights for AI Agents

1. **Always use `uv run python -m`** - not `python` directly (virtual env management)
2. **Match training parameters at eval** - model architecture must align with saved checkpoint
3. **Check dataset in checkpoint path** - models are dataset-specific (`_STEAD_`, `_ETHZ_`)
4. **UMamba V3 is current SOTA** - prefer over V1/V2 for new experiments
5. **Warmup is critical for stability** - don't disable unless testing convergence
6. **Multi-GPU via `--cuda` flag** - not automatic, must specify device explicitly
7. **Response formatting** - do not create new code files or markdown reports unless explicitly asked for

## Testing
No formal test suite. Validation via:
- Ablation scripts (`ablation_study*.sh`) - systematic parameter sweeps
- Evaluation mode (`--mode eval_mag`) - MAE/RMSE/R² on test split
- Manual inspection of plots (`--plot --num_examples 5`)
