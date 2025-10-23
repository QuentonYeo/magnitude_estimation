# Machine Learning in Seismic Magnitude Estimation

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Model Configuration](#model-configuration)
- [Complete Command Reference](#complete-command-reference)
- [Unified Architecture](#unified-architecture)
- [Hyperparameter Tuning](#hyperparameter-tuning)

## Description

This repository provides a unified framework for seismic phase detection and magnitude estimation using deep learning models. The system supports multiple model architectures with configurable parameters, including PhaseNet variants and advanced magnitude estimation networks.

## Installation

1. After cloning the repo, run the following depending on the platform

- If CPU based: `uv sync --extra cpu`
- If GPU based: `uv sync --extra gpu`

2. Verify CUDA installation:

   ```bash
   uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

3. If a previous version of custom seisbench persists, run `uv lock --upgrade` then `uv sync`

4. If the system still does not install the cuda version correctly, copy the contents of `pyproject-cuda.txt` into the pyproject and try again with `uv sync`.

## Quick Start

The main script provides a unified interface for all seismic analysis workflows. Commands are organized by mode (`train_phase`, `train_mag`, `eval_phase`, `eval_mag`) with `--model_type` specifying the specific model.

#### Basic Training

```bash
# Train a PhaseNet model for phase detection
uv run python -m src.my_project.main --mode train_phase --model_type phasenet --dataset ETHZ --epochs 10

# Train a magnitude estimation model
uv run python -m src.my_project.main --mode train_mag --model_type amag_v2 --dataset ETHZ --epochs 20

# Train with custom learning rate, batch size, and warmup
uv run python -m src.my_project.main --mode train_mag --model_type phasenet_mag --learning_rate 0.002 --batch_size 64 --warmup_epochs 10
```

#### Basic Evaluation

```bash
# Evaluate a phase detection model
uv run python -m src.my_project.main --mode eval_phase --model_type phasenet --dataset ETHZ --model_path path/to/model.pt

# Evaluate a magnitude model with plots
uv run python -m src.my_project.main --mode eval_mag --model_type amag_v2 --dataset ETHZ --model_path path/to/model.pt --plot
```

#### Getting Help

```bash
# See all available options
uv run python -m src.my_project.main --help
```

> 📖 **For complete command reference and model configuration, see the sections below.**

## Model Configuration

All models (except the standard PhaseNet) support configurable parameters that can be set via command line arguments. This allows easy experimentation with different model architectures and capacities.

**Universal Training Parameters for Magnitude Models**: All magnitude models (`phasenet_mag`, `eqtransformer_mag`, `amag_v2`) now support configurable `learning_rate`, `batch_size`, and `warmup_epochs` parameters for optimized training with learning rate warmup scheduling.

### Available Models and Parameters

#### 1. PhaseNetLSTM (`phasenet_lstm`)

**Description**: PhaseNet with LSTM layers for enhanced temporal modeling.

**Configurable Parameters:**

- `--filter_factor` (int, default=1): Controls model capacity (multiplies filter sizes)
- `--lstm_hidden_size` (int, default=auto): LSTM hidden size (None=auto-calculated)
- `--lstm_num_layers` (int, default=1): Number of LSTM layers
- `--lstm_bidirectional` (flag, default=True): Use bidirectional LSTM

**Examples:**

```bash
# Default configuration
uv run python -m src.my_project.main --mode train_phase --model_type phasenet_lstm --dataset ETHZ

# High capacity model
uv run python -m src.my_project.main --mode train_phase --model_type phasenet_lstm \
       --dataset ETHZ --filter_factor 2 --lstm_hidden_size 256 --lstm_num_layers 3

# Lightweight model
uv run python -m src.my_project.main --mode train_phase --model_type phasenet_lstm \
       --dataset ETHZ --lstm_hidden_size 64 --lstm_num_layers 1
```

#### 2. PhaseNetConvLSTM (`phasenet_conv_lstm`)

**Description**: PhaseNet with Convolutional LSTM layers for spatial-temporal modeling.

**Configurable Parameters:**

- `--filter_factor` (int, default=1): Controls model capacity
- `--convlstm_hidden` (int, default=64): ConvLSTM hidden size

**Examples:**

```bash
# Default configuration
uv run python -m src.my_project.main --mode train_phase --model_type phasenet_conv_lstm --dataset ETHZ

# High capacity model
uv run python -m src.my_project.main --mode train_phase --model_type phasenet_conv_lstm \
       --dataset ETHZ --filter_factor 2 --convlstm_hidden 128
```

#### 3. PhaseNetMag (`phasenet_mag`)

**Description**: PhaseNet architecture adapted for magnitude regression.

**Configurable Parameters:**

- `--filter_factor` (int, default=1): Controls model capacity
- `--norm` (str, default="std"): Normalization method ("std" or "peak")
- `--learning_rate` (float, default=0.001): Learning rate for training
- `--batch_size` (int, default=32): Batch size for training
- `--warmup_epochs` (int, default=5): Linear warmup epochs (10% → 100% learning rate)

**Examples:**

```bash
# Default configuration with standard normalization
uv run python -m src.my_project.main --mode train_mag --model_type phasenet_mag --dataset ETHZ

# High capacity model with peak normalization and custom training params
uv run python -m src.my_project.main --mode train_mag --model_type phasenet_mag \
       --dataset ETHZ --filter_factor 2 --norm peak --learning_rate 0.002 --batch_size 64 --warmup_epochs 10

# Fast training with larger batch and extended warmup
uv run python -m src.my_project.main --mode train_mag --model_type phasenet_mag \
       --dataset ETHZ --batch_size 128 --warmup_epochs 15
```

#### 4. EQTransformerMag (`eqtransformer_mag`)

**Description**: Transformer-based magnitude estimation model with 30-second input windows and attention mechanisms.

**Configurable Parameters:**

- `--n_class` (int, default=1): Number of output classes (magnitude regression)
- `--phases` (str, default="PS"): Phase types to use
- `--sampling_rate` (float, default=100.0): Sampling rate in Hz
- `--cnn_blocks` (int, default=5): Number of CNN blocks in encoder
- `--lstm_blocks` (int, default=2): Number of LSTM blocks
- `--transformer_d_model` (int, default=128): Transformer model dimension
- `--transformer_nhead` (int, default=8): Number of attention heads
- `--transformer_num_encoder_layers` (int, default=4): Number of transformer encoder layers
- `--transformer_dim_feedforward` (int, default=512): Transformer feedforward dimension
- `--learning_rate` (float, default=0.0001): Learning rate for training (transformer-optimized)
- `--batch_size` (int, default=16): Batch size for training (transformer-optimized)
- `--warmup_epochs` (int, default=5): Linear warmup epochs for training stability

**Examples:**

```bash
# Default configuration (30-second input windows)
uv run python -m src.my_project.main --mode train_mag --model_type eqtransformer_mag --dataset ETHZ

# High capacity model with larger transformer and extended warmup
uv run python -m src.my_project.main --mode train_mag --model_type eqtransformer_mag \
       --dataset ETHZ --transformer_d_model 256 --transformer_nhead 16 --transformer_num_encoder_layers 6 \
       --learning_rate 0.0001 --batch_size 16 --warmup_epochs 10

# Lightweight model with smaller transformer
uv run python -m src.my_project.main --mode train_mag --model_type eqtransformer_mag \
       --dataset ETHZ --transformer_d_model 64 --transformer_nhead 4 --transformer_num_encoder_layers 2 \
       --batch_size 32 --warmup_epochs 3
```

#### 5. MagnitudeNet (`amag_v2`)

**Description**: Advanced magnitude estimation model with LSTM and attention mechanisms.

**Configurable Parameters:**

- `--filter_factor` (int, default=1): Controls model capacity
- `--lstm_hidden` (int, default=128): LSTM hidden size
- `--lstm_layers` (int, default=2): Number of LSTM layers
- `--dropout` (float, default=0.2): Dropout rate
- `--learning_rate` (float, default=0.001): Learning rate for training
- `--batch_size` (int, default=64): Batch size for training
- `--warmup_epochs` (int, default=5): Linear warmup epochs for training stability

**Examples:**

```bash
# Default configuration
uv run python -m src.my_project.main --mode train_mag --model_type amag_v2 --dataset ETHZ

# High capacity model with custom training parameters
uv run python -m src.my_project.main --mode train_mag --model_type amag_v2 \
       --dataset ETHZ --filter_factor 2 --lstm_hidden 256 --lstm_layers 3 --dropout 0.3 \
       --learning_rate 0.001 --batch_size 64 --warmup_epochs 8

# Lightweight model with faster training
uv run python -m src.my_project.main --mode train_mag --model_type amag_v2 \
       --dataset ETHZ --lstm_hidden 64 --lstm_layers 1 --dropout 0.1 \
       --batch_size 128 --warmup_epochs 3
```

### Important Notes

1. **Parameter Consistency**: When evaluating models, use the **same parameters** that were used during training.

   ```bash
   # Training with custom parameters
   uv run python -m src.my_project.main --mode train_phase --model_type phasenet_lstm \
          --filter_factor 2 --lstm_hidden_size 256

   # Evaluation with matching parameters
   uv run python -m src.my_project.main --mode eval_phase --model_type phasenet_lstm \
          --model_path path/to/model.pt --filter_factor 2 --lstm_hidden_size 256
   ```

2. **Filter Factor**: The `--filter_factor` parameter is available for all models and is the primary way to control overall model capacity.

3. **Default Values**: All parameters have sensible defaults, so only specify the ones you want to change.

4. **Warmup Scheduling**: All magnitude models now support learning rate warmup for improved training stability:
   - Learning rate starts at 10% of specified value and linearly increases to 100% over `warmup_epochs`
   - Helps prevent early overfitting and improves convergence, especially for transformer-based models
   - Particularly beneficial for complex models and small batch sizes

## Complete Command Reference

uv run python -m src.my_project.main --mode eval_phase --model_type phasenet --dataset ETHZ --model_path path/to/model.pt

# Evaluate a magnitude model with plots

uv run python -m src.my_project.main --mode eval_mag --model_type amag_v2 --dataset ETHZ --model_path path/to/model.pt --plot

````

#### Getting Help

```bash
# See all available options
uv run python -m src.my_project.main --help
````

> 📖 **For complete command reference and examples, see the [Command Reference](#complete-command-reference) section below.**

## Complete Command Reference

### Command Syntax

```bash
uv run python -m src.my_project.main [OPTIONS]
```

### Required Arguments

| Argument | Description   | Choices                                                                                                                                                                       | Default                            |
| -------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------- |
| `--mode` | Workflow mode | `train_phase`, `train_mag`, `eval_phase`, `eval_mag`, `tutorial_test_load_and_generator`, `tutorial_train_phasenet`, `tutorial_evaluate_phasenet`, `tutorial`, `plot_history` | `tutorial_test_load_and_generator` |

### Core Arguments

| Argument       | Description        | Choices                                                                                           | Default    | Notes                             |
| -------------- | ------------------ | ------------------------------------------------------------------------------------------------- | ---------- | --------------------------------- |
| `--dataset`    | Dataset to use     | `ETHZ`, `STEAD`, `GEOFON`, `MLAAPDE`                                                              | `ETHZ`     |                                   |
| `--model_type` | Model type         | `phasenet`, `phasenet_lstm`, `phasenet_conv_lstm`, `phasenet_mag`, `eqtransformer_mag`, `amag_v2` | `phasenet` |                                   |
| `--model_path` | Path to model file | Any valid path                                                                                    | `""`       | Required for evaluation modes     |
| `--epochs`     | Training epochs    | Positive integer                                                                                  | `5`        | For training modes only           |
| `--plot`       | Show/save plots    | Flag (no value)                                                                                   | `False`    | For `eval_mag` and `plot_history` |

### Training Parameters (Magnitude Models Only)

| Argument          | Description                                   | Type  | Default | Notes                                                                    |
| ----------------- | --------------------------------------------- | ----- | ------- | ------------------------------------------------------------------------ |
| `--learning_rate` | Learning rate for training                    | float | varies  | Model-specific defaults: EQTransformerMag=0.0001, others=0.001           |
| `--batch_size`    | Batch size for training                       | int   | varies  | Model-specific defaults: EQTransformerMag=16, PhaseNetMag=32, AMAG_v2=64 |
| `--warmup_epochs` | Number of warmup epochs with linear LR warmup | int   | 5       | Learning rate starts at 10% and linearly increases to 100%               |

**Examples:**

```bash
# Train with custom learning rate and batch size
uv run python -m src.my_project.main --mode train_mag --model_type phasenet_mag \
       --learning_rate 0.002 --batch_size 64 --warmup_epochs 10

# Train EQTransformerMag with optimized parameters
uv run python -m src.my_project.main --mode train_mag --model_type eqtransformer_mag \
       --learning_rate 0.0001 --batch_size 16 --warmup_epochs 5

# Train AMAG_v2 with stability warmup
uv run python -m src.my_project.main --mode train_mag --model_type amag_v2 \
       --learning_rate 0.001 --batch_size 64 --warmup_epochs 3
```

### Model Configuration Arguments

#### Common Parameters (All Models)

| Argument          | Description           | Type | Default | Notes                                    |
| ----------------- | --------------------- | ---- | ------- | ---------------------------------------- |
| `--filter_factor` | Model capacity factor | int  | 1       | Multiplies filter sizes throughout model |

#### PhaseNetLSTM Parameters

| Argument               | Description            | Type | Default | Notes                     |
| ---------------------- | ---------------------- | ---- | ------- | ------------------------- |
| `--lstm_hidden_size`   | LSTM hidden size       | int  | auto    | None for auto-calculation |
| `--lstm_num_layers`    | Number of LSTM layers  | int  | 1       |                           |
| `--lstm_bidirectional` | Use bidirectional LSTM | flag | True    | Include flag to enable    |

#### PhaseNetConvLSTM Parameters

| Argument            | Description          | Type | Default | Notes |
| ------------------- | -------------------- | ---- | ------- | ----- |
| `--convlstm_hidden` | ConvLSTM hidden size | int  | 64      |       |

#### PhaseNetMag Parameters

| Argument | Description          | Type | Default | Notes                  |
| -------- | -------------------- | ---- | ------- | ---------------------- |
| `--norm` | Normalization method | str  | "std"   | Choices: "std", "peak" |

#### EQTransformerMag Parameters

| Argument                           | Description                          | Type  | Default | Notes                    |
| ---------------------------------- | ------------------------------------ | ----- | ------- | ------------------------ |
| `--n_class`                        | Number of output classes             | int   | 1       | For magnitude regression |
| `--phases`                         | Phase types to use                   | str   | "PS"    |                          |
| `--sampling_rate`                  | Sampling rate in Hz                  | float | 100.0   |                          |
| `--cnn_blocks`                     | Number of CNN blocks                 | int   | 5       |                          |
| `--lstm_blocks`                    | Number of LSTM blocks                | int   | 2       |                          |
| `--transformer_d_model`            | Transformer model dimension          | int   | 128     |                          |
| `--transformer_nhead`              | Number of attention heads            | int   | 8       |                          |
| `--transformer_num_encoder_layers` | Number of transformer encoder layers | int   | 4       |                          |
| `--transformer_dim_feedforward`    | Transformer feedforward dimension    | int   | 512     |                          |

#### MagnitudeNet (AMAG v2) Parameters

| Argument        | Description           | Type  | Default | Notes |
| --------------- | --------------------- | ----- | ------- | ----- |
| `--lstm_hidden` | LSTM hidden size      | int   | 128     |       |
| `--lstm_layers` | Number of LSTM layers | int   | 2       |       |
| `--dropout`     | Dropout rate          | float | 0.2     |       |

### Mode-Specific Requirements

#### Training Modes (`train_phase`, `train_mag`)

- **Required**: `--model_type`, `--dataset`
- **Optional**: `--epochs` (default: 5)
- **Example**: `--mode train_phase --model_type phasenet --dataset ETHZ --epochs 10`

#### Evaluation Modes (`eval_phase`, `eval_mag`)

- **Required**: `--model_type`, `--dataset`, `--model_path`
- **Optional**: `--plot` (for `eval_mag` only)
- **Example**: `--mode eval_phase --model_type phasenet --dataset ETHZ --model_path path/to/model.pt`

#### Tutorial Modes

- **`tutorial_test_load_and_generator`**: Only requires `--dataset`
- **`tutorial_train_phasenet`**: Requires `--dataset`, optional `--epochs`
- **`tutorial_evaluate_phasenet`**: Requires `--dataset`, `--model_path`
- **`tutorial`** (deprecated): Requires `--model_path`

#### Utility Modes

- **`plot_history`**: Requires `--model_path`, optional `--plot`

### Available Model Types

- **Phase Models:**

  - `phasenet`: Standard PhaseNet for seismic phase detection
  - `phasenet_lstm`: PhaseNet with LSTM layers for enhanced temporal modeling
  - `phasenet_conv_lstm`: PhaseNet with Convolutional LSTM layers

- **Magnitude Models:**
  - `phasenet_mag`: PhaseNet adapted for magnitude regression
  - `eqtransformer_mag`: Transformer-based magnitude estimation with 30-second windows
  - `amag_v2`: Advanced Magnitude estimation model (MagnitudeNet)

### Available Datasets

- `ETHZ`: Swiss Seismological Service dataset
- `STEAD`: Stanford Earthquake Dataset
- `GEOFON`: GEOFON network dataset
- `MLAAPDE`: Machine Learning Applied to Analyze Passive seismic Data and Earthquakes

### Complete Command Examples

#### Phase Model Training

```bash
# Train PhaseNet for 50 epochs (standard model)
uv run python -m src.my_project.main --mode train_phase --model_type phasenet --dataset ETHZ --epochs 50

# Train PhaseNet-LSTM with default parameters
uv run python -m src.my_project.main --mode train_phase --model_type phasenet_lstm --dataset STEAD --epochs 30

# Train PhaseNet-LSTM with high capacity
uv run python -m src.my_project.main --mode train_phase --model_type phasenet_lstm \
       --dataset ETHZ --epochs 50 --filter_factor 2 --lstm_hidden_size 256 --lstm_num_layers 3

# Train PhaseNet-ConvLSTM with custom parameters
uv run python -m src.my_project.main --mode train_phase --model_type phasenet_conv_lstm \
       --dataset GEOFON --epochs 25 --filter_factor 2 --convlstm_hidden 128
```

#### Magnitude Model Training

```bash
# Train PhaseNetMag with default parameters and warmup
uv run python -m src.my_project.main --mode train_mag --model_type phasenet_mag --dataset ETHZ --epochs 40

# Train PhaseNetMag with custom training parameters and warmup
uv run python -m src.my_project.main --mode train_mag --model_type phasenet_mag \
       --dataset ETHZ --epochs 50 --filter_factor 2 --norm peak \
       --learning_rate 0.002 --batch_size 64 --warmup_epochs 10

# Train EQTransformerMag with default parameters (30-second windows)
uv run python -m src.my_project.main --mode train_mag --model_type eqtransformer_mag --dataset ETHZ --epochs 50

# Train EQTransformerMag with transformer-optimized parameters
uv run python -m src.my_project.main --mode train_mag --model_type eqtransformer_mag \
       --dataset ETHZ --epochs 50 --transformer_d_model 256 --transformer_nhead 16 \
       --transformer_num_encoder_layers 6 --learning_rate 0.0001 --batch_size 16 --warmup_epochs 8

# Train AMAG_v2 (MagnitudeNet) with default parameters
uv run python -m src.my_project.main --mode train_mag --model_type amag_v2 --dataset ETHZ --epochs 50

# Train AMAG_v2 with high capacity and stability warmup
uv run python -m src.my_project.main --mode train_mag --model_type amag_v2 \
       --dataset ETHZ --epochs 50 --filter_factor 2 --lstm_hidden 256 --lstm_layers 3 \
       --dropout 0.3 --learning_rate 0.001 --batch_size 64 --warmup_epochs 5
```

#### Phase Model Evaluation

```bash
# Evaluate PhaseNet model
uv run python -m src.my_project.main --mode eval_phase --model_type phasenet --dataset ETHZ \
       --model_path src/trained_weights/PhaseNet_ETHZ/model_final_*.pt

# Evaluate PhaseNet-LSTM model (use same parameters as training!)
uv run python -m src.my_project.main --mode eval_phase --model_type phasenet_lstm --dataset ETHZ \
       --model_path src/trained_weights/PhaseNetLSTM_*/model_final_*.pt \
       --filter_factor 2 --lstm_hidden_size 256 --lstm_num_layers 3

# Evaluate PhaseNet-ConvLSTM model
uv run python -m src.my_project.main --mode eval_phase --model_type phasenet_conv_lstm --dataset ETHZ \
       --model_path src/trained_weights/PhaseNetLSTM_*/model_final_*.pt \
       --filter_factor 2 --convlstm_hidden 128
```

#### Magnitude Model Evaluation

```bash
# Evaluate PhaseNetMag model (matching training parameters)
uv run python -m src.my_project.main --mode eval_mag --model_type phasenet_mag --dataset ETHZ \
       --model_path src/trained_weights/PhaseNetMag_ETHZ/model_final_*.pt \
       --filter_factor 2 --norm peak

# Evaluate EQTransformerMag model with plots (matching training parameters)
uv run python -m src.my_project.main --mode eval_mag --model_type eqtransformer_mag --dataset ETHZ \
       --model_path src/trained_weights/EQTransformerMag_*/model_final_*.pt --plot \
       --transformer_d_model 256 --transformer_nhead 16 --transformer_num_encoder_layers 6

# Evaluate AMAG_v2 model with plots (matching training parameters)
uv run python -m src.my_project.main --mode eval_mag --model_type amag_v2 --dataset ETHZ \
       --model_path trained_weights/magnitudenet_v1/model_final_*.pt --plot \
       --filter_factor 2 --lstm_hidden 256 --lstm_layers 3 --dropout 0.3
```

#### Tutorial Commands

```bash
# Test data loading and generators
uv run python -m src.my_project.main --mode tutorial_test_load_and_generator --dataset ETHZ

# Tutorial PhaseNet training (5 epochs)
uv run python -m src.my_project.main --mode tutorial_train_phasenet --dataset ETHZ --epochs 5

# Tutorial PhaseNet evaluation
uv run python -m src.my_project.main --mode tutorial_evaluate_phasenet --dataset ETHZ --model_path path/to/model.pt

# Legacy tutorial mode (deprecated)
uv run python -m src.my_project.main --mode tutorial --model_path path/to/model.pt
```

#### Utility Commands

```bash
# Plot training history (save only)
uv run python -m src.my_project.main --mode plot_history --model_path path/to/training_history_*.pt

# Plot training history (show and save)
uv run python -m src.my_project.main --mode plot_history --model_path path/to/training_history_*.pt --plot
```

### Help Command

```bash
# Display all available options
uv run python -m src.my_project.main --help
```

### Unified Architecture

The codebase has been refactored to provide a unified interface for all models:

#### Key Benefits

1. **Consistent Interface**: All models of the same type (phase/magnitude) use the same training and evaluation interface
2. **Simplified Commands**: Logical command structure with `--mode` defining the workflow and `--model_type` specifying the model
3. **Modular Design**: Easy to add new models by extending the unified functions
4. **Code Reuse**: Eliminates duplicate training/evaluation code across different models

#### Architecture Overview

- **Unified Training Functions**:
  - `train_phase_model()`: Handles PhaseNet and PhaseNet-LSTM training
  - `train_magnitude_model()`: Handles PhaseNetMag, EQTransformerMag, and AMAG_v2 training
- **Unified Evaluation Functions**:

  - `evaluate_phase_model_unified()`: Handles phase model evaluation
  - `evaluate_magnitude_model()`: Handles magnitude model evaluation

- **Model Factory Functions**: Automatically create and configure models based on `--model_type`

This design makes it easy to experiment with different models using the same workflow commands.

## Hyperparameter Tuning

The training script now supports automatic hyperparameter optimization using [Optuna](https://optuna.org/), a state-of-the-art hyperparameter optimization framework. This allows you to automatically find the best combination of hyperparameters for your magnitude estimation model.

#### Basic Hyperparameter Tuning

To run hyperparameter tuning with default settings:

```bash
uv run src/my_project/models/phasenet_mag/train.py --tune --dataset ETHZ
```

#### Advanced Hyperparameter Options

```bash
uv run src/my_project/models/phasenet_mag/train.py \
    --tune \
    --dataset ETHZ \
    --n_trials 50 \
    --max_epochs_per_trial 20 \
    --study_name "my_magnitude_study"
```

#### Hyperparameter Tuning Options

- `--tune`: Enable hyperparameter tuning mode
- `--n_trials`: Number of trials to run (default: 100)
- `--max_epochs_per_trial`: Maximum epochs per trial during tuning (default: 30)
- `--study_name`: Name for the Optuna study (default: auto-generated)

#### Regular Training Options (when `--tune` is not used)

- `--epochs`: Number of training epochs (default: 50)
- `--lr`: Learning rate (default: 1e-3)
- `--batch_size`: Batch size (default: 256)
- `--optimizer`: Optimizer type (Adam or AdamW, default: Adam)
- `--weight_decay`: Weight decay for optimizer (default: 1e-5)
- `--scheduler_patience`: Patience for learning rate scheduler (default: 5)
- `--filter_factor`: Filter factor for model architecture (default: 1)
- `--save_every`: Save model every N epochs (default: 5)

### Output Files

When running hyperparameter tuning, several files are created in `src/trained_weights/{model_name}/optuna_studies/`:

#### CSV Files

- `{study_name}_results.csv`: Detailed results for all trials including:
  - Trial number, validation loss, state, timing
  - All hyperparameter values
  - Intermediate validation losses for each epoch
- `{study_name}_best_params.csv`: Best hyperparameters found:
  - Optimal parameter values
  - Best validation loss achieved
  - Trial number of best result

#### Visualization

- `{study_name}_analysis.png`: Multi-panel plot showing:
  - Optimization history (validation loss vs trial number)
  - Parameter importance (simplified)
  - Learning rate distribution for top trials
  - Training progress for best trial

#### Study Object

- `{study_name}.pkl`: Pickled Optuna study object for advanced analysis
