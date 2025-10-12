# Machine Learning in Seismic Magnitude Estimation

### Description

### Installation

1. After cloning the repo, run the following depending on the platform

- If CPU based: `uv sync --extra cpu`
- If GPU based: `uv sync --extra gpu`

2. Verify CUDA installation:

   ```bash
   uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

3. If a previous version of custom seisbench persists, run `uv lock --upgrade` then `uv sync`

4. If the system still does not install the cuda version correctly, copy the contents of `pyproject-cuda.txt` into the pyproject and try again with `uv sync`.

### Basic Commands

#### Seisbench phasenet tutorial

Depending on the code uncommented, this will:

- test data downloading from the seisbench API
  - by default seisbench downloads the waveforms and metadata to `~/.seisbench`
- test generator initialisation from seisbench
- train a phasenet model on the provided dataset
- evaluates the trained model from a path on a given dataset

`uv run src/my_project/main.py --mode tutorial --model_path path/to/model.pt`

#### Custom manitude regression PhaseNet

**Train**
`uv run src/my_project/main.py --mode magnitude_train --dataset ETHZ --epochs 5`

**Evaluate**
`uv run src/my_project/main.py --mode magnitude_eval --dataset ETHZ --model_path path/to/model.pt`

**Plot model history**
This one is more general and I hope to use it for other models later down the line
`uv run src/my_project/main.py --mode plot_history --model_path path/to/training_history.pt`

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
