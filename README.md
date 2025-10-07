# Machine Learning in Seismic Magnitude Estimation

### Description

### Installation

1. After cloning the repo, run `uv sync` to install the required dependencies
   
   This will automatically install PyTorch with CUDA support (version 2.5.1+cu121) if available on your system.

2. Verify CUDA installation (optional):
   ```bash
   uv run python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

3. If a previous version of custom seisbench persists, run `uv lock --upgrade` then `uv sync`

### Commands

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
