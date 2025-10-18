import torch
import os
from datetime import datetime

import seisbench.models as sbm
import seisbench.data as sbd
from seisbench.data import BenchmarkDataset

from my_project.loaders import data_loader as dl
from my_project.models.phasenetLSTM.model import PhaseNetLSTM
from my_project.models.phasenetLSTM.modelv2 import PhaseNetConvLSTM


def train_phasenetLSTM(
    model_name: str,
    model: sbm.WaveformModel,
    data: BenchmarkDataset,
    learning_rate=1e-2,
    epochs=5,
    batch_size: int = 256,
    num_workers: int = 4,
):
    train_generator, train_loader, _ = dl.load_dataset(
        data, model, "train", batch_size=batch_size, num_workers=num_workers
    )
    dev_generator, dev_loader, _ = dl.load_dataset(
        data, model, "dev", batch_size=batch_size, num_workers=num_workers
    )

    print("Data successfully loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss_fn(y_pred, y_true, eps=1e-5):
        # vector cross entropy loss
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(
            -1
        )  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h

    def train_loop(dataloader):
        size = len(dataloader.dataset)
        for batch_id, batch in enumerate(dataloader):
            # Compute prediction and loss
            x = batch["X"].to(model.device)
            x_preproc = model.annotate_batch_pre(
                x, {}
            )  # Remove mean and normalize amplitude
            pred = model(x_preproc)
            loss = loss_fn(pred, batch["y"].float().to(model.device))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 5 == 0:
                loss, current = loss.item(), batch_id * batch["X"].shape[0]
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader):
        num_batches = len(dataloader)
        test_loss = 0

        model.eval()  # close the model for evaluation

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(model.device)
                x_preproc = model.annotate_batch_pre(
                    x, {}
                )  # Remove mean and normalize amplitude
                pred = model(x_preproc)
                test_loss += loss_fn(pred, batch["y"].float().to(model.device)).item()

        model.train()  # re-open model for training stage

        test_loss /= num_batches
        print(f"Test avg loss: {test_loss:>8f} \n")

    # Train model with checkpoint each 5 epochs
    save_dir = f"src/trained_weights/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader)
        test_loop(dev_loader)
        if (t + 1) % 5 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"model_epoch_{t+1}_{timestamp}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_final_{timestamp}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete, model saved to {save_path}")


def train_phasenet_lstm_model(
    data: sbd.BenchmarkDataset,
    epochs: int = 50,
    filter_factor: int = 1,
    lstm_hidden_size: int = None,
    lstm_num_layers: int = 1,
    lstm_bidirectional: bool = True,
    learning_rate: float = 1e-3,
    batch_size: int = 32,
    num_workers: int = 4,
):
    """
    Train PhaseNet-LSTM for seismic phase picking.

    Args:
        data: SeisBench dataset (e.g., STEAD, INSTANCE, etc.)
        epochs: Number of training epochs
        filter_factor: Multiplier for number of filters in each layer
        lstm_hidden_size: LSTM hidden dimension (None = use bottleneck filters)
        lstm_num_layers: Number of LSTM layers at bottleneck
        lstm_bidirectional: Whether to use bidirectional LSTM
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size for training
        num_workers: Number of workers for data loading

    Returns:
        Trained model and best validation loss
    """
    print("=" * 70)
    print(f"Training PhaseNet-LSTM on {data.name} dataset")
    print("=" * 70)
    print(f"Configuration:")
    print(f"  Epochs: {epochs}")
    print(f"  Filter factor: {filter_factor}")
    print(f"  LSTM hidden size: {lstm_hidden_size if lstm_hidden_size else 'auto'}")
    print(f"  LSTM layers: {lstm_num_layers}")
    print(f"  LSTM bidirectional: {lstm_bidirectional}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Batch size: {batch_size}")
    print("=" * 70)
    print()

    # Create model
    # model = PhaseNetLSTM(
    #     filter_factor=filter_factor,
    #     lstm_hidden_size=lstm_hidden_size,
    #     lstm_num_layers=lstm_num_layers,
    #     lstm_bidirectional=lstm_bidirectional,
    # )
    model = PhaseNetConvLSTM()

    # Move to preferred device (GPU if available)
    if hasattr(model, "to_preferred_device"):
        model.to_preferred_device(verbose=True)
    else:
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        print(f"Model moved to device: {device}")

    # Generate model name for saving
    lstm_config = (
        f"h{lstm_hidden_size if lstm_hidden_size else 'auto'}_l{lstm_num_layers}"
    )
    if lstm_bidirectional:
        lstm_config += "_bi"

    model_name = f"PhaseNetLSTM_{data.name}_f{filter_factor}_{lstm_config}"

    print(f"\nModel name: {model_name}\n")

    # Train model
    best_loss = train_phasenetLSTM(
        model_name=model_name,
        model=model,
        data=data,
        learning_rate=learning_rate,
        epochs=epochs,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return model, best_loss


def train_phasenet_lstm_default(data: sbd.BenchmarkDataset, epochs: int = 5):
    """
    Train PhaseNet-LSTM with default configuration.

    This is a simple wrapper for quick training with sensible defaults.

    Args:
        data: SeisBench dataset
        epochs: Number of training epochs

    Returns:
        Trained model and best validation loss
    """
    return train_phasenet_lstm_model(
        data=data,
        epochs=epochs,
        filter_factor=1,
        lstm_hidden_size=None,  # Auto: uses bottleneck filters (128)
        lstm_num_layers=1,
        lstm_bidirectional=True,
        learning_rate=1e-3,
        batch_size=256,
        num_workers=4,
    )
