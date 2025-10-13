import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from seisbench.data import BenchmarkDataset
from seisbench.generate import GenericGenerator


def dump_metadata_to_csv(sbd: BenchmarkDataset, filename="metadata.csv"):
    """
    Dumps metadata values from the sbd object to a CSV file in /metadata/.
    Args:
        sbd: The seisbench dataset object with metadata attribute (assumed to be a DataFrame or dict-like).
        filename: Name of the CSV file to write (default: metadata.csv).
    """
    metadata_dir = os.path.join(os.path.dirname(__file__), "../metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    csv_path = os.path.join(metadata_dir, filename)

    metadata = getattr(sbd, "metadata", None)
    if metadata is None:
        raise ValueError("sbd object does not have a 'metadata' attribute.")

    # If pandas DataFrame
    try:
        import pandas as pd

        if isinstance(metadata, pd.DataFrame):
            metadata.to_csv(csv_path, index=False)
            return csv_path
    except ImportError:
        pass

    # If dict-like
    if isinstance(metadata, dict):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["key", "value"])
            for k, v in metadata.items():
                writer.writerow([k, v])
        return csv_path

    # If list of dicts
    if isinstance(metadata, list) and all(isinstance(item, dict) for item in metadata):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
        return csv_path

    raise TypeError(
        "Unsupported metadata format. Must be pandas DataFrame, dict, or list of dicts."
    )


def plot_magnitude_distribution(data: BenchmarkDataset) -> None:
    magnitudes = data.metadata["source_magnitude"]

    # Plot histogram: frequency vs magnitude bins (0-9, step 0.5)
    bins = [x * 0.5 for x in range(19)]  # 0, 0.5, ..., 10
    plt.figure()
    plt.hist(magnitudes, bins=bins, edgecolor="black")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.title("Magnitude Distribution in DummyDataset")
    plt.xticks(bins)
    plt.show()


def plot_training_history(history_path: str):
    """
    Load and visualize training history from PhaseNetMag training.

    Args:
        history_path: Path to the training_history_*.pt file
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"Loading training history from: {history_path}")

    # Load the history dictionary
    history = torch.load(history_path, map_location="cpu")

    # Extract data
    train_losses = history["train_losses"]
    val_losses = history["val_losses"]
    best_val_loss = history["best_val_loss"]

    print(f"Training epochs: {len(train_losses)}")
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")

    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(train_losses) + 1)

    # Plot 1: Loss curves
    ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax1.axhline(
        y=best_val_loss,
        color="g",
        linestyle="--",
        label=f"Best Val Loss: {best_val_loss:.4f}",
    )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss (MSE)")
    ax1.set_title("Training and Validation Loss")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Learning progress (log scale)
    ax2.semilogy(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
    ax2.semilogy(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
    ax2.axhline(
        y=best_val_loss,
        color="g",
        linestyle="--",
        label=f"Best Val Loss: {best_val_loss:.4f}",
    )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss (MSE) - Log Scale")
    ax2.set_title("Training Progress (Log Scale)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # Print training analysis
    print("\n" + "=" * 50)
    print("TRAINING ANALYSIS")
    print("=" * 50)

    # Check for overfitting
    if len(val_losses) > 1:
        val_trend = np.diff(val_losses[-3:])  # Last 3 epochs trend
        if np.mean(val_trend) > 0:
            print("Validation loss increasing - possible overfitting")
        else:
            print("Validation loss stable/decreasing - good training")

    # Check convergence
    if len(train_losses) > 2:
        train_change = abs(train_losses[-1] - train_losses[-2])
        val_change = abs(val_losses[-1] - val_losses[-2])

        if train_change < 0.001 and val_change < 0.001:
            print("Training appears to have converged")
        else:
            print("Training still progressing - could benefit from more epochs")

    # Loss gap analysis
    final_gap = val_losses[-1] - train_losses[-1]
    print(f"Train-Val gap: {final_gap:.4f}")
    if final_gap > 0.1:
        print("Large train-val gap suggests overfitting")
    elif final_gap < 0.05:
        print("Small train-val gap indicates good generalization")

    return history


def plot_samples(generator: GenericGenerator, single: bool = False):
    try:
        while True:
            # Get random sample
            random_index = np.random.randint(len(generator))
            sample = generator[random_index]

            print(f"\nSample index: {random_index}")
            print(sample)

            fig = plt.figure(figsize=(15, 12))
            axs = fig.subplots(
                3,
                1,
                sharex=True,
                gridspec_kw={"hspace": 0.2, "height_ratios": [3, 1, 1]},
            )

            # Plot Z, N, E waveforms
            channel_names = ["Z", "N", "E"]
            for i in range(sample["X"].shape[0]):
                axs[0].plot(sample["X"][i], label=channel_names[i])
            axs[0].set_ylabel("Waveform")
            axs[0].legend()

            # Plot P, S, Noise phase labels with distinctive styles
            phase_names = ["P", "S", "Noise"]
            colors = ["tab:blue", "tab:green", "tab:orange"]
            linestyles = ["-", "--", ":"]  # solid, dashed, dotted
            linewidths = [2.5, 2.5, 2.5]
            alphas = [0.8, 0.8, 0.8]

            for i in range(sample["y"].shape[0]):
                axs[1].plot(
                    sample["y"][i],
                    label=phase_names[i],
                    color=colors[i],
                    linestyle=linestyles[i],
                    linewidth=linewidths[i],
                    alpha=alphas[i],
                )
            axs[1].set_ylabel("Phase Label")
            axs[1].legend()

            # Plot magnitude label
            if "magnitude" in sample:
                mag_data = sample["magnitude"]
                axs[2].plot(mag_data, color="tab:orange")
                axs[2].set_ylabel("Magnitude Label")
            else:
                axs[2].text(0.5, 0.5, "No magnitude label", ha="center", va="center")
            axs[2].set_xlabel("Sample Index")

            plt.show()

            if single:
                return

    except KeyboardInterrupt:
        print("\nStopped by user")
