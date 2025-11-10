import os
import csv
import pandas as pd
import re
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


def plot_training_history(history_path: str, show_plot: bool = False):
    """
    Load and visualize training history from PhaseNetMag training.

    Args:
        history_path: Path to the training_history_*.pt file
        show_plot: Whether to display the plot (always saves PNG)
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

    # Always save plot as PNG
    import os

    base_name = os.path.splitext(os.path.basename(history_path))[0]
    plot_filename = f"{base_name}_plot.png"
    plot_path = os.path.join(os.path.dirname(history_path), plot_filename)

    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Plot saved to: {plot_path}")

    # Only show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()  # Close the figure to free memory

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


def plot_snr_distribution(data: BenchmarkDataset, output_dir: str = None) -> str:
    """
    Plot SNR distribution for a dataset and save as PNG.
    
    Args:
        data: BenchmarkDataset with trace_snr_db metadata
        output_dir: Directory to save the plot. If None, saves to current directory.
    
    Returns:
        Path to the saved PNG file
    """
    # Use helper to extract per-trace average SNR (handles 3-component arrays and string forms)
    snr_series = get_mean_snr_series(data)

    original_count = len(data)
    snr_values = snr_series.dropna()
    dropped_count = original_count - len(snr_values)

    if dropped_count > 0:
        print(f"\nWarning: Dropped {dropped_count} samples with invalid or missing SNR values")

    if len(snr_values) == 0:
        raise ValueError(f"No valid SNR values found in dataset {data.name}")
    
    dataset_name = data.name
    
    # Calculate statistics
    snr_min = snr_values.min()
    snr_max = snr_values.max()
    snr_mean = snr_values.mean()
    snr_median = snr_values.median()
    snr_std = snr_values.std()
    
    print(f"\nSNR Statistics for {dataset_name}:")
    print(f"  Samples: {len(snr_values):,}")
    print(f"  Min:     {snr_min:.2f} dB")
    print(f"  Max:     {snr_max:.2f} dB")
    print(f"  Mean:    {snr_mean:.2f} dB")
    print(f"  Median:  {snr_median:.2f} dB")
    print(f"  Std:     {snr_std:.2f} dB")
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Subplot 1: Histogram
    axes[0].hist(snr_values, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    axes[0].axvline(snr_mean, color='red', linestyle='--', linewidth=2,
                    label=f'Mean: {snr_mean:.2f} dB')
    axes[0].axvline(snr_median, color='green', linestyle='--', linewidth=2,
                    label=f'Median: {snr_median:.2f} dB')
    axes[0].set_xlabel('SNR (dB)', fontsize=12)
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title(f'{dataset_name} Dataset: SNR Distribution', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # Add statistics text box
    stats_text = f'Samples: {len(snr_values):,}\nMin: {snr_min:.2f} dB\nMax: {snr_max:.2f} dB\nStd: {snr_std:.2f} dB'
    axes[0].text(0.98, 0.97, stats_text, transform=axes[0].transAxes,
                fontsize=9, verticalalignment='top', horizontalalignment='right',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Subplot 2: Cumulative distribution
    sorted_snr = np.sort(snr_values)
    cumulative = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr) * 100
    axes[1].plot(sorted_snr, cumulative, linewidth=2, color='steelblue')
    
    # Mark common thresholds
    thresholds = [5, 10, 15, 20]
    for threshold in thresholds:
        # Find percentage of data above threshold
        pct_above = (snr_values >= threshold).sum() / len(snr_values) * 100
        idx = np.searchsorted(sorted_snr, threshold)
        if idx < len(cumulative):
            pct_at = cumulative[idx]
            axes[1].axvline(threshold, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            axes[1].text(threshold, pct_at - 5, f'{threshold} dB\n{pct_above:.1f}% ≥',
                        ha='left', va='top', fontsize=8,
                        bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    axes[1].set_xlabel('SNR Threshold (dB)', fontsize=12)
    axes[1].set_ylabel('Cumulative Percentage (%)', fontsize=12)
    axes[1].set_title('Data Retention vs SNR Threshold', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim(left=max(0, snr_min - 5))
    axes[1].set_ylim([0, 100])
    
    plt.tight_layout()
    
    # Determine output path
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    output_path = os.path.join(output_dir, f'{dataset_name}_snr_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    plt.close()
    
    return output_path


def get_mean_snr_series(data: BenchmarkDataset):
    """Return a pandas Series of per-trace mean SNR (dB).

    This function handles several formats for `trace_snr_db` metadata entries:
    - numpy arrays or lists of 3 component SNRs (Z,N,E) -> mean of finite values
    - scalar numeric values -> returned as-is
    - string representations like "[18.3 19. 14.5]" -> parsed and averaged
    - NaN/None -> returned as NaN

    The returned Series is aligned with `data.metadata` index.
    """
    if "trace_snr_db" not in data.metadata.columns:
        raise ValueError(f"Dataset {data.name} does not have 'trace_snr_db' metadata column")

    raw = data.metadata["trace_snr_db"]

    def _unwrap_mean(val):
        # pandas NA
        try:
            if pd.isna(val):
                return np.nan
        except Exception:
            pass

        # numpy array / list-like
        if isinstance(val, (list, tuple, np.ndarray)):
            try:
                arr = np.array(val, dtype=float)
                if arr.size == 0:
                    return np.nan
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return np.nan
                return float(np.nanmean(arr))
            except Exception:
                return np.nan

        # scalar numeric
        if isinstance(val, (int, float, np.number)):
            if np.isfinite(val):
                return float(val)
            return np.nan

        # string representation - try to find numbers
        s = str(val)
        nums = re.findall(r"[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?", s)
        if nums:
            try:
                arr = np.array([float(x) for x in nums], dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    return np.nan
                return float(np.nanmean(arr))
            except Exception:
                return np.nan

        return np.nan

    # Apply the unwrap function to each entry and return a pandas Series
    import pandas as pd
    series = pd.Series([_unwrap_mean(v) for v in raw], index=raw.index)
    return series


def plot_samples(generator: GenericGenerator, single: bool = False):
    try:
        while True:
            # Get random sample
            random_index = np.random.randint(len(generator))
            sample = generator[random_index]

            print(f"Sample index: {random_index}")

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
