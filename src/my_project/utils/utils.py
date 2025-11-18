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
    plt.yscale("log")
    # plt.title("Magnitude Distribution in DummyDataset")
    plt.xticks(range(0, 10))  # Show only whole numbers 0-9
    plt.show()



def plot_training_history(history_path: str, show_plot: bool = False, detailed_metrics: bool = False):
    """
    Load and visualize training history from model training.
    
    Supports two formats:
    1. Simple format (default): Used by PhaseNet, EQTransformer, UMamba V1/V2
       - Keys: train_losses, val_losses, best_val_loss, train_maes, val_maes, learning_rates
    2. Detailed format: Used by UMamba V3 with multi-head architecture
       - Keys: train_loss, val_loss, train_loss_scalar, train_loss_temporal, 
               val_loss_scalar, val_loss_temporal, learning_rates
               Optional: train_uncertainty, val_uncertainty (if using uncertainty head)

    Args:
        history_path: Path to the training_history*.pt file
        show_plot: Whether to display the plot (always saves PNG)
        detailed_metrics: If True, plots detailed scalar/temporal/uncertainty metrics.
                         Set to False by default for backward compatibility.
    """
    import torch
    import matplotlib.pyplot as plt
    import numpy as np

    print(f"Loading training history from: {history_path}")

    # Load the history dictionary
    history = torch.load(history_path, map_location="cpu")

    # Detect format based on keys
    has_detailed = "train_loss_scalar" in history
    
    if detailed_metrics and not has_detailed:
        print("Warning: detailed_metrics=True but history doesn't contain detailed metrics.")
        print("Falling back to simple plotting mode.")
        detailed_metrics = False
    
    if has_detailed and not detailed_metrics:
        print("Note: Detailed metrics available but detailed_metrics=False. Using simple view.")
    
    # Extract data based on format
    if detailed_metrics and has_detailed:
        # UMamba V3 detailed format
        train_losses = history["train_loss"]
        val_losses = history["val_loss"]
        train_loss_scalar = history["train_loss_scalar"]
        train_loss_temporal = history["train_loss_temporal"]
        val_loss_scalar = history["val_loss_scalar"]
        val_loss_temporal = history["val_loss_temporal"]
        learning_rates = history.get("learning_rates", [])
        
        # Check for uncertainty metrics
        has_uncertainty = "train_uncertainty" in history
        if has_uncertainty:
            train_uncertainty = history["train_uncertainty"]
            val_uncertainty = history["val_uncertainty"]
        
        best_val_loss = min(val_losses)
        
        print(f"Training epochs: {len(train_losses)}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        print(f"Final scalar loss (train/val): {train_loss_scalar[-1]:.6f} / {val_loss_scalar[-1]:.6f}")
        print(f"Final temporal loss (train/val): {train_loss_temporal[-1]:.6f} / {val_loss_temporal[-1]:.6f}")
        if has_uncertainty:
            print(f"Final uncertainty log_var (train/val): {train_uncertainty[-1]:.3f} / {val_uncertainty[-1]:.3f}")
        
        # Create detailed plots
        if has_uncertainty:
            fig = plt.figure(figsize=(20, 10))
            gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
            ax1 = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            ax3 = fig.add_subplot(gs[0, 2])
            ax4 = fig.add_subplot(gs[1, 0])
            ax5 = fig.add_subplot(gs[1, 1])
            ax6 = fig.add_subplot(gs[1, 2])
        else:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            ax1, ax2, ax3, ax4 = axes.flatten()
        
        epochs = range(1, len(train_losses) + 1)
        
        # Plot 1: Total Loss
        ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        ax1.axhline(y=best_val_loss, color="g", linestyle="--", 
                   label=f"Best Val: {best_val_loss:.4f}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Total Loss")
        ax1.set_title("Combined Loss (Scalar + Temporal)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scalar Loss
        ax2.plot(epochs, train_loss_scalar, "b-", label="Train Scalar", linewidth=2)
        ax2.plot(epochs, val_loss_scalar, "r-", label="Val Scalar", linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Scalar Loss (MSE)")
        ax2.set_title("Scalar Head Loss (Global Magnitude)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temporal Loss
        ax3.plot(epochs, train_loss_temporal, "b-", label="Train Temporal", linewidth=2)
        ax3.plot(epochs, val_loss_temporal, "r-", label="Val Temporal", linewidth=2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("Temporal Loss (MSE)")
        ax3.set_title("Temporal Head Loss (Per-timestep)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: RMSE Comparison
        train_rmse_scalar = np.sqrt(train_loss_scalar)
        val_rmse_scalar = np.sqrt(val_loss_scalar)
        ax4.plot(epochs, train_rmse_scalar, "b-", label="Train RMSE", linewidth=2)
        ax4.plot(epochs, val_rmse_scalar, "r-", label="Val RMSE", linewidth=2)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("RMSE (Magnitude Units)")
        ax4.set_title("Scalar Head RMSE")
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Learning Rate
        if learning_rates:
            ax5.plot(epochs, learning_rates, "purple", linewidth=2)
            ax5.set_xlabel("Epoch")
            ax5.set_ylabel("Learning Rate")
            ax5.set_title("Learning Rate Schedule")
            ax5.set_yscale("log")
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No LR data", ha="center", va="center", 
                    transform=ax5.transAxes)
            ax5.axis("off")
        
        # Plot 6: Uncertainty (if available) or Loss Components
        if has_uncertainty:
            ax6.plot(epochs, train_uncertainty, "b-", label="Train log(σ²)", linewidth=2)
            ax6.plot(epochs, val_uncertainty, "r-", label="Val log(σ²)", linewidth=2)
            ax6.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
            ax6.set_xlabel("Epoch")
            ax6.set_ylabel("Log Variance")
            ax6.set_title("Uncertainty Head (Learned Sample Weighting)")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        else:
            # Show loss components stacked
            ax6.plot(epochs, train_loss_scalar, "b-", label="Scalar", linewidth=2, alpha=0.7)
            ax6.plot(epochs, train_loss_temporal, "g-", label="Temporal", linewidth=2, alpha=0.7)
            ax6.set_xlabel("Epoch")
            ax6.set_ylabel("Loss Components")
            ax6.set_title("Training Loss Breakdown")
            ax6.legend()
            ax6.grid(True, alpha=0.3)
        
    else:
        # Simple format (PhaseNet, EQTransformer, UMamba V1/V2)
        train_losses = history.get("train_losses", history.get("train_loss", []))
        val_losses = history.get("val_losses", history.get("val_loss", []))
        best_val_loss = history.get("best_val_loss", min(val_losses))
        
        print(f"Training epochs: {len(train_losses)}")
        print(f"Best validation loss: {best_val_loss:.6f}")
        print(f"Final training loss: {train_losses[-1]:.6f}")
        print(f"Final validation loss: {val_losses[-1]:.6f}")
        
        # Create simple plots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        epochs = range(1, len(train_losses) + 1)
        
        # Plot 1: Loss curves
        ax1.plot(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        ax1.axhline(y=best_val_loss, color="g", linestyle="--",
                   label=f"Best Val Loss: {best_val_loss:.4f}")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss (MSE)")
        ax1.set_title("Training and Validation Loss")
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Learning progress (log scale)
        ax2.semilogy(epochs, train_losses, "b-", label="Training Loss", linewidth=2)
        ax2.semilogy(epochs, val_losses, "r-", label="Validation Loss", linewidth=2)
        ax2.axhline(y=best_val_loss, color="g", linestyle="--",
                   label=f"Best Val Loss: {best_val_loss:.4f}")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (MSE) - Log Scale")
        ax2.set_title("Training Progress (Log Scale)")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Always save plot as PNG
    import os

    base_name = os.path.splitext(os.path.basename(history_path))[0]
    suffix = "_detailed" if detailed_metrics and has_detailed else ""
    plot_filename = f"{base_name}_plot{suffix}.png"
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
            print("⚠ Validation loss increasing - possible overfitting")
        else:
            print("✓ Validation loss stable/decreasing - good training")

    # Check convergence
    if len(train_losses) > 2:
        train_change = abs(train_losses[-1] - train_losses[-2])
        val_change = abs(val_losses[-1] - val_losses[-2])

        if train_change < 0.001 and val_change < 0.001:
            print("✓ Training appears to have converged")
        else:
            print("→ Training still progressing - could benefit from more epochs")

    # Loss gap analysis
    final_gap = val_losses[-1] - train_losses[-1]
    print(f"Train-Val gap: {final_gap:.4f}")
    if final_gap > 0.1:
        print("⚠ Large train-val gap suggests overfitting")
    elif final_gap < 0.05:
        print("✓ Small train-val gap indicates good generalization")
    
    # Detailed metrics analysis
    if detailed_metrics and has_detailed:
        print("\n" + "=" * 50)
        print("DETAILED METRICS ANALYSIS")
        print("=" * 50)
        
        # Scalar vs Temporal performance
        scalar_improvement = (val_loss_scalar[0] - val_loss_scalar[-1]) / val_loss_scalar[0] * 100
        temporal_improvement = (val_loss_temporal[0] - val_loss_temporal[-1]) / val_loss_temporal[0] * 100
        
        print(f"Scalar loss improvement: {scalar_improvement:.1f}%")
        print(f"Temporal loss improvement: {temporal_improvement:.1f}%")
        
        # Check which head is learning better
        if abs(scalar_improvement) > abs(temporal_improvement) * 1.2:
            print("→ Scalar head learning faster than temporal head")
        elif abs(temporal_improvement) > abs(scalar_improvement) * 1.2:
            print("→ Temporal head learning faster than scalar head")
        else:
            print("✓ Balanced learning between both heads")
        
        # Uncertainty analysis
        if has_uncertainty:
            final_uncertainty = val_uncertainty[-1]
            uncertainty_change = val_uncertainty[-1] - val_uncertainty[0]
            print(f"\nUncertainty log_var final: {final_uncertainty:.3f}")
            print(f"Uncertainty change: {uncertainty_change:+.3f}")
            
            # Interpret uncertainty
            final_sigma = np.sqrt(np.exp(final_uncertainty))
            print(f"Implied prediction std: ±{final_sigma:.3f} magnitude units")
            
            if uncertainty_change < -1.0:
                print("✓ Model confidence increased significantly")
            elif uncertainty_change > 1.0:
                print("⚠ Model confidence decreased - may need tuning")
            else:
                print("→ Model uncertainty stable")

    return history


def plot_snr_distribution(data: BenchmarkDataset, output_dir: str = None) -> tuple:
    """
    Plot SNR distribution for a dataset and save as two separate PNG files.
    
    Args:
        data: BenchmarkDataset with trace_snr_db metadata
        output_dir: Directory to save the plots. If None, saves to current directory.
    
    Returns:
        Tuple of (histogram_path, cumulative_path) - paths to the saved PNG files
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
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Histogram
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(snr_values, bins=100, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(snr_mean, color='red', linestyle='--', linewidth=2,
                label=f'Mean: {snr_mean:.2f} dB')
    ax1.axvline(snr_median, color='green', linestyle='--', linewidth=2,
                label=f'Median: {snr_median:.2f} dB')
    ax1.set_xlabel('SNR (dB)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path1 = os.path.join(output_dir, f'{dataset_name}_snr_histogram.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"\nHistogram saved to: {output_path1}")
    plt.close()
    
    # Plot 2: Cumulative distribution
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    sorted_snr = np.sort(snr_values)
    cumulative = np.arange(1, len(sorted_snr) + 1) / len(sorted_snr) * 100
    ax2.plot(sorted_snr, cumulative, linewidth=2, color='steelblue')
    
    # Mark common thresholds
    thresholds = [5, 10, 15, 20]
    for threshold in thresholds:
        # Find percentage of data above threshold
        pct_above = (snr_values >= threshold).sum() / len(snr_values) * 100
        idx = np.searchsorted(sorted_snr, threshold)
        if idx < len(cumulative):
            pct_at = cumulative[idx]
            ax2.axvline(threshold, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
            ax2.text(threshold, pct_at - 5, f'{threshold} dB\n{pct_above:.1f}% ≥',
                    ha='left', va='top', fontsize=8,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    ax2.set_xlabel('SNR Threshold (dB)', fontsize=12)
    ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(left=max(0, snr_min - 5))
    ax2.set_ylim([0, 100])
    
    # Add statistics text box
    stats_text = f'Samples: {len(snr_values):,}\nMin: {snr_min:.2f} dB\nMax: {snr_max:.2f} dB\nStd: {snr_std:.2f} dB'
    ax2.text(0.98, 0.97, stats_text, transform=ax2.transAxes,
            fontsize=9, verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, f'{dataset_name}_snr_cumulative.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Cumulative plot saved to: {output_path2}")
    plt.close()
    
    return output_path1, output_path2


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
