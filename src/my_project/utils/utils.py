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

    # Plot histogram: frequency vs magnitude bins (-1 to 8, 26 bins total)
    bins = np.linspace(-1, 8, 27)  # 27 edges = 26 bins
    plt.figure(figsize=(10, 6))
    plt.hist(magnitudes, bins=bins, edgecolor='black', alpha=0.7, color='steelblue')
    plt.xlabel("Magnitude", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.yscale("log")
    # plt.title("Magnitude Distribution in DummyDataset")
    plt.xticks(range(-1, 9))  # Show whole numbers from -1 to 8
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
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
        ax1.plot(epochs, train_losses, color='steelblue', label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, color='coral', label="Validation Loss", linewidth=2)
        ax1.axhline(y=best_val_loss, color="red", linestyle="--", linewidth=1.5, alpha=0.5,
                   label=f"Best Val: {best_val_loss:.4f}")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Total Loss", fontsize=12)
        ax1.set_title("Combined Loss (Scalar + Temporal)")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Scalar Loss
        ax2.plot(epochs, train_loss_scalar, color='steelblue', label="Train Scalar", linewidth=2)
        ax2.plot(epochs, val_loss_scalar, color='coral', label="Val Scalar", linewidth=2)
        ax2.set_xlabel("Epoch", fontsize=12)
        ax2.set_ylabel("Scalar Loss (MSE)", fontsize=12)
        ax2.set_title("Scalar Head Loss (Global Magnitude)")
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Temporal Loss
        ax3.plot(epochs, train_loss_temporal, color='steelblue', label="Train Temporal", linewidth=2)
        ax3.plot(epochs, val_loss_temporal, color='coral', label="Val Temporal", linewidth=2)
        ax3.set_xlabel("Epoch", fontsize=12)
        ax3.set_ylabel("Temporal Loss (MSE)", fontsize=12)
        ax3.set_title("Temporal Head Loss (Per-timestep)")
        ax3.legend(fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: RMSE Comparison
        train_rmse_scalar = np.sqrt(train_loss_scalar)
        val_rmse_scalar = np.sqrt(val_loss_scalar)
        ax4.plot(epochs, train_rmse_scalar, color='steelblue', label="Train RMSE", linewidth=2)
        ax4.plot(epochs, val_rmse_scalar, color='coral', label="Val RMSE", linewidth=2)
        ax4.set_xlabel("Epoch", fontsize=12)
        ax4.set_ylabel("RMSE (Magnitude Units)", fontsize=12)
        ax4.set_title("Scalar Head RMSE")
        ax4.legend(fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        # Plot 5: Learning Rate
        if learning_rates:
            ax5.plot(epochs, learning_rates, color='steelblue', linewidth=2)
            ax5.set_xlabel("Epoch", fontsize=12)
            ax5.set_ylabel("Learning Rate", fontsize=12)
            ax5.set_title("Learning Rate Schedule")
            ax5.set_yscale("log")
            ax5.grid(True, alpha=0.3)
        else:
            ax5.text(0.5, 0.5, "No LR data", ha="center", va="center", 
                    transform=ax5.transAxes)
            ax5.axis("off")
        
        # Plot 6: Uncertainty (if available) or Loss Components
        if has_uncertainty:
            ax6.plot(epochs, train_uncertainty, color='steelblue', label="Train log(σ²)", linewidth=2)
            ax6.plot(epochs, val_uncertainty, color='coral', label="Val log(σ²)", linewidth=2)
            ax6.axhline(y=0, color="gray", linestyle=":", alpha=0.5)
            ax6.set_xlabel("Epoch", fontsize=12)
            ax6.set_ylabel("Log Variance", fontsize=12)
            ax6.set_title("Uncertainty Head (Learned Sample Weighting)")
            ax6.legend(fontsize=10)
            ax6.grid(True, alpha=0.3)
        else:
            # Show loss components stacked
            ax6.plot(epochs, train_loss_scalar, color='steelblue', label="Scalar", linewidth=2, alpha=0.7)
            ax6.plot(epochs, train_loss_temporal, color='coral', label="Temporal", linewidth=2, alpha=0.7)
            ax6.set_xlabel("Epoch", fontsize=12)
            ax6.set_ylabel("Loss Components", fontsize=12)
            ax6.set_title("Training Loss Breakdown")
            ax6.legend(fontsize=10)
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
        
        # Create simple plot
        fig, ax1 = plt.subplots(1, 1, figsize=(10, 6))
        epochs = range(1, len(train_losses) + 1)
        
        # Plot: Loss curves
        ax1.plot(epochs, train_losses, color='steelblue', label="Training Loss", linewidth=2)
        ax1.plot(epochs, val_losses, color='coral', label="Validation Loss", linewidth=2)
        ax1.axhline(y=best_val_loss, color="red", linestyle="--", linewidth=1.5, alpha=0.5,
                   label=f"Best Val Loss: {best_val_loss:.4f}")
        ax1.set_xlabel("Epoch", fontsize=12)
        ax1.set_ylabel("Loss (MSE)", fontsize=12)
        ax1.set_title("Training and Validation Loss")
        ax1.legend(fontsize=10)
        ax1.grid(True, alpha=0.3)

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


def plot_distance_depth_distribution(data: BenchmarkDataset, output_dir: str = None) -> tuple:
    """
    Plot histograms of source distance and depth distributions as two separate PNG files.
    
    Args:
        data: BenchmarkDataset with source_distance_km and source_depth_km metadata
        output_dir: Directory to save the plots. If None, saves to current directory.
    
    Returns:
        Tuple of (distance_path, depth_path) - paths to the saved PNG files
    """
    # Check for required metadata columns
    if "source_distance_km" not in data.metadata.columns:
        raise ValueError(f"Dataset {data.name} does not have 'source_distance_km' metadata column")
    if "source_depth_km" not in data.metadata.columns:
        raise ValueError(f"Dataset {data.name} does not have 'source_depth_km' metadata column")
    
    # Extract data and remove NaN values
    distances = data.metadata["source_distance_km"].dropna()
    depths = data.metadata["source_depth_km"].dropna()
    
    dataset_name = data.name
    
    # Calculate statistics for distance
    dist_min = distances.min()
    dist_max = distances.max()
    dist_mean = distances.mean()
    dist_median = distances.median()
    dist_std = distances.std()
    
    # Calculate statistics for depth
    depth_min = depths.min()
    depth_max = depths.max()
    depth_mean = depths.mean()
    depth_median = depths.median()
    depth_std = depths.std()
    
    print(f"\nDistance Statistics for {dataset_name}:")
    print(f"  Samples: {len(distances):,}")
    print(f"  Min:     {dist_min:.2f} km")
    print(f"  Max:     {dist_max:.2f} km")
    print(f"  Mean:    {dist_mean:.2f} km")
    print(f"  Median:  {dist_median:.2f} km")
    print(f"  Std:     {dist_std:.2f} km")
    
    print(f"\nDepth Statistics for {dataset_name}:")
    print(f"  Samples: {len(depths):,}")
    print(f"  Min:     {depth_min:.2f} km")
    print(f"  Max:     {depth_max:.2f} km")
    print(f"  Mean:    {depth_mean:.2f} km")
    print(f"  Median:  {depth_median:.2f} km")
    print(f"  Std:     {depth_std:.2f} km")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.getcwd()
    os.makedirs(output_dir, exist_ok=True)
    
    # Plot 1: Distance histogram
    fig1, ax1 = plt.subplots(figsize=(10, 6))
    ax1.hist(distances, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.axvline(dist_mean, color='red', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Mean: {dist_mean:.2f} km')
    ax1.axvline(dist_median, color='green', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Median: {dist_median:.2f} km')
    ax1.set_xlabel('Source Distance (km)', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_yscale('log')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path1 = os.path.join(output_dir, f'{dataset_name}_distance_distribution.png')
    plt.savefig(output_path1, dpi=300, bbox_inches='tight')
    print(f"\nDistance plot saved to: {output_path1}")
    plt.close()
    
    # Plot 2: Depth histogram
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.hist(depths, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax2.axvline(depth_mean, color='red', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Mean: {depth_mean:.2f} km')
    ax2.axvline(depth_median, color='green', linestyle='--', linewidth=2, alpha=0.5,
                label=f'Median: {depth_median:.2f} km')
    ax2.set_xlabel('Source Depth (km)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_yscale('log')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path2 = os.path.join(output_dir, f'{dataset_name}_depth_distribution.png')
    plt.savefig(output_path2, dpi=300, bbox_inches='tight')
    print(f"Depth plot saved to: {output_path2}")
    plt.close()
    
    return output_path1, output_path2


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
            waveform_colors = ['steelblue', 'coral', 'mediumseagreen']
            for i in range(sample["X"].shape[0]):
                axs[0].plot(sample["X"][i], label=channel_names[i], 
                           color=waveform_colors[i], linewidth=1.5, alpha=0.8)
            axs[0].set_ylabel("Waveform", fontsize=12)
            axs[0].legend(fontsize=10)
            axs[0].grid(True, alpha=0.3)

            # Plot P, S, Noise phase labels with distinctive styles
            phase_names = ["P", "S", "Noise"]
            colors = ["steelblue", "coral", "mediumseagreen"]
            linestyles = ["-", "--", ":"]  # solid, dashed, dotted
            linewidths = [2, 2, 2]
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
            axs[1].set_ylabel("Phase Label", fontsize=12)
            axs[1].legend(fontsize=10)
            axs[1].grid(True, alpha=0.3)

            # Plot magnitude label
            if "magnitude" in sample:
                mag_data = sample["magnitude"]
                axs[2].plot(mag_data, color="steelblue", linewidth=2)
                axs[2].set_ylabel("Magnitude Label", fontsize=12)
            else:
                axs[2].text(0.5, 0.5, "No magnitude label", ha="center", va="center")
            axs[2].set_xlabel("Sample Index", fontsize=12)
            axs[2].grid(True, alpha=0.3)

            plt.show()

            if single:
                return

    except KeyboardInterrupt:
        print("\nStopped by user")


def plot_scalar_summary(pred, target, mse, rmse, mae, r2, test_data, output_dir, timestamp, model_name="model"):
    """
    Plot summary of scalar predictions (7 individual plots).
    
    Args:
        pred: Predicted magnitude values (numpy array)
        target: True magnitude values (numpy array)
        mse: Mean squared error
        rmse: Root mean squared error
        mae: Mean absolute error
        r2: R² score
        test_data: Test dataset with metadata (BenchmarkDataset split)
        output_dir: Directory to save plots
        timestamp: Timestamp string for filename uniqueness
        model_name: Name of the model for filename prefix (default: "model")
    """
    residuals = pred - target
    std_dev = np.std(residuals)
    
    # Plot 1: Scatter - predicted vs true with stats box
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(target, pred, alpha=0.6, s=20, edgecolors='k', linewidths=0.5)
    ax.plot([target.min(), target.max()], [target.min(), target.max()], "r--", lw=2, label="Perfect Prediction")
    ax.set_xlabel("True Magnitude", fontsize=12, fontweight='bold')
    ax.set_ylabel("Predicted Magnitude", fontsize=12, fontweight='bold')
    # ax.set_title(f"{model_name}: Predicted vs True Magnitude", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=10)
    
    # # Add statistics box
    # textstr = '\n'.join([
    #     f'$R^2$ = {r2:.4f}',
    #     f'MAE = {mae:.4f}',
    #     f'RMSE = {rmse:.4f}',
    #     f'Std Dev = {std_dev:.4f}',
    #     f'N = {len(pred)}'
    # ])
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
    #         verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_scatter_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved scatter plot to: {save_path}")
    plt.close()

    # Plot 2: Residual plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(target, residuals, alpha=0.6, s=20, edgecolors='k', linewidths=0.5)
    ax.axhline(y=0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("True Magnitude", fontsize=12, fontweight='bold')
    ax.set_ylabel("Residual (Predicted - True)", fontsize=12, fontweight='bold')
    # ax.set_title(f"Residual Plot", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # # Add statistics box
    # textstr = '\n'.join([
    #     f'RMSE = {rmse:.4f}',
    #     f'MAE = {mae:.4f}',
    #     f'Std Dev = {std_dev:.4f}',
    #     f'Mean = {np.mean(residuals):.4f}'
    # ])
    # props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    # ax.text(0.95, 0.95, textstr, transform=ax.transAxes, fontsize=11,
    #         verticalalignment='top', horizontalalignment='right', bbox=props, family='monospace')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_residuals_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved residual plot to: {save_path}")
    plt.close()

    # Plot 3: Histogram of residuals
    fig, ax = plt.subplots(figsize=(10, 8))
    n, bins, patches = ax.hist(residuals, bins=50, alpha=0.7, edgecolor='black', color='steelblue')
    ax.axvline(x=0, color="r", linestyle="--", lw=2)
    ax.set_xlabel("Residual (Predicted - True)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Count", fontsize=12, fontweight='bold')
    # ax.set_title(f"Residual Distribution", fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # # Add statistics box
    # textstr = '\n'.join([
    #     f'MAE = {mae:.4f}',
    #     f'RMSE = {rmse:.4f}',
    #     f'Std Dev = {std_dev:.4f}',
    #     f'Mean = {np.mean(residuals):.4f}',
    #     f'Median = {np.median(residuals):.4f}'
    # ])
    # props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    # ax.text(0.72, 0.95, textstr, transform=ax.transAxes, fontsize=11,
    #         verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_histogram_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved histogram plot to: {save_path}")
    plt.close()

    # Plot 4: Magnitude distribution comparison
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.hist(target, bins=30, alpha=0.7, label="True", edgecolor='black', color='blue')
    ax.hist(pred, bins=30, alpha=0.7, label="Predicted", edgecolor='black', color='red')
    ax.set_xlabel("Magnitude", fontsize=12, fontweight='bold')
    ax.set_ylabel("Frequency", fontsize=12, fontweight='bold')
    # ax.set_title("Magnitude Distribution Comparison", fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_distribution_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved distribution plot to: {save_path}")
    plt.close()

    # Plot 5: Binned box-and-whisker plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create bin centers at 0.5, 1.0, 1.5, 2.0, etc.
    bin_width = 0.5
    bin_centers = np.arange(0.5, np.ceil(target.max()) + bin_width, bin_width)
    
    # Create bin edges around centers (e.g., center 0.5 has edges 0.25-0.75)
    bin_edges = np.concatenate([
        [bin_centers[0] - bin_width/2],  # First edge
        bin_centers + bin_width/2         # Upper edges for all bins
    ])
    
    # Assign each sample to a bin
    bin_indices = np.digitize(target, bin_edges)
    
    # Prepare data for box plot - group predictions by bin
    boxplot_data = []
    bin_labels = []
    valid_bin_centers = []
    
    for i in range(len(bin_centers)):
        # bin_indices == i+1 because digitize returns 1-indexed bins
        mask = bin_indices == (i + 1)
        if np.sum(mask) > 0:  # Only include bins with data
            boxplot_data.append(pred[mask])
            bin_labels.append(f"{bin_edges[i]:.2f}-{bin_edges[i+1]:.2f}")
            valid_bin_centers.append(bin_centers[i])
    
    # Create box plot
    bp = ax.boxplot(boxplot_data, positions=valid_bin_centers, widths=0.35,
                    patch_artist=True, showfliers=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7),
                    medianprops=dict(color='red', linewidth=2),
                    whiskerprops=dict(color='black', linewidth=1.5),
                    capprops=dict(color='black', linewidth=1.5),
                    flierprops=dict(marker='o', markerfacecolor='gray', markersize=4, alpha=0.5))
    
    # Add perfect prediction line
    ax.plot([target.min(), target.max()], [target.min(), target.max()], 
            "r--", lw=2, label="Perfect Prediction", zorder=1)
    
    ax.set_xlabel("True Magnitude (Binned)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Predicted Magnitude Distribution", fontsize=12, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    # # Add statistics box
    # textstr = '\n'.join([
    #     f'$R^2$ = {r2:.4f}',
    #     f'MAE = {mae:.4f}',
    #     f'RMSE = {rmse:.4f}',
    #     f'Bin Width = 0.5',
    #     f'N = {len(pred)}'
    # ])
    # props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    # ax.text(0.95, 0.05, textstr, transform=ax.transAxes, fontsize=11,
    #         verticalalignment='bottom', horizontalalignment='right', bbox=props, family='monospace')
    
    plt.tight_layout()
    save_path = os.path.join(output_dir, f"{model_name}_boxplot_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved box-and-whisker plot to: {save_path}")
    plt.close()

    # Plot 6: Error vs Depth
    if "source_depth_km" in test_data.metadata.columns:
        # Get depth values directly from test data
        depths = test_data.metadata["source_depth_km"].values
        
        # Filter out NaN values
        valid_mask = ~np.isnan(depths)
        if valid_mask.sum() > 0:
            depths_valid = depths[valid_mask]
            errors_valid = residuals[valid_mask]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(depths_valid, errors_valid, alpha=0.5, s=20, 
                               c='steelblue', edgecolors='k', linewidths=0.3)
            
            # Add horizontal line at zero error
            ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
            
            ax.set_xlabel("Source Depth (km)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Prediction Error", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # # Add statistics box
            # corr = np.corrcoef(depths_valid, errors_valid)[0, 1]
            # textstr = '\n'.join([
            #     f'Samples = {len(depths_valid)}',
            #     f'Correlation = {corr:.3f}',
            #     f'Mean Error = {errors_valid.mean():.4f}',
            #     f'RMSE = {np.sqrt((errors_valid**2).mean()):.4f}'
            # ])
            # props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.8)
            # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            #        verticalalignment='top', bbox=props, family='monospace')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{model_name}_error_vs_depth_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved error vs depth plot to: {save_path}")
            plt.close()
        else:
            print("Warning: No valid depth data available for error vs depth plot")
    else:
        print("Warning: 'source_depth_km' not found in metadata, skipping error vs depth plot")

    # Plot 7: Error vs SNR
    if "trace_snr_db" in test_data.metadata.columns:
        # Get SNR values directly from test data
        snr_series = get_mean_snr_series(test_data)
        snr_values = snr_series.values
        
        # Filter out NaN values
        valid_mask = ~np.isnan(snr_values)
        if valid_mask.sum() > 0:
            snr_valid = snr_values[valid_mask]
            errors_valid = residuals[valid_mask]
            
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(snr_valid, errors_valid, alpha=0.5, s=20,
                               c='steelblue', edgecolors='k', linewidths=0.3)
            
            # Add horizontal line at zero error
            ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
            
            ax.set_xlabel("Signal-to-Noise Ratio (dB)", fontsize=12, fontweight='bold')
            ax.set_ylabel("Prediction Error", fontsize=12, fontweight='bold')
            ax.legend(fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # # Add statistics box
            # corr = np.corrcoef(snr_valid, errors_valid)[0, 1]
            # textstr = '\n'.join([
            #     f'Samples = {len(snr_valid)}',
            #     f'Correlation = {corr:.3f}',
            #     f'Mean Error = {errors_valid.mean():.4f}',
            #     f'RMSE = {np.sqrt((errors_valid**2).mean()):.4f}'
            # ])
            # props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
            # ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
            #        verticalalignment='top', bbox=props, family='monospace')
            
            plt.tight_layout()
            save_path = os.path.join(output_dir, f"{model_name}_error_vs_snr_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved error vs SNR plot to: {save_path}")
            plt.close()
        else:
            print("Warning: No valid SNR data available for error vs SNR plot")
    else:
        print("Warning: 'trace_snr_db' not found in metadata, skipping error vs SNR plot")
