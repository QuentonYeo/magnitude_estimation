import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import seisbench.data as sbd
from my_project.models.phasenet_mag_v2.model import PhaseNetMagv2
from my_project.loaders import data_loader as dl


def evaluate_phasenet_mag_v2(
    model: PhaseNetMagv2,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 256,
    plot_examples: bool = True,
    num_examples: int = 5,
    output_dir: str = None,
):
    """
    Evaluate PhaseNetMag model for scalar magnitude regression.
    
    The model predicts a single magnitude value per waveform.
    Evaluation compares predictions against the maximum of the labeled magnitudes
    (which corresponds to the true event magnitude after P-arrival).

    Args:
        model: PhaseNetMag model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
        output_dir: Directory to save plots (if None, uses model directory)
    """
    print("\n" + "=" * 60)
    print("EVALUATING PHASENETMAG V2 (SCALAR PREDICTION)")
    print("=" * 60)
    print(f"Model path: {model_path}")
    print("NOTE: Metrics computed on SCALAR predictions (1 per waveform)")
    print("=" * 60)

    # Load model checkpoint or weights
    checkpoint = torch.load(model_path, map_location=model.device, weights_only=True)

    # Handle different checkpoint formats
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        # Full checkpoint format
        state_dict = checkpoint["model_state_dict"]
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        if "val_loss" in checkpoint:
            print(f"Checkpoint validation loss: {checkpoint['val_loss']:.6f}")
        if "best_val_loss" in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    else:
        # Legacy weights format
        state_dict = checkpoint
        print("Loaded legacy model weights format")

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Model loaded and set to evaluation mode")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Load test data
    test_generator, test_loader, _ = dl.load_dataset(
        data, model, "test", batch_size=batch_size
    )
    print(f"Test samples: {len(test_generator)}")
    print(f"Test batches: {len(test_loader)}")

    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    all_waveforms = []
    all_temporal_labels = []  # Store for plotting
    sample_indices = []
    inference_times = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Time the forward pass
            start_time = (
                torch.cuda.Event(enable_timing=True) if model.device.type == "cuda" else None
            )
            end_time = (
                torch.cuda.Event(enable_timing=True) if model.device.type == "cuda" else None
            )

            if start_time is not None:
                start_time.record()

            x = batch["X"].to(model.device)
            y_temporal = batch["magnitude"].to(model.device)  # (batch, samples)
            # True magnitude is the max value (after P-pick it's constant at source_magnitude)
            y_scalar = y_temporal.max(dim=1)[0]  # (batch,) - scalar target

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)  # (batch,) - scalar output

            if end_time is not None:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = (
                    start_time.elapsed_time(end_time) / 1000.0
                )  # Convert to seconds
                inference_times.append(inference_time)

            # Move to CPU and convert to numpy
            y_pred_np = y_pred.cpu().numpy()
            y_scalar_np = y_scalar.cpu().numpy()
            y_temporal_np = y_temporal.cpu().numpy()
            x_np = x.cpu().numpy()

            all_predictions.append(y_pred_np)
            all_targets.append(y_scalar_np)
            all_temporal_labels.append(y_temporal_np)
            all_waveforms.append(x_np)

            # Store some sample indices for plotting
            if len(sample_indices) < num_examples:
                batch_start = batch_idx * batch_size
                for i in range(min(batch_size, num_examples - len(sample_indices))):
                    sample_indices.append(batch_start + i)

    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)  # (n_samples,)
    all_targets = np.concatenate(all_targets, axis=0)  # (n_samples,)
    all_temporal_labels = np.concatenate(all_temporal_labels, axis=0)  # (n_samples, time_steps)
    all_waveforms = np.concatenate(all_waveforms, axis=0)

    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Targets shape: {all_targets.shape}")

    # Calculate metrics on scalar predictions
    y_true = all_targets
    y_pred = all_predictions

    # Remove NaN values if any
    mask = ~(np.isnan(y_true) | np.isnan(y_pred))
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)

    # Calculate timing metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = np.sum(inference_times) if inference_times else 0

    print("\n" + "=" * 60)
    print("PHASENETMAG EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of test samples: {len(test_generator)}")
    print(f"Valid predictions: {len(y_true_clean)} / {len(y_true)}")
    print(f"Model parameters: {total_params:,}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    if inference_times:
        print(f"Average inference time: {avg_inference_time:.4f}s per batch")
        print(f"Total inference time: {total_inference_time:.2f}s")

    # Calculate magnitude-specific metrics
    unique_mags = np.unique(y_true_clean[y_true_clean > 0])  # Only non-zero magnitudes
    print(
        f"Magnitude range in test set: {unique_mags.min():.2f} - {unique_mags.max():.2f}"
    )
    print(f"Number of unique magnitudes: {len(unique_mags)}")
    print("=" * 60)

    if plot_examples:
        if output_dir is None:
            output_dir = os.path.dirname(model_path)

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create standard scalar evaluation plots (same as UMamba V3)
        plot_scalar_summary(
            y_pred_clean, 
            y_true_clean, 
            mse, 
            rmse, 
            mae, 
            r2, 
            output_dir, 
            timestamp
        )
        
        # Plot example waveforms with scalar predictions
        if num_examples > 0:
            plot_waveform_examples(
                all_waveforms,
                all_temporal_labels,
                all_predictions,
                all_targets,
                num_examples=min(num_examples, len(all_waveforms)),
                output_dir=output_dir,
                timestamp=timestamp
            )

        # Show all plots at once
        plt.show()

    return {
        "n_samples": len(test_generator),
        "valid_predictions": len(y_true_clean),
        "model_params": total_params,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": all_predictions,
        "targets": all_targets,
        "temporal_labels": all_temporal_labels,
        "waveforms": all_waveforms,
        "avg_inference_time": avg_inference_time,
        "total_inference_time": total_inference_time,
    }


def plot_scalar_summary(pred, target, mse, rmse, mae, r2, output_dir, timestamp):
    """
    Plot summary of scalar predictions (4 subplots).
    Standard evaluation plots matching UMamba V3 and other scalar models.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Scatter: predicted vs true
    axes[0, 0].scatter(target, pred, alpha=0.6, s=10)
    axes[0, 0].plot([target.min(), target.max()], [target.min(), target.max()], "r--", lw=2)
    axes[0, 0].set_xlabel("True Magnitude")
    axes[0, 0].set_ylabel("Predicted Magnitude")
    axes[0, 0].set_title(f"PhaseNetMag V2: Predicted vs True (R² = {r2:.3f})")
    axes[0, 0].grid(True, alpha=0.3)

    # Residual plot
    residuals = pred - target
    axes[0, 1].scatter(target, residuals, alpha=0.6, s=10)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("True Magnitude")
    axes[0, 1].set_ylabel("Residual (Predicted - True)")
    axes[0, 1].set_title(f"Residual Plot (RMSE = {rmse:.3f})")
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[1, 0].set_xlabel("Residual (Predicted - True)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title(f"Residual Distribution (MAE = {mae:.3f})")
    axes[1, 0].grid(True, alpha=0.3)

    # Magnitude distribution comparison
    axes[1, 1].hist(target, bins=30, alpha=0.7, label="True", density=True, edgecolor='black')
    axes[1, 1].hist(pred, bins=30, alpha=0.7, label="Predicted", density=True, edgecolor='black')
    axes[1, 1].set_xlabel("Magnitude")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Magnitude Distribution Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"phasenet_mag_v2_evaluation_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved scalar evaluation plots to: {save_path}")
    plt.close()


def plot_waveform_examples(waveforms, temporal_labels, scalar_pred, scalar_targ, 
                          num_examples=5, output_dir=None, timestamp=None):
    """
    Plot example waveforms with scalar predictions.
    Shows 3-channel waveforms and scalar magnitude predictions as horizontal lines.
    
    Args:
        waveforms: (n_samples, 3, time_steps) - raw waveforms
        temporal_labels: (n_samples, time_steps) - temporal magnitude labels (for context)
        scalar_pred: (n_samples,) - predicted scalar magnitudes
        scalar_targ: (n_samples,) - true scalar magnitudes
        num_examples: Number of examples to plot
        output_dir: Directory to save plots
        timestamp: Timestamp string for filename
    """
    num_examples = min(num_examples, len(waveforms))
    
    for idx in range(num_examples):
        fig, axes = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot 1: Waveforms
        ax1 = axes[0]
        channel_names = ["Z (Vertical)", "N (North)", "E (East)"]
        colors = ["tab:blue", "tab:orange", "tab:green"]
        
        for ch in range(3):
            ax1.plot(waveforms[idx, ch], label=channel_names[ch], 
                    color=colors[ch], alpha=0.7, linewidth=1)
        
        ax1.set_ylabel("Amplitude")
        ax1.set_title(f"Example {idx + 1}: 3-Channel Seismic Waveform")
        ax1.legend(loc="upper right")
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Temporal labels + scalar predictions
        ax2 = axes[1]
        
        # Show temporal labels for context (how the data is labeled)
        ax2.plot(temporal_labels[idx], 'k-', linewidth=1, alpha=0.5,
                label="Temporal labels (training data)")
        
        # Show scalar predictions as horizontal lines
        ax2.axhline(y=scalar_targ[idx], color='red', linestyle='-', linewidth=2,
                   label=f"True magnitude: {scalar_targ[idx]:.2f}")
        ax2.axhline(y=scalar_pred[idx], color='blue', linestyle='--', linewidth=2,
                   label=f"Predicted magnitude: {scalar_pred[idx]:.2f}")
        
        error = abs(scalar_pred[idx] - scalar_targ[idx])
        ax2.set_xlabel("Sample")
        ax2.set_ylabel("Magnitude")
        ax2.set_title(f"Magnitude Prediction (Error: {error:.3f})")
        ax2.legend(loc="upper right")
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim([-0.5, max(scalar_targ[idx], scalar_pred[idx]) + 1.0])
        
        plt.tight_layout()
        
        if output_dir is not None:
            save_path = os.path.join(output_dir, 
                                    f"phasenet_mag_v2_example_{idx+1}_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Saved example {idx + 1} to: {save_path}")
        
        plt.close()


def plot_magnitude_distribution_comparison(y_true, y_pred, output_dir=None):
    """Plot histogram comparison of true vs predicted magnitudes."""
    # Only consider non-zero values
    y_true_nonzero = y_true[y_true > 0]
    y_pred_nonzero = y_pred[y_true > 0]  # Predicted values where true magnitude > 0

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    bins = np.arange(0, 8, 0.2)

    ax1.hist(
        y_true_nonzero, bins=bins, alpha=0.7, label="True", color="red", density=True
    )
    ax1.hist(
        y_pred_nonzero,
        bins=bins,
        alpha=0.7,
        label="Predicted",
        color="blue",
        density=True,
    )
    ax1.set_xlabel("Magnitude")
    ax1.set_ylabel("Density")
    ax1.set_title("Magnitude Distribution (Non-zero values)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot magnitude vs prediction error
    error = y_pred_nonzero - y_true_nonzero
    ax2.scatter(y_true_nonzero, error, alpha=0.5, s=1)
    ax2.axhline(y=0, color="red", linestyle="--")
    ax2.set_xlabel("True Magnitude")
    ax2.set_ylabel("Prediction Error")
    ax2.set_title("Prediction Error vs True Magnitude")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the figure if output directory is provided
    if output_dir is not None:
        output_path = os.path.join(output_dir, "magnitude_distribution_comparison.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved magnitude distribution plot to: {output_path}")


def plot_prediction_scatter(y_true, y_pred, output_dir=None):
    """Plot scatter plot of true vs predicted magnitudes."""
    # Only consider non-zero values
    mask = y_true > 0
    y_true_nonzero = y_true[mask]
    y_pred_nonzero = y_pred[mask]

    plt.figure(figsize=(8, 8))
    plt.scatter(y_true_nonzero, y_pred_nonzero, alpha=0.5, s=1)

    # Plot perfect prediction line
    min_val = min(y_true_nonzero.min(), y_pred_nonzero.min())
    max_val = max(y_true_nonzero.max(), y_pred_nonzero.max())
    plt.plot(
        [min_val, max_val],
        [min_val, max_val],
        "r--",
        linewidth=2,
        label="Perfect prediction",
    )

    plt.xlabel("True Magnitude")
    plt.ylabel("Predicted Magnitude")
    plt.title("True vs Predicted Magnitude")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.axis("equal")

    # Save the figure if output directory is provided
    if output_dir is not None:
        output_path = os.path.join(output_dir, "prediction_scatter.png")
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        print(f"Saved prediction scatter plot to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate PhaseNetMag for magnitude regression"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to trained model weights"
    )
    parser.add_argument(
        "--dataset", type=str, default="ETHZ", help="Dataset name (ETHZ, GEOFON, etc.)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size for evaluation"
    )
    parser.add_argument("--plot", action="store_true", help="Generate and save plots")
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of examples to plot"
    )

    args = parser.parse_args()

    # Load dataset
    if args.dataset == "ETHZ":
        data = sbd.ETHZ(sampling_rate=100)
    elif args.dataset == "GEOFON":
        data = sbd.GEOFON(sampling_rate=100)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"Loaded dataset: {data}")

    # Create model
    model = PhaseNetMagv2(in_channels=3, sampling_rate=100, norm="std", filter_factor=1)
    model.to_preferred_device(verbose=True)  # NOTE for cpu inference, move to CPU

    # Evaluate model
    results = evaluate_phasenet_mag_v2(
        model=model,
        model_path=args.model_path,
        data=data,
        batch_size=args.batch_size,
        plot_examples=args.plot,
        num_examples=args.num_examples,
    )

    # Save results
    results_dir = os.path.dirname(args.model_path)
    results_path = os.path.join(results_dir, "evaluation_results.txt")

    with open(results_path, "w") as f:
        f.write("PhaseNetMag Evaluation Results\n")
        f.write("=" * 30 + "\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Dataset: {args.dataset}\n")
        f.write(f"MSE: {results['mse']:.6f}\n")
        f.write(f"RMSE: {results['rmse']:.6f}\n")
        f.write(f"MAE: {results['mae']:.6f}\n")
        f.write(f"R²: {results['r2']:.6f}\n")

    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
