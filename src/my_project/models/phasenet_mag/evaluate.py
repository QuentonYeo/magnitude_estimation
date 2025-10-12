import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import seisbench.data as sbd
from my_project.models.phasenet_mag.model import PhaseNetMag
from my_project.loaders import data_loader as dl


def evaluate_phasenet_mag(
    model: PhaseNetMag,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 256,
    plot_examples: bool = True,
    num_examples: int = 5,
):
    """
    Evaluate PhaseNetMag model for magnitude regression.

    Args:
        model: PhaseNetMag model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
    """
    print(f"Evaluating PhaseNetMag from {model_path}")

    # Load model weights
    state_dict = torch.load(model_path, map_location=model.device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model weights from {model_path}")

    # Load test data
    test_generator, test_loader, _ = dl.load_dataset(
        data, model, "test", batch_size=batch_size
    )
    print(f"Test samples: {len(test_generator)}")

    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    all_waveforms = []
    sample_indices = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            x = batch["X"].to(model.device)
            y_true = batch["magnitude"].to(model.device)

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)
            y_pred = y_pred.squeeze(1)  # Remove channel dimension

            # Move to CPU and convert to numpy
            y_pred_np = y_pred.cpu().numpy()
            y_true_np = y_true.cpu().numpy()
            x_np = x.cpu().numpy()

            all_predictions.append(y_pred_np)
            all_targets.append(y_true_np)
            all_waveforms.append(x_np)

            # Store some sample indices for plotting
            if len(sample_indices) < num_examples:
                batch_start = batch_idx * batch_size
                for i in range(min(batch_size, num_examples - len(sample_indices))):
                    sample_indices.append(batch_start + i)

    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_waveforms = np.concatenate(all_waveforms, axis=0)

    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Targets shape: {all_targets.shape}")

    # Calculate metrics on flattened arrays
    y_true_flat = all_targets.flatten()
    y_pred_flat = all_predictions.flatten()

    # Remove NaN values if any
    mask = ~(np.isnan(y_true_flat) | np.isnan(y_pred_flat))
    y_true_clean = y_true_flat[mask]
    y_pred_clean = y_pred_flat[mask]

    mse = mean_squared_error(y_true_clean, y_pred_clean)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true_clean, y_pred_clean)
    r2 = r2_score(y_true_clean, y_pred_clean)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Test samples: {len(test_generator)}")
    print(f"Valid predictions: {len(y_true_clean)} / {len(y_true_flat)}")
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R²:   {r2:.6f}")

    # Calculate magnitude-specific metrics
    unique_mags = np.unique(y_true_clean[y_true_clean > 0])  # Only non-zero magnitudes
    print(
        f"\nMagnitude range in test set: {unique_mags.min():.2f} - {unique_mags.max():.2f}"
    )
    print(f"Number of unique magnitudes: {len(unique_mags)}")

    if plot_examples:
        plot_prediction_examples(
            all_waveforms,
            all_targets,
            all_predictions,
            num_examples=min(num_examples, len(all_waveforms)),
        )
        plot_magnitude_distribution_comparison(y_true_clean, y_pred_clean)
        plot_prediction_scatter(y_true_clean, y_pred_clean)

    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": all_predictions,
        "targets": all_targets,
        "waveforms": all_waveforms,
    }


def plot_prediction_examples(waveforms, targets, predictions, num_examples=5):
    """Plot example waveforms with true and predicted magnitude labels."""
    fig, axes = plt.subplots(num_examples, 1, figsize=(15, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]

    channel_names = ["Z", "N", "E"]
    colors = ["tab:blue", "tab:orange", "tab:green"]

    for i in range(num_examples):
        ax = axes[i]

        # Plot waveforms
        for ch in range(3):
            ax.plot(
                waveforms[i, ch],
                color=colors[ch],
                alpha=0.7,
                label=f"{channel_names[ch]}" if i == 0 else "",
            )

        # Plot magnitude predictions
        ax2 = ax.twinx()
        ax2.plot(
            targets[i], "r-", linewidth=2, label="True magnitude" if i == 0 else ""
        )
        ax2.plot(
            predictions[i],
            "b--",
            linewidth=2,
            label="Predicted magnitude" if i == 0 else "",
        )

        ax.set_ylabel("Amplitude")
        ax2.set_ylabel("Magnitude")
        ax.set_title(f"Example {i+1}")

        if i == 0:
            ax.legend(loc="upper left")
            ax2.legend(loc="upper right")

        if i == num_examples - 1:
            ax.set_xlabel("Sample")

    plt.tight_layout()
    # plt.show()


def plot_magnitude_distribution_comparison(y_true, y_pred):
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
    plt.show()


def plot_prediction_scatter(y_true, y_pred):
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
    plt.show()


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
    parser.add_argument("--no_plot", action="store_true", help="Skip plotting examples")
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
    model = PhaseNetMag(in_channels=3, sampling_rate=100, norm="std", filter_factor=1)
    model.to_preferred_device(verbose=True)  # NOTE for cpu inference, move to CPU

    # Evaluate model
    results = evaluate_phasenet_mag(
        model=model,
        model_path=args.model_path,
        data=data,
        batch_size=args.batch_size,
        plot_examples=not args.no_plot,
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
