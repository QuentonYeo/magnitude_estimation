"""
Evaluation functions for AMAG_v2 MagnitudeNet model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import seisbench.data as sbd
from my_project.models.AMAG_v2.model import MagnitudeNet
from my_project.loaders import data_loader as dl
import os
from datetime import datetime


def evaluate_magnitude_net(
    model: MagnitudeNet,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 256,
    plot_examples: bool = True,
    num_examples: int = 5,
    output_dir: str = None,
):
    """
    Evaluate MagnitudeNet model for magnitude regression.

    Args:
        model: MagnitudeNet model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
        output_dir: Directory to save plots (if None, uses model directory)
    """
    print(f"Evaluating MagnitudeNet from {model_path}")

    # Load model weights
    checkpoint = torch.load(model_path, map_location=model.device, weights_only=True)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

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
    all_mags = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Forward pass
            x = batch["X"].to(model.device)
            x_preproc = model.annotate_batch_pre(x, {})
            predictions = model(x_preproc)

            # Get targets
            targets = batch["magnitude"].to(model.device)

            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Also store original magnitude values if available
            if "trace_local_magnitude" in batch:
                all_mags.append(batch["trace_local_magnitude"].numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0).flatten()
    all_targets = np.concatenate(all_targets, axis=0).flatten()

    if all_mags:
        all_mags = np.concatenate(all_mags, axis=0).flatten()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    print(f"Number of test samples: {len(all_predictions)}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    print("=" * 50)

    # Plot examples if requested
    if plot_examples and num_examples > 0:
        # Set up output directory
        if output_dir is None:
            output_dir = os.path.dirname(model_path)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_path = os.path.join(output_dir, f"magnitude_evaluation_{timestamp}.png")

        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # Scatter plot: Predicted vs True
        axes[0, 0].scatter(all_targets, all_predictions, alpha=0.6, s=10)
        axes[0, 0].plot(
            [all_targets.min(), all_targets.max()],
            [all_targets.min(), all_targets.max()],
            "r--",
            linewidth=2,
        )
        axes[0, 0].set_xlabel("True Magnitude")
        axes[0, 0].set_ylabel("Predicted Magnitude")
        axes[0, 0].set_title(f"Predicted vs True Magnitude (R² = {r2:.3f})")
        axes[0, 0].grid(True, alpha=0.3)

        # Residual plot
        residuals = all_predictions - all_targets
        axes[0, 1].scatter(all_targets, residuals, alpha=0.6, s=10)
        axes[0, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("True Magnitude")
        axes[0, 1].set_ylabel("Residual (Predicted - True)")
        axes[0, 1].set_title(f"Residual Plot (RMSE = {rmse:.3f})")
        axes[0, 1].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True)
        axes[1, 0].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[1, 0].set_xlabel("Residual (Predicted - True)")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title(f"Residual Distribution (MAE = {mae:.3f})")
        axes[1, 0].grid(True, alpha=0.3)

        # Magnitude distribution comparison
        axes[1, 1].hist(all_targets, bins=30, alpha=0.7, label="True", density=True)
        axes[1, 1].hist(
            all_predictions, bins=30, alpha=0.7, label="Predicted", density=True
        )
        axes[1, 1].set_xlabel("Magnitude")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Magnitude Distribution Comparison")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Evaluation plots saved to: {plot_path}")

        if plot_examples:  # Also display if requested
            plt.show()
        else:
            plt.close()

    # Return results dictionary
    results = {
        "n_samples": len(all_predictions),
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": all_predictions,
        "targets": all_targets,
        "residuals": all_predictions - all_targets,
    }

    if all_mags:
        results["original_magnitudes"] = all_mags

    return results
