"""
Evaluation functions for AMAG_v3 MagnitudeNet model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import seisbench.data as sbd
from my_project.models.AMAG_v3.model import MagnitudeNet
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
    Evaluate MagnitudeNet (AMAG_v3) model for magnitude regression.
    
    AMAG_v3 outputs a SCALAR magnitude prediction per waveform (shape: batch_size,),
    unlike AMAG_v2 which outputs temporal predictions (shape: batch_size, time_steps).

    Args:
        model: MagnitudeNet model instance (AMAG_v3)
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
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        if "val_loss" in checkpoint:
            print(f"Checkpoint validation loss: {checkpoint['val_loss']:.6f}")
        if "best_val_loss" in checkpoint:
            print(f"Best validation loss: {checkpoint['best_val_loss']:.6f}")
    else:
        state_dict = checkpoint

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
    all_mags = []
    inference_times = []

    print("Running inference on test set...")
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

            # Forward pass
            x = batch["X"].to(model.device)
            x_preproc = model.annotate_batch_pre(x, {})
            predictions = model(x_preproc)  # Shape: (batch,) for v3

            if end_time is not None:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = (
                    start_time.elapsed_time(end_time) / 1000.0
                )  # Convert to seconds
                inference_times.append(inference_time)

            # Get targets - handle temporal labels from data loader
            # The magnitude labeller returns temporal labels (batch, time_steps)
            # where values are 0 before onset and magnitude after onset
            # We need to extract the scalar magnitude value
            y_temporal = batch["magnitude"].to(model.device)  # (batch, samples)
            
            # Take max to get scalar target (same as training)
            targets = y_temporal.max(dim=1)[0]  # (batch,)

            # Store predictions and targets (both are now scalars)
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Also store original magnitude values if available
            if "trace_local_magnitude" in batch:
                all_mags.append(batch["trace_local_magnitude"].numpy())

    # Concatenate all batches (predictions are already 1D per batch for v3)
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    if all_mags:
        all_mags = np.concatenate(all_mags, axis=0).flatten()

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # Calculate timing metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = np.sum(inference_times) if inference_times else 0

    print("\n" + "=" * 60)
    print("MAGNITUDENET (AMAG_V3) EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of test samples: {len(all_predictions)}")
    print(f"Model parameters: {total_params:,}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    if inference_times:
        print(f"Average inference time: {avg_inference_time:.4f}s per batch")
        print(f"Total inference time: {total_inference_time:.2f}s")
    print("=" * 60)

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
        "model_params": total_params,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": all_predictions,
        "targets": all_targets,
        "residuals": all_predictions - all_targets,
        "avg_inference_time": avg_inference_time,
        "total_inference_time": total_inference_time,
    }

    if all_mags:
        results["original_magnitudes"] = all_mags

    return results
