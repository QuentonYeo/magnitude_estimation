"""
Evaluation functions for MagNet BiLSTM magnitude estimation model.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
from typing import Optional
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

from my_project.models.MagNet.model import MagNet
from my_project.loaders import data_loader as dl
from my_project.utils.utils import plot_scalar_summary


def evaluate_magnet(
    model: MagNet,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 256,
    plot_examples: bool = False,
    num_examples: int = 5,
    output_dir: str = None,
    device: Optional[str] = None,
):
    """
    Evaluate MagNet model on test set.

    Args:
        model: MagNet model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Deprecated parameter (MagNet outputs scalar predictions only)
        num_examples: Deprecated parameter (MagNet outputs scalar predictions only)
        output_dir: Directory to save outputs (defaults to model directory)
        device: Device to use for inference
    """
    print("\n" + "=" * 60)
    print("EVALUATING MAGNET (BILSTM MAGNITUDE ESTIMATOR)")
    print("=" * 60)

    # Load model weights
    # Set device appropriately
    if device is not None:
        model = model.to(device)
        map_location = device
    else:
        map_location = "cpu"

    checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)

    checkpoint = torch.load(model_path, map_location=map_location, weights_only=False)

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
        print("Loaded legacy model weights format")

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from: {model_path}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Determine device for inference
    if device is None:
        device_used = next(model.parameters()).device
    else:
        # allow device to be a string like 'cuda:0' or an int
        if isinstance(device, int):
            device_used = torch.device(f"cuda:{device}")
        else:
            device_used = torch.device(device)
        model = model.to(device_used)

    print(f"Model is on device: {device_used}")

    # Load test data
    print("\nLoading test data...")
    test_generator, test_loader, _ = dl.load_dataset(
        data, model, "test", batch_size=batch_size
    )
    
    # Get test split dataset with metadata
    _, _, test_data = data.train_dev_test()
    
    print(f"Test samples: {len(test_generator)}")
    print(f"Test batches: {len(test_loader)}")

    # Collect all predictions and targets
    all_predictions = []
    all_targets = []
    inference_times = []

    print("Running inference on test set...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Time the forward pass
            start_event = (
                torch.cuda.Event(enable_timing=True) if device_used.type == "cuda" else None
            )
            end_event = (
                torch.cuda.Event(enable_timing=True) if device_used.type == "cuda" else None
            )

            if start_event is not None:
                start_event.record()

            # Forward pass
            x = batch["X"].to(device_used)
            x_preproc = model.annotate_batch_pre(x, {})
            predictions = model(x_preproc)  # (batch, 1)

            if end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event) / 1000.0)

            # Get targets (max of temporal magnitude labels)
            y_temporal = batch["magnitude"]  # (batch, samples)
            targets = y_temporal.max(dim=1)[0]  # (batch,)

            # Store results
            predictions_np = predictions.squeeze().cpu().numpy()
            targets_np = targets.cpu().numpy()

            # Handle single-sample batches
            if predictions_np.ndim == 0:
                predictions_np = predictions_np.reshape(1)
                targets_np = targets_np.reshape(1)

            all_predictions.append(predictions_np)
            all_targets.append(targets_np)

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # MagNet directly outputs scalar predictions
    pred_final = all_predictions
    target_final = all_targets

    # Calculate metrics
    mse = mean_squared_error(target_final, pred_final)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_final, pred_final)
    r2 = r2_score(target_final, pred_final)

    # Calculate residuals
    residuals = pred_final - target_final

    # Performance metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = np.sum(inference_times) if inference_times else 0

    # Print results
    print("\n" + "=" * 60)
    print("MAGNET EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of test samples: {len(pred_final)}")
    print(f"Model parameters: {total_params:,}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    if inference_times:
        print(f"Average inference time: {avg_inference_time:.4f}s per batch")
        print(f"Total inference time: {total_inference_time:.2f}s")
    print("=" * 60)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate standardized scalar summary plots
    plot_scalar_summary(
        pred_final,
        target_final,
        mse,
        rmse,
        mae,
        r2,
        test_data,
        output_dir,
        timestamp,
        model_name="magnet"
    )
    
    plot_path = os.path.join(output_dir, f"magnet_evaluation_{timestamp}.png")

    # Create summary figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Scatter: predicted vs true
    axes[0, 0].scatter(target_final, pred_final, alpha=0.6, s=10)
    axes[0, 0].plot([target_final.min(), target_final.max()], [target_final.min(), target_final.max()], "r--")
    axes[0, 0].set_xlabel("True Magnitude")
    axes[0, 0].set_ylabel("Predicted Magnitude")
    axes[0, 0].set_title(f"MagNet: Predicted vs True (R² = {r2:.3f})")
    axes[0, 0].grid(True, alpha=0.3)

    # Residual plot
    axes[0, 1].scatter(target_final, residuals, alpha=0.6, s=10)
    axes[0, 1].axhline(y=0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("True Magnitude")
    axes[0, 1].set_ylabel("Residual (Predicted - True)")
    axes[0, 1].set_title(f"Residual Plot (RMSE = {rmse:.3f})")
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[1, 0].axvline(x=0, color="r", linestyle="--")
    axes[1, 0].set_xlabel("Residual (Predicted - True)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title(f"Residual Distribution (MAE = {mae:.3f})")
    axes[1, 0].grid(True, alpha=0.3)

    # Magnitude distribution comparison
    axes[1, 1].hist(target_final, bins=30, alpha=0.7, label="True", density=True, edgecolor='black')
    axes[1, 1].hist(pred_final, bins=30, alpha=0.7, label="Predicted", density=True, edgecolor='black')
    axes[1, 1].set_xlabel("Magnitude")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Magnitude Distribution")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    # Error vs magnitude
    abs_errors = np.abs(residuals)
    axes[2, 0].scatter(target_final, abs_errors, alpha=0.6, s=10)
    axes[2, 0].set_xlabel("True Magnitude")
    axes[2, 0].set_ylabel("Absolute Error")
    axes[2, 0].set_title("Absolute Error vs Magnitude")
    axes[2, 0].grid(True, alpha=0.3)

    # Boxplot of absolute errors by magnitude bin
    mag_bins = np.linspace(target_final.min(), target_final.max(), 6)
    binned_errors = []
    bin_labels = []
    for i in range(len(mag_bins) - 1):
        mask = (target_final >= mag_bins[i]) & (target_final < mag_bins[i + 1])
        if mask.sum() > 0:
            binned_errors.append(abs_errors[mask])
            bin_labels.append(f"{mag_bins[i]:.1f}-{mag_bins[i+1]:.1f}")
    
    if binned_errors:
        axes[2, 1].boxplot(binned_errors, labels=bin_labels)
        axes[2, 1].set_xlabel("Magnitude Range")
        axes[2, 1].set_ylabel("Absolute Error")
        axes[2, 1].set_title("Error Distribution by Magnitude")
        axes[2, 1].tick_params(axis='x', rotation=45)
        axes[2, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Evaluation plots saved to: {plot_path}")
    plt.close()

    # Return metrics dictionary
    results = {
        "n_samples": len(pred_final),
        "model_params": total_params,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions": pred_final,
        "targets": target_final,
        "residuals": residuals,
        "avg_inference_time": avg_inference_time,
        "total_inference_time": total_inference_time,
        "plot_path": plot_path,
    }

    return results
