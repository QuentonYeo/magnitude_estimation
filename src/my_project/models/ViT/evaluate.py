"""
Evaluation script for Vision Transformer magnitude estimation model.

Evaluation methodology aligned with UMamba v3:
- Uses .max() for target magnitude extraction (correct after P-arrival)
- Scalar predictions only (single magnitude per waveform)
- Comprehensive metrics and visualization

Important:
    The evaluation target uses y_temporal.max(dim=1)[0] instead of .mean()
    because after the P-pick, the magnitude is constant at the source_magnitude
    value. This matches the training methodology for fair comparison.
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

import seisbench.data as sbd
from my_project.models.ViT.model import ViTMagnitudeEstimator
from my_project.loaders import data_loader as dl
from my_project.utils.utils import plot_scalar_summary


def evaluate_vit_magnitude(
    model: ViTMagnitudeEstimator,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 64,  # Smaller batch size for ViT
    plot_examples: bool = True,
    num_examples: int = 5,
    output_dir: str = None,
):
    """
    Evaluate ViTMagnitudeEstimator model for magnitude regression.
    
    Evaluation methodology aligned with UMamba v3:
    - Uses .max() for target magnitude (not .mean())
    - Computes MSE, RMSE, MAE, R² on scalar predictions
    - One prediction per waveform for fair comparison

    Args:
        model: ViTMagnitudeEstimator model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
        output_dir: Directory to save plots (if None, uses model directory)
    """
    print(f"Evaluating ViT Magnitude Estimator from {model_path}")

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

    # Move model to device
    model.to_preferred_device(verbose=True)
    device = next(model.parameters()).device

    # Load test data
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
    all_waveforms = []
    attention_weights = []  # For visualization if needed
    sample_indices = []
    inference_times = []

    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Evaluating")):
            # Time the forward pass
            start_time = (
                torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            )
            end_time = (
                torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            )

            if start_time is not None:
                start_time.record()

            x = batch["X"].to(device)
            y_temporal = batch["magnitude"].to(device)  # (batch, samples)

            # CRITICAL: Use .max() instead of .mean() to get true magnitude
            # After P-pick, magnitude is constant at source_magnitude value
            # Using .mean() incorrectly averages with pre-P zeros, lowering the target
            if y_temporal.dim() == 2:
                y_true = y_temporal.max(dim=1)[0]  # (batch,) - matches UMamba v3 methodology
            else:
                y_true = y_temporal.squeeze()

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)  # Shape: (batch,)
            y_pred = y_pred.squeeze()  # Ensure 1D

            if end_time is not None:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = (
                    start_time.elapsed_time(end_time) / 1000.0
                )  # Convert to seconds
                inference_times.append(inference_time)

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
                for i in range(min(x.shape[0], num_examples - len(sample_indices))):
                    sample_indices.append(batch_start + i)

    # Concatenate all results
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    all_waveforms = np.concatenate(all_waveforms, axis=0)

    print(f"Predictions shape: {all_predictions.shape}")
    print(f"Targets shape: {all_targets.shape}")

    # Calculate metrics
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # Calculate additional metrics
    residuals = all_predictions - all_targets
    std_residuals = np.std(residuals)

    # Magnitude-specific metrics
    magnitude_range = np.max(all_targets) - np.min(all_targets)
    normalized_rmse = rmse / magnitude_range if magnitude_range > 0 else float("inf")

    # Calculate timing metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = np.sum(inference_times) if inference_times else 0

    print("\n" + "=" * 60)
    print("VIT MAGNITUDE ESTIMATOR EVALUATION RESULTS")
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
    print(f"Standard deviation of residuals: {std_residuals:.4f}")
    print(f"Normalized RMSE: {normalized_rmse:.4f}")
    print(
        f"Target magnitude range: [{np.min(all_targets):.2f}, {np.max(all_targets):.2f}]"
    )
    print(
        f"Predicted magnitude range: [{np.min(all_predictions):.2f}, {np.max(all_predictions):.2f}]"
    )
    print("=" * 60)

    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    os.makedirs(output_dir, exist_ok=True)

    # Create plots
    if plot_examples:
        print(f"\nCreating evaluation plots...")
        
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Generate standardized scalar summary plots
        plot_scalar_summary(
            all_predictions,
            all_targets,
            mse,
            rmse,
            mae,
            r2,
            test_data,
            output_dir,
            timestamp,
            model_name="vit"
        )

        # 1. Scatter plot: Predicted vs True magnitudes
        plt.figure(figsize=(15, 12))

        # Main scatter plot
        plt.subplot(2, 3, 1)
        plt.scatter(all_targets, all_predictions, alpha=0.6, s=10)
        plt.plot(
            [all_targets.min(), all_targets.max()],
            [all_targets.min(), all_targets.max()],
            "r--",
            lw=2,
        )
        plt.xlabel("True Magnitude")
        plt.ylabel("Predicted Magnitude")
        plt.title(f"Predicted vs True Magnitudes\nR² = {r2:.3f}, RMSE = {rmse:.3f}")
        plt.grid(True, alpha=0.3)

        # 2. Residuals plot
        plt.subplot(2, 3, 2)
        plt.scatter(all_targets, residuals, alpha=0.6, s=10)
        plt.axhline(y=0, color="r", linestyle="--")
        plt.xlabel("True Magnitude")
        plt.ylabel("Residuals (Predicted - True)")
        plt.title(f"Residuals vs True Magnitudes\nStd = {std_residuals:.3f}")
        plt.grid(True, alpha=0.3)

        # 3. Histogram of residuals
        plt.subplot(2, 3, 3)
        plt.hist(residuals, bins=50, alpha=0.7, density=True)
        plt.xlabel("Residuals")
        plt.ylabel("Density")
        plt.title("Distribution of Residuals")
        plt.axvline(x=0, color="r", linestyle="--")
        plt.grid(True, alpha=0.3)

        # 4. Magnitude distribution comparison
        plt.subplot(2, 3, 4)
        plt.hist(all_targets, bins=50, alpha=0.7, label="True", density=True)
        plt.hist(all_predictions, bins=50, alpha=0.7, label="Predicted", density=True)
        plt.xlabel("Magnitude")
        plt.ylabel("Density")
        plt.title("Magnitude Distribution")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 5. Error vs magnitude
        plt.subplot(2, 3, 5)
        abs_errors = np.abs(residuals)
        plt.scatter(all_targets, abs_errors, alpha=0.6, s=10)
        plt.xlabel("True Magnitude")
        plt.ylabel("Absolute Error")
        plt.title("Absolute Error vs True Magnitude")
        plt.grid(True, alpha=0.3)

        # 6. Cumulative error distribution
        plt.subplot(2, 3, 6)
        sorted_abs_errors = np.sort(abs_errors)
        cumulative_prob = np.arange(1, len(sorted_abs_errors) + 1) / len(
            sorted_abs_errors
        )
        plt.plot(sorted_abs_errors, cumulative_prob)
        plt.xlabel("Absolute Error")
        plt.ylabel("Cumulative Probability")
        plt.title("Cumulative Error Distribution")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        results_plot_path = os.path.join(output_dir, "evaluation_results.png")
        plt.savefig(results_plot_path, dpi=300, bbox_inches="tight")
        print(f"Results plot saved to: {results_plot_path}")

        # 2. Example waveforms and predictions
        if num_examples > 0:
            plt.figure(figsize=(15, 3 * min(num_examples, 5)))

            examples_to_show = min(num_examples, len(sample_indices), 5)
            for i in range(examples_to_show):
                idx = sample_indices[i]
                if idx < len(all_waveforms):
                    waveform = all_waveforms[idx]  # Shape: (3, 3001)
                    true_mag = all_targets[idx]
                    pred_mag = all_predictions[idx]

                    plt.subplot(examples_to_show, 1, i + 1)

                    # Plot all three components
                    time_axis = np.arange(waveform.shape[1]) / 100  # Convert to seconds
                    for comp_idx, comp_name in enumerate(["E", "N", "Z"]):
                        plt.plot(
                            time_axis,
                            waveform[comp_idx],
                            label=f"{comp_name}",
                            alpha=0.7,
                        )

                    plt.xlabel("Time (s)")
                    plt.ylabel("Amplitude")
                    plt.title(
                        f"Example {i+1}: True Mag = {true_mag:.2f}, Predicted Mag = {pred_mag:.2f}"
                    )
                    plt.legend()
                    plt.grid(True, alpha=0.3)

            plt.tight_layout()
            examples_plot_path = os.path.join(output_dir, "example_waveforms.png")
            plt.savefig(examples_plot_path, dpi=300, bbox_inches="tight")
            print(f"Example waveforms plot saved to: {examples_plot_path}")

    # Save detailed results
    results_dict = {
        "n_samples": len(all_predictions),
        "model_params": total_params,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "std_residuals": std_residuals,
        "normalized_rmse": normalized_rmse,
        "target_range": [float(np.min(all_targets)), float(np.max(all_targets))],
        "prediction_range": [
            float(np.min(all_predictions)),
            float(np.max(all_predictions)),
        ],
        "predictions": all_predictions.tolist(),
        "targets": all_targets.tolist(),
        "residuals": residuals.tolist(),
        "avg_inference_time": avg_inference_time,
        "total_inference_time": total_inference_time,
    }

    results_file = os.path.join(output_dir, "evaluation_results.pt")
    torch.save(results_dict, results_file)
    print(f"Detailed results saved to: {results_file}")

    return results_dict


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ViT Magnitude Estimation Model"
    )
    parser.add_argument("--model_path", required=True, help="Path to trained model")
    parser.add_argument(
        "--dataset",
        default="ETHZ",
        choices=["ETHZ", "GEOFON", "STEAD"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--batch_size", type=int, default=64, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num_examples", type=int, default=5, help="Number of example plots"
    )
    parser.add_argument(
        "--output_dir", help="Output directory for plots (default: model directory)"
    )
    parser.add_argument(
        "--embed_dim", type=int, default=100, help="Embedding dimension"
    )
    parser.add_argument(
        "--num_heads", type=int, default=4, help="Number of attention heads"
    )
    parser.add_argument(
        "--num_blocks", type=int, default=4, help="Number of transformer blocks"
    )
    parser.add_argument("--patch_size", type=int, default=5, help="Patch size")

    args = parser.parse_args()

    # Load dataset
    if args.dataset == "ETHZ":
        data = sbd.ETHZ(sampling_rate=100)
    elif args.dataset == "GEOFON":
        data = sbd.GEOFON(sampling_rate=100)
    elif args.dataset == "STEAD":
        data = sbd.STEAD(sampling_rate=100)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Initialize model with same architecture as training
    model = ViTMagnitudeEstimator(
        in_channels=3,
        sampling_rate=100,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_blocks,
        patch_size=args.patch_size,
        norm="std",
    )

    # Evaluate model
    results = evaluate_vit_magnitude(
        model=model,
        model_path=args.model_path,
        data=data,
        batch_size=args.batch_size,
        plot_examples=True,
        num_examples=args.num_examples,
        output_dir=args.output_dir,
    )

    print(f"\nEvaluation completed!")
    print(f"Final RMSE: {results['rmse']:.6f}")
    print(f"Final R²: {results['r2']:.6f}")
