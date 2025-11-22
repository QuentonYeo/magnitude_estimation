"""
Evaluation functions for EQTransformerMag model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import seisbench.data as sbd
from my_project.models.EQTransformer.model import EQTransformerMag
from my_project.loaders import data_loader as dl
from my_project.utils.utils import plot_scalar_summary
import os
from datetime import datetime


def evaluate_eqtransformer_mag(
    model: EQTransformerMag,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 128,  # Smaller batch size for transformer
    plot_examples: bool = True,
    num_examples: int = 5,
    output_dir: str = None,
    device: str = None,
):
    """
    Evaluate EQTransformerMag model for magnitude regression.

    Args:
        model: EQTransformerMag model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
        output_dir: Directory to save plots (if None, uses model directory)
        device: Device to run evaluation on
    """
    print(f"Evaluating EQTransformerMag from {model_path}")

    # Set device
    if device is None:
        device = next(model.parameters()).device
    else:
        model = model.to(device)

    # Load model weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    # Handle different checkpoint formats
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
        if "val_loss" in checkpoint:
            print(f"Checkpoint validation loss: {checkpoint['val_loss']:.6f}")
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
    
    # Get test split dataset with metadata
    _, _, test_data = data.train_dev_test()
    
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
                torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            )
            end_time = (
                torch.cuda.Event(enable_timing=True) if device.type == "cuda" else None
            )

            if start_time is not None:
                start_time.record()

            # Forward pass
            x = batch["X"].to(device)
            x_preproc = model.annotate_batch_pre(x, {})
            predictions = model(x_preproc)  # Shape: (batch, samples)

            if end_time is not None:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = (
                    start_time.elapsed_time(end_time) / 1000.0
                )  # Convert to seconds
                inference_times.append(inference_time)

            # Get targets
            targets = batch["magnitude"].to(device)

            # Handle different target shapes for comparison
            if targets.dim() == 1:
                # Expand to match prediction shape
                targets = targets.unsqueeze(1).expand(-1, predictions.shape[-1])
            elif targets.dim() == 2 and targets.shape[1] == 1:
                targets = targets.expand(-1, predictions.shape[-1])

            # Store predictions and targets
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())

            # Also store original magnitude values if available
            if "trace_local_magnitude" in batch:
                all_mags.append(batch["trace_local_magnitude"].numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(
        all_predictions, axis=0
    )  # Shape: (n_samples, time_steps)
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (n_samples, time_steps)

    # For evaluation, we can take the mean prediction over time or use the final prediction
    # Since magnitude should be constant after P-arrival, let's use the final half of the prediction
    final_half_start = all_predictions.shape[1] // 2
    pred_final = np.mean(
        all_predictions[:, final_half_start:], axis=1
    )  # Mean over final half
    target_final = np.mean(
        all_targets[:, final_half_start:], axis=1
    )  # Mean over final half

    if all_mags:
        all_mags = np.concatenate(all_mags, axis=0).flatten()

    # Calculate metrics
    mse = mean_squared_error(target_final, pred_final)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_final, pred_final)
    r2 = r2_score(target_final, pred_final)

    # Calculate additional transformer-specific metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = np.sum(inference_times) if inference_times else 0

    print("\n" + "=" * 60)
    print("EQTRANSFORMERMAG EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of test samples: {len(pred_final)}")
    print(f"Model parameters: {total_params:,}")
    print(
        f"Input length: {model.in_samples} samples ({model.in_samples/model.sampling_rate:.1f}s)"
    )
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
            model_name="eqtransformermag"
        )
        
        plot_path = os.path.join(
            output_dir, f"eqtransformermag_evaluation_{timestamp}.png"
        )

        # Create comprehensive evaluation plots
        fig, axes = plt.subplots(3, 2, figsize=(15, 12))

        # Scatter plot: Predicted vs True
        axes[0, 0].scatter(target_final, pred_final, alpha=0.6, s=15)
        axes[0, 0].plot(
            [target_final.min(), target_final.max()],
            [target_final.min(), target_final.max()],
            "r--",
            linewidth=2,
        )
        axes[0, 0].set_xlabel("True Magnitude")
        axes[0, 0].set_ylabel("Predicted Magnitude")
        axes[0, 0].set_title(f"EQTransformerMag: Predicted vs True (R² = {r2:.3f})")
        axes[0, 0].grid(True, alpha=0.3)

        # Residual plot
        residuals = pred_final - target_final
        axes[0, 1].scatter(target_final, residuals, alpha=0.6, s=15)
        axes[0, 1].axhline(y=0, color="r", linestyle="--", linewidth=2)
        axes[0, 1].set_xlabel("True Magnitude")
        axes[0, 1].set_ylabel("Residual (Predicted - True)")
        axes[0, 1].set_title(f"Residual Plot (RMSE = {rmse:.3f})")
        axes[0, 1].grid(True, alpha=0.3)

        # Histogram of residuals
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True, color="blue")
        axes[1, 0].axvline(x=0, color="r", linestyle="--", linewidth=2)
        axes[1, 0].set_xlabel("Residual (Predicted - True)")
        axes[1, 0].set_ylabel("Density")
        axes[1, 0].set_title(f"Residual Distribution (MAE = {mae:.3f})")
        axes[1, 0].grid(True, alpha=0.3)

        # Magnitude distribution comparison
        axes[1, 1].hist(
            target_final, bins=30, alpha=0.7, label="True", density=True, color="green"
        )
        axes[1, 1].hist(
            pred_final,
            bins=30,
            alpha=0.7,
            label="Predicted",
            density=True,
            color="orange",
        )
        axes[1, 1].set_xlabel("Magnitude")
        axes[1, 1].set_ylabel("Density")
        axes[1, 1].set_title("Magnitude Distribution Comparison")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        # Time series examples
        n_examples_plot = min(num_examples, len(all_predictions))
        time_axis = np.arange(all_predictions.shape[1]) / model.sampling_rate

        axes[2, 0].set_title("Example Time Series Predictions")
        for i in range(n_examples_plot):
            axes[2, 0].plot(
                time_axis,
                all_predictions[i],
                alpha=0.7,
                linewidth=1,
                label=f"Pred {i+1}" if i < 3 else None,
            )
            axes[2, 0].plot(
                time_axis,
                all_targets[i],
                alpha=0.7,
                linewidth=1,
                linestyle="--",
                label=f"True {i+1}" if i < 3 else None,
            )
        axes[2, 0].set_xlabel("Time (s)")
        axes[2, 0].set_ylabel("Magnitude")
        axes[2, 0].grid(True, alpha=0.3)
        if n_examples_plot <= 3:
            axes[2, 0].legend()

        # Error statistics box plot
        error_stats = [
            np.abs(residuals),  # Absolute errors
            residuals[residuals >= 0],  # Positive residuals
            residuals[residuals < 0],  # Negative residuals
        ]
        labels = ["Absolute Errors", "Over-predictions", "Under-predictions"]

        axes[2, 1].boxplot(
            [err for err in error_stats if len(err) > 0],
            labels=[lbl for i, lbl in enumerate(labels) if len(error_stats[i]) > 0],
        )
        axes[2, 1].set_ylabel("Error Magnitude")
        axes[2, 1].set_title("Error Distribution Statistics")
        axes[2, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"Evaluation plots saved to: {plot_path}")

        if plot_examples:  # Also display if requested
            plt.show()
        else:
            plt.close()

    # Return results dictionary
    results = {
        "n_samples": len(pred_final),
        "model_params": total_params,
        "input_samples": model.in_samples,
        "input_duration": model.in_samples / model.sampling_rate,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions_final": pred_final,
        "targets_final": target_final,
        "predictions_timeseries": all_predictions,
        "targets_timeseries": all_targets,
        "residuals": pred_final - target_final,
        "avg_inference_time": avg_inference_time,
        "total_inference_time": total_inference_time,
    }

    if all_mags:
        results["original_magnitudes"] = all_mags

    return results


def compare_models(results_dict, output_dir=".", save_plots=True):
    """
    Compare multiple model evaluation results.

    Args:
        results_dict: Dictionary with model names as keys and results as values
        output_dir: Directory to save comparison plots
        save_plots: Whether to save plots to file
    """
    print("\n" + "=" * 60)
    print("MODEL COMPARISON")
    print("=" * 60)

    # Print comparison table
    print(
        f"{'Model':<20} {'RMSE':<8} {'MAE':<8} {'R²':<8} {'Params':<12} {'Time/batch':<12}"
    )
    print("-" * 80)

    for model_name, results in results_dict.items():
        params_str = (
            f"{results.get('model_params', 0):,}"
            if "model_params" in results
            else "N/A"
        )
        time_str = (
            f"{results.get('avg_inference_time', 0):.4f}s"
            if "avg_inference_time" in results
            else "N/A"
        )

        print(
            f"{model_name:<20} {results['rmse']:<8.4f} {results['mae']:<8.4f} "
            f"{results['r2']:<8.4f} {params_str:<12} {time_str:<12}"
        )

    print("=" * 60)

    if save_plots and len(results_dict) > 1:
        # Create comparison plots
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # RMSE comparison
        models = list(results_dict.keys())
        rmses = [results_dict[model]["rmse"] for model in models]
        maes = [results_dict[model]["mae"] for model in models]
        r2s = [results_dict[model]["r2"] for model in models]

        axes[0, 0].bar(models, rmses, alpha=0.7, color="skyblue")
        axes[0, 0].set_ylabel("RMSE")
        axes[0, 0].set_title("Root Mean Squared Error Comparison")
        axes[0, 0].tick_params(axis="x", rotation=45)
        axes[0, 0].grid(True, alpha=0.3)

        # MAE comparison
        axes[0, 1].bar(models, maes, alpha=0.7, color="lightcoral")
        axes[0, 1].set_ylabel("MAE")
        axes[0, 1].set_title("Mean Absolute Error Comparison")
        axes[0, 1].tick_params(axis="x", rotation=45)
        axes[0, 1].grid(True, alpha=0.3)

        # R² comparison
        axes[1, 0].bar(models, r2s, alpha=0.7, color="lightgreen")
        axes[1, 0].set_ylabel("R² Score")
        axes[1, 0].set_title("R² Score Comparison")
        axes[1, 0].tick_params(axis="x", rotation=45)
        axes[1, 0].grid(True, alpha=0.3)

        # Scatter plot comparison
        colors = ["blue", "red", "green", "orange", "purple"]
        for i, (model_name, results) in enumerate(results_dict.items()):
            if i < len(colors):
                axes[1, 1].scatter(
                    results["targets_final"],
                    results["predictions_final"],
                    alpha=0.6,
                    s=10,
                    label=model_name,
                    color=colors[i],
                )

        # Add perfect prediction line
        all_targets = np.concatenate(
            [results["targets_final"] for results in results_dict.values()]
        )
        axes[1, 1].plot(
            [all_targets.min(), all_targets.max()],
            [all_targets.min(), all_targets.max()],
            "k--",
            linewidth=2,
            alpha=0.8,
        )
        axes[1, 1].set_xlabel("True Magnitude")
        axes[1, 1].set_ylabel("Predicted Magnitude")
        axes[1, 1].set_title("Predictions vs True Values")
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        comparison_path = os.path.join(output_dir, f"model_comparison_{timestamp}.png")
        plt.savefig(comparison_path, dpi=300, bbox_inches="tight")
        print(f"Comparison plots saved to: {comparison_path}")
        plt.show()


# Example usage
if __name__ == "__main__":
    # Initialize model
    model = EQTransformerMag(
        in_channels=3,
        in_samples=3001,
        sampling_rate=100,
        lstm_blocks=3,
        drop_rate=0.1,
        norm="std",
    )

    # Load dataset
    data = sbd.ETHZ(sampling_rate=100)

    # Example model path (update this to your actual model path)
    model_path = "src/trained_weights/eqtransformermag_v1_20241022_123456/model_best.pt"

    # Evaluate model
    if os.path.exists(model_path):
        results = evaluate_eqtransformer_mag(
            model=model,
            model_path=model_path,
            data=data,
            batch_size=64,
            plot_examples=True,
            num_examples=5,
        )
    else:
        print(f"Model path {model_path} does not exist. Please train the model first.")
