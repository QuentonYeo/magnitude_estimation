"""
Evaluation functions for EQTransformerMag model.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from tqdm import tqdm
import seisbench.data as sbd
from my_project.models.EQTransformer_v2.model import EQTransformerMagV2
from my_project.loaders import data_loader as dl
from my_project.utils.utils import plot_scalar_summary
import os
from datetime import datetime


def evaluate_eqtransformer_mag(
    model: EQTransformerMagV2,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 64,  # Match training batch size
    plot_examples: bool = True,
    num_examples: int = 5,
    output_dir: str = None,
    device: str = None,
):
    """
    Evaluate EQTransformerMagV2 model for scalar magnitude regression.
    
    Updated to follow UMamba V3 evaluation approach:
    - Model outputs scalar magnitude directly
    - Extract scalar target as max of temporal labels
    - Direct comparison of scalar predictions vs targets

    Args:
        model: EQTransformerMagV2 model instance
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

    # Get test data for metadata access
    _, _, test_data = data.train_dev_test()

    # Load test data
    test_generator, test_loader, _ = dl.load_dataset(
        data, model, "test", batch_size=batch_size
    )
    print(f"Test samples: {len(test_generator)}")
    print(f"Test batches: {len(test_loader)}")

    # Collect all predictions and targets
    all_predictions = []  # Scalar predictions
    all_targets = []  # Scalar targets
    all_temporal_labels = []  # For visualization
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

            # Forward pass - model now outputs scalar magnitude
            x = batch["X"].to(device)
            x_preproc = model.annotate_batch_pre(x, {})
            predictions_scalar = model(x_preproc)  # Shape: (batch,)

            if end_time is not None:
                end_time.record()
                torch.cuda.synchronize()
                inference_time = (
                    start_time.elapsed_time(end_time) / 1000.0
                )  # Convert to seconds
                inference_times.append(inference_time)

            # Get temporal labels and extract scalar targets
            y_temporal = batch["magnitude"].to(device)  # Shape: (batch, time_steps)
            y_scalar = y_temporal.max(dim=1)[0]  # Shape: (batch,)

            # Store predictions and targets
            all_predictions.append(predictions_scalar.cpu().numpy())
            all_targets.append(y_scalar.cpu().numpy())
            all_temporal_labels.append(y_temporal.cpu().numpy())

    # Concatenate all batches
    all_predictions = np.concatenate(all_predictions, axis=0)  # Shape: (n_samples,)
    all_targets = np.concatenate(all_targets, axis=0)  # Shape: (n_samples,)
    all_temporal_labels = np.concatenate(all_temporal_labels, axis=0)  # Shape: (n_samples, time_steps)

    # Calculate metrics on scalar predictions
    mse = mean_squared_error(all_targets, all_predictions)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(all_targets, all_predictions)
    r2 = r2_score(all_targets, all_predictions)

    # Calculate additional transformer-specific metrics
    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = np.sum(inference_times) if inference_times else 0

    print("\n" + "=" * 60)
    print("EQTRANSFORMERMAG V2 (SCALAR HEAD) EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of test samples: {len(all_predictions)}")
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

    # Generate standardized scalar evaluation plots (7 individual plots)
    if plot_examples:
        if output_dir is None:
            output_dir = os.path.dirname(model_path)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Call standardized plotting function
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
            model_name="eqtransformermag"
        )

    # Return results dictionary
    results = {
        "n_samples": len(all_predictions),
        "model_params": total_params,
        "input_samples": model.in_samples,
        "input_duration": model.in_samples / model.sampling_rate,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions_scalar": all_predictions,
        "targets_scalar": all_targets,
        "temporal_labels": all_temporal_labels,
        "residuals": all_predictions - all_targets,
        "avg_inference_time": avg_inference_time,
        "total_inference_time": total_inference_time,
    }

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
    model = EQTransformerMagV2(
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
