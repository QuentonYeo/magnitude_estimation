import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
from seisbench.models.base import WaveformModel
from typing import Optional
from tqdm import tqdm
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from datetime import datetime

from my_project.models.UMamba_mag.model import UMambaMag
from my_project.loaders import data_loader as dl


def evaluate_umamba_mag(
    model: WaveformModel,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 64,
    plot_examples: bool = False,
    num_examples: int = 5,
    output_dir: str = None,
    device: Optional[str] = None,
):
    """
    Evaluate UMambaMag model on test set.
    
    Evaluation methodology aligned with UMamba v3:
    - Uses .max() for target magnitude extraction (correct after P-arrival)
    - Scalar predictions only (single magnitude per waveform)

    Important:
        The evaluation target uses y_temporal.max(axis=1) instead of .mean()
        because after the P-pick, the magnitude is constant at the source_magnitude
        value. Using .mean() incorrectly averages with pre-P zeros, artificially
        lowering the target and causing poor evaluation metrics.

    Args:
        model: UMambaMag model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
    """
    print("\n" + "=" * 60)
    print("EVALUATING UMAMBAMAG")
    print("=" * 60)

    # Load model weights
    # Set device appropriately
    if device is not None:
        model = model.to(device)
        map_location = device
    else:
        map_location = "cpu"

    checkpoint = torch.load(model_path, map_location=map_location)
    # Support both plain state_dict and wrapped checkpoint
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
    print(f"Loaded model from: {model_path}")
    print(f"Model loaded and set to evaluation mode")

    print(f"Model loaded and set to evaluation mode")

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

    model.eval()

    # Load test data
    print("\nLoading test data...")
    test_generator, test_loader, _ = dl.load_dataset(
        data, model, "test", batch_size=batch_size
    )
    print(f"Test samples: {len(test_generator)}")
    print(f"Test batches: {len(test_loader)}")

    # Evaluation metrics
    all_predictions = []
    all_targets = []
    inference_times = []

    print("Running inference on test set...")
    with torch.no_grad():
        # Use tqdm for progress
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

            x = batch["X"].to(device_used)
            y_true = batch["magnitude"].to(device_used)

            # Forward pass - now outputs scalar (batch,) directly
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)

            if end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event) / 1000.0)

            # Store predictions - already scalar per sample
            pred_magnitudes = y_pred.cpu().numpy()
            true_magnitudes = y_true.cpu().numpy()
            
            # CRITICAL: Use .max() instead of .mean() to get true magnitude
            # After P-pick, magnitude is constant at source_magnitude value
            # Using .mean() incorrectly averages with pre-P zeros, lowering the target
            if true_magnitudes.ndim > 1:
                # If shape is (batch, samples), take max to get true magnitude
                true_magnitudes = true_magnitudes.max(axis=1)

            all_predictions.append(pred_magnitudes)
            all_targets.append(true_magnitudes)

    # Concatenate lists into arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # Use predictions directly (already scalar per sample)
    pred_final = all_predictions
    target_final = all_targets

    # Compute metrics
    mse = mean_squared_error(target_final, pred_final)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_final, pred_final)
    r2 = r2_score(target_final, pred_final)

    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = np.sum(inference_times) if inference_times else 0

    # Print results
    print("\n" + "=" * 60)
    print("UMAMBAMAG EVALUATION RESULTS")
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

    # Plot examples and summary plots similar to EQTransformer
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"umamba_evaluation_{timestamp}.png")

    # Create summary figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Scatter: predicted vs true
    axes[0, 0].scatter(target_final, pred_final, alpha=0.6, s=10)
    axes[0, 0].plot([target_final.min(), target_final.max()], [target_final.min(), target_final.max()], "r--")
    axes[0, 0].set_xlabel("True Magnitude")
    axes[0, 0].set_ylabel("Predicted Magnitude")
    axes[0, 0].set_title(f"UMamba: Predicted vs True (R² = {r2:.3f})")

    # Residual plot
    residuals = pred_final - target_final
    axes[0, 1].scatter(target_final, residuals, alpha=0.6, s=10)
    axes[0, 1].axhline(y=0, color="r", linestyle="--")
    axes[0, 1].set_xlabel("True Magnitude")
    axes[0, 1].set_ylabel("Residual (Predicted - True)")

    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True)
    axes[1, 0].axvline(x=0, color="r", linestyle="--")
    axes[1, 0].set_xlabel("Residual (Predicted - True)")
    axes[1, 0].set_title(f"Residual Distribution (MAE = {mae:.3f})")

    # Magnitude distribution comparison
    axes[1, 1].hist(target_final, bins=30, alpha=0.7, label="True", density=True)
    axes[1, 1].hist(pred_final, bins=30, alpha=0.7, label="Predicted", density=True)
    axes[1, 1].set_xlabel("Magnitude")
    axes[1, 1].legend()

    # Time series examples (if available)
    n_examples_plot = min(num_examples, len(pred_final))
    if n_examples_plot > 0:
        axes[2, 0].set_title("Example Predictions (final values)")
        # Plot a few predicted vs true as points
        for i in range(n_examples_plot):
            axes[2, 0].plot([i], [pred_final[i]], "o", label=f"Pred {i+1}" if i < 3 else None)
            axes[2, 0].plot([i], [target_final[i]], "x", label=f"True {i+1}" if i < 3 else None)
        axes[2, 0].legend()

    # Boxplot of absolute errors
    axes[2, 1].boxplot([np.abs(residuals)])
    axes[2, 1].set_title("Absolute Error Distribution")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    print(f"Evaluation plots saved to: {plot_path}")
    if plot_examples:
        plt.show()
    else:
        plt.close()

    results = {
        "n_samples": len(pred_final),
        "model_params": total_params,
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
        "predictions_final": pred_final,
        "targets_final": target_final,
        "residuals": residuals,
        "avg_inference_time": avg_inference_time,
        "total_inference_time": total_inference_time,
        "plot_path": plot_path,
    }

    return results


def plot_prediction_examples(
    model: WaveformModel,
    test_loader,
    device,
    num_examples: int = 5,
):
    """
    Plot example waveforms with their magnitude predictions.

    Args:
        model: Trained UMambaMag model
        test_loader: DataLoader for test data
        device: Device to run inference on
        num_examples: Number of examples to plot
    """
    model.eval()

    fig, axes = plt.subplots(num_examples, 1, figsize=(15, 3 * num_examples))
    if num_examples == 1:
        axes = [axes]

    examples_plotted = 0

    with torch.no_grad():
        for batch in test_loader:
            if examples_plotted >= num_examples:
                break

            x = batch["X"].to(device)
            y_true = batch["magnitude"].to(device)

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)

            # Plot examples from this batch
            batch_size = x.shape[0]
            for i in range(min(batch_size, num_examples - examples_plotted)):
                ax = axes[examples_plotted]

                # Get waveform and predictions
                waveform = x[i].cpu().numpy()
                pred_mag = y_pred[i].mean().item()
                true_mag = y_true[i].item() if y_true[i].dim() == 0 else y_true[i, 0].item()

                # Plot 3-component waveform
                time = np.arange(waveform.shape[1]) / 100.0  # Assuming 100 Hz
                for comp_idx, comp_name in enumerate(["Z", "N", "E"]):
                    ax.plot(
                        time,
                        waveform[comp_idx] + comp_idx * 2,
                        label=f"{comp_name} component",
                        alpha=0.7,
                    )

                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Normalized Amplitude")
                ax.set_title(
                    f"Example {examples_plotted + 1}: True Mag={true_mag:.2f}, "
                    f"Pred Mag={pred_mag:.2f}, Error={pred_mag - true_mag:.2f}"
                )
                ax.legend(loc="upper right")
                ax.grid(True, alpha=0.3)

                examples_plotted += 1
                if examples_plotted >= num_examples:
                    break

    plt.tight_layout()
    plt.show()


# Example usage
if __name__ == "__main__":

    # Initialize model
    model = UMambaMag(
        in_channels=3,
        sampling_rate=100,
        norm="std",
        n_stages=4,
        features_per_stage=[8, 16, 32, 64],
        kernel_size=7,
        strides=[2, 2, 2, 2],
        n_blocks_per_stage=2,
        n_conv_per_stage_decoder=2,
        deep_supervision=False,
    )

    # Load dataset
    data = sbd.ETHZ(sampling_rate=100)

    # Path to trained model
    model_path = "src/trained_weights/umambamag_v1_20250127_120000/model_best.pt"

    # Evaluate
    results = evaluate_umamba_mag(
        model=model,
        model_path=model_path,
        data=data,
        batch_size=64,
        plot_examples=True,
        num_examples=5,
    )
