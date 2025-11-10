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

from my_project.models.UMamba_mag_v2.model import UMambaMag
from my_project.loaders import data_loader as dl


def evaluate_umamba_mag_v2(
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
    Evaluate UMambaMag V2 model on test set.

    Args:
        model: UMambaMag V2 model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
        output_dir: Directory to save outputs (defaults to model directory)
        device: Device to use for inference
    """
    print("\n" + "=" * 60)
    print("EVALUATING UMAMBAMAG V2 (ENCODER-ONLY WITH POOLING)")
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

            # Forward pass - V2 outputs scalar predictions (batch,)
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)  # (batch,)

            if end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event) / 1000.0)

            # Store predictions - already scalar per sample
            pred_magnitudes = y_pred.cpu().numpy()
            true_magnitudes = y_true.cpu().numpy()
            
            # Handle target shape
            if true_magnitudes.ndim > 1:
                # If shape is (batch, samples), take mean
                true_magnitudes = true_magnitudes.mean(axis=1)

            all_predictions.append(pred_magnitudes)
            all_targets.append(true_magnitudes)

    # Concatenate lists into arrays
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    # V2 directly outputs scalar predictions (no need to average over time)
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
    print("UMAMBAMAG V2 EVALUATION RESULTS")
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

    # Plot examples and summary plots
    if output_dir is None:
        output_dir = os.path.dirname(model_path)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_path = os.path.join(output_dir, f"umamba_v2_evaluation_{timestamp}.png")

    # Create summary figure
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))

    # Scatter: predicted vs true
    axes[0, 0].scatter(target_final, pred_final, alpha=0.6, s=10)
    axes[0, 0].plot([target_final.min(), target_final.max()], [target_final.min(), target_final.max()], "r--")
    axes[0, 0].set_xlabel("True Magnitude")
    axes[0, 0].set_ylabel("Predicted Magnitude")
    axes[0, 0].set_title(f"UMamba V2: Predicted vs True (R² = {r2:.3f})")
    axes[0, 0].grid(True, alpha=0.3)

    # Residual plot
    residuals = pred_final - target_final
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
    if plot_examples:
        plt.show()
    else:
        plt.close()

    # Plot example waveforms if requested
    if plot_examples and num_examples > 0:
        plot_prediction_examples(
            model=model,
            test_loader=test_loader,
            device=device_used,
            num_examples=num_examples,
            output_dir=output_dir,
            timestamp=timestamp,
        )

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


def plot_prediction_examples(
    model: WaveformModel,
    test_loader,
    device,
    num_examples: int = 5,
    output_dir: str = None,
    timestamp: str = None,
):
    """
    Plot example waveforms with their magnitude predictions.

    Args:
        model: Trained UMambaMag V2 model
        test_loader: DataLoader for test data
        device: Device to run inference on
        num_examples: Number of examples to plot
        output_dir: Directory to save plot
        timestamp: Timestamp for filename
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

            # Forward pass - V2 outputs scalar predictions
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)

            # Plot examples from this batch
            batch_size = x.shape[0]
            for i in range(min(batch_size, num_examples - examples_plotted)):
                ax = axes[examples_plotted]

                # Get waveform and predictions
                waveform = x[i].cpu().numpy()
                pred_mag = y_pred[i].item()
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
    
    if output_dir and timestamp:
        examples_path = os.path.join(output_dir, f"umamba_v2_examples_{timestamp}.png")
        plt.savefig(examples_path, dpi=300, bbox_inches="tight")
        print(f"Example plots saved to: {examples_path}")
    
    plt.show()


# Example usage
if __name__ == "__main__":

    # Initialize model (V2 - encoder-only with pooling)
    model = UMambaMag(
        in_channels=3,
        sampling_rate=100,
        norm="std",
        n_stages=4,
        features_per_stage=[8, 16, 32, 64],
        kernel_size=7,
        strides=[2, 2, 2, 2],
        n_blocks_per_stage=2,
        pooling_type="avg",
        hidden_dims=[128, 64],
        dropout=0.3,
    )

    # Load dataset
    data = sbd.STEAD(sampling_rate=100)

    # Path to trained model
    model_path = "src/trained_weights/UMambaMag_STEAD_20250128_000000/model_best.pt"

    # Evaluate
    results = evaluate_umamba_mag_v2(
        model=model,
        model_path=model_path,
        data=data,
        batch_size=64,
        plot_examples=True,
        num_examples=5,
    )
