import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seisbench.data as sbd
from seisbench.models.base import WaveformModel
from typing import Optional

from my_project.models.UMamba_mag.model import UMambaMag
from my_project.loaders import data_loader as dl


def evaluate_umamba_mag(
    model: WaveformModel,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 64,
    plot_examples: bool = False,
    num_examples: int = 5,
):
    """
    Evaluate UMambaMag model on test set.

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
    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    print(f"Loaded model from: {model_path}")
    print(f"Epoch: {checkpoint.get('epoch', 'unknown')}")
    print(f"Val Loss: {checkpoint.get('val_loss', 'N/A')}")

    # Move model to device
    model.to("cuda:1")
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    model.eval()

    # Load test data
    print("\nLoading test data...")
    test_generator, test_loader, _ = dl.load_dataset(
        data, model, "test", batch_size=batch_size
    )
    print(f"Test batches: {len(test_loader)}")

    # Evaluation metrics
    all_predictions = []
    all_targets = []
    all_errors = []

    print("\nEvaluating...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(test_loader):
            x = batch["X"].to(device)
            y_true = batch["magnitude"].to(device)

            # Handle different target shapes
            if y_true.dim() == 1:
                y_true_expanded = y_true.unsqueeze(1).expand(-1, x.shape[-1])
            elif y_true.dim() == 2 and y_true.shape[1] == 1:
                y_true_expanded = y_true.expand(-1, x.shape[-1])
            else:
                y_true_expanded = y_true

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)

            # Store predictions (use mean across time for final magnitude)
            pred_magnitudes = y_pred.mean(dim=1).cpu().numpy()
            true_magnitudes = y_true.cpu().numpy()
            if true_magnitudes.ndim > 1:
                true_magnitudes = true_magnitudes[:, 0]

            all_predictions.extend(pred_magnitudes)
            all_targets.extend(true_magnitudes)
            all_errors.extend(pred_magnitudes - true_magnitudes)

            if batch_idx % 10 == 0:
                print(f"Processed batch {batch_idx}/{len(test_loader)}")

    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_errors = np.array(all_errors)

    # Calculate metrics
    mae = np.abs(all_errors).mean()
    rmse = np.sqrt(np.mean(all_errors**2))
    bias = np.mean(all_errors)
    std_error = np.std(all_errors)

    # Calculate correlation
    correlation = np.corrcoef(all_targets, all_predictions)[0, 1]

    # Print results
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    print(f"Number of samples: {len(all_targets)}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Bias: {bias:.4f}")
    print(f"Standard Deviation of Error: {std_error:.4f}")
    print(f"Correlation: {correlation:.4f}")
    print(f"R²: {correlation**2:.4f}")
    print("=" * 60)

    # Calculate magnitude range statistics
    print("\nMagnitude Range Statistics:")
    for mag_min in range(int(np.floor(all_targets.min())), int(np.ceil(all_targets.max()))):
        mag_max = mag_min + 1
        mask = (all_targets >= mag_min) & (all_targets < mag_max)
        if mask.sum() > 0:
            range_mae = np.abs(all_errors[mask]).mean()
            range_rmse = np.sqrt(np.mean(all_errors[mask]**2))
            print(f"  M{mag_min}-{mag_max}: MAE={range_mae:.4f}, RMSE={range_rmse:.4f}, N={mask.sum()}")

    # Plot results
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    # 1. Scatter plot: Predicted vs True
    ax = axes[0, 0]
    ax.scatter(all_targets, all_predictions, alpha=0.5, s=10)
    ax.plot(
        [all_targets.min(), all_targets.max()],
        [all_targets.min(), all_targets.max()],
        "r--",
        label="Perfect prediction",
    )
    ax.set_xlabel("True Magnitude")
    ax.set_ylabel("Predicted Magnitude")
    ax.set_title(f"Predicted vs True Magnitude\nR²={correlation**2:.3f}, MAE={mae:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. Error distribution
    ax = axes[0, 1]
    ax.hist(all_errors, bins=50, edgecolor="black", alpha=0.7)
    ax.axvline(0, color="r", linestyle="--", label="Zero error")
    ax.axvline(bias, color="g", linestyle="--", label=f"Bias={bias:.3f}")
    ax.set_xlabel("Prediction Error (Predicted - True)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Error Distribution\nRMSE={rmse:.3f}, Std={std_error:.3f}")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Error vs True Magnitude
    ax = axes[1, 0]
    ax.scatter(all_targets, all_errors, alpha=0.5, s=10)
    ax.axhline(0, color="r", linestyle="--", label="Zero error")
    ax.axhline(bias, color="g", linestyle="--", label=f"Bias={bias:.3f}")
    ax.set_xlabel("True Magnitude")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Prediction Error vs True Magnitude")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Residual plot
    ax = axes[1, 1]
    ax.scatter(all_predictions, all_errors, alpha=0.5, s=10)
    ax.axhline(0, color="r", linestyle="--", label="Zero error")
    ax.set_xlabel("Predicted Magnitude")
    ax.set_ylabel("Prediction Error")
    ax.set_title("Residual Plot")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save plot if model_path is provided
    if model_path:
        save_dir = os.path.dirname(model_path)
        plot_path = os.path.join(save_dir, "evaluation_results.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        print(f"\nEvaluation plot saved to: {plot_path}")

    # Plot example predictions
    if plot_examples:
        plot_prediction_examples(
            model, test_loader, device, num_examples=num_examples
        )

    plt.show()

    results = {
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "std_error": std_error,
        "correlation": correlation,
        "r2": correlation**2,
        "predictions": all_predictions,
        "targets": all_targets,
        "errors": all_errors,
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
