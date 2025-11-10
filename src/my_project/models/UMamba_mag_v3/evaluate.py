"""
Evaluation script for UMamba V3 magnitude estimation model.

Evaluates ONLY the scalar predictions (1 per waveform) for fair comparison.
Optionally saves temporal predictions for analysis.
"""

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

from my_project.models.UMamba_mag_v3.model import UMambaMag
from my_project.loaders import data_loader as dl


def evaluate_umamba_mag_v3(
    model: WaveformModel,
    model_path: str,
    data: sbd.BenchmarkDataset,
    batch_size: int = 64,
    plot_examples: bool = False,
    num_examples: int = 5,
    output_dir: str = None,
    device: Optional[str] = None,
    save_temporal: bool = False,
    save_uncertainty: bool = False,
):
    """
    Evaluate UMamba V3 model on test set.
    
    IMPORTANT: Evaluation uses ONLY scalar predictions (1 per waveform).
    Temporal predictions are optionally saved for analysis but do NOT affect metrics.
    
    Args:
        model: UMambaMag V3 model instance
        model_path: Path to trained model weights
        data: Dataset to evaluate on
        batch_size: Batch size for evaluation
        plot_examples: Whether to plot example predictions
        num_examples: Number of examples to plot
        output_dir: Directory to save outputs (defaults to model directory)
        device: Device to use for inference
        save_temporal: If True, save temporal predictions for analysis
        save_uncertainty: If True, save uncertainty predictions (requires model.use_uncertainty=True)
    """
    print("\n" + "=" * 60)
    use_uncertainty = hasattr(model, 'use_uncertainty') and model.use_uncertainty
    if use_uncertainty:
        print("EVALUATING UMAMBA V3 (TRIPLE-HEAD ARCHITECTURE)")
    else:
        print("EVALUATING UMAMBA V3 (DUAL-HEAD ARCHITECTURE)")
    print("=" * 60)
    print("NOTE: Metrics computed on SCALAR predictions only (1 per waveform)")
    
    # Load model weights
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
        if "val_loss_scalar" in checkpoint:
            print(f"Checkpoint scalar loss: {checkpoint['val_loss_scalar']:.6f}")
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
        if isinstance(device, int):
            device_used = torch.device(f"cuda:{device}")
        else:
            device_used = torch.device(device)
        model = model.to(device_used)

    print(f"Model is on device: {device_used}")
    print("=" * 60)

    model.eval()

    # Load test data
    print("\nLoading test data...")
    test_generator, test_loader, _ = dl.load_dataset(
        data, model, "test", batch_size=batch_size
    )
    print(f"Test samples: {len(test_generator)}")
    print(f"Test batches: {len(test_loader)}")

    # Evaluation metrics
    scalar_predictions = []
    scalar_targets = []
    
    # Optional: collect temporal predictions for analysis
    if save_temporal or save_uncertainty:
        temporal_predictions = []
        temporal_targets = []
        waveforms = []
    
    # Optional: collect uncertainty predictions
    if save_uncertainty and use_uncertainty:
        uncertainty_predictions = []
    
    inference_times = []

    print("\nRunning inference on test set...")
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

            x = batch["X"].to(device_used)
            y_temporal = batch["magnitude"].to(device_used)  # (batch, samples)
            # True magnitude is the max value (after P-pick it's constant at source_magnitude)
            y_scalar = y_temporal.max(dim=1)[0]  # (batch,) - true scalar target from metadata

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            
            if save_uncertainty and use_uncertainty:
                # Get all outputs (scalar, temporal, uncertainty)
                pred_scalar, pred_temporal, log_var = model(x_preproc, return_all=True)
                uncertainty_predictions.append(log_var.cpu().numpy())
            elif save_temporal:
                # Get both scalar and temporal
                pred_scalar, pred_temporal = model(x_preproc, return_temporal=True)
            else:
                # Get only scalar output (default behavior)
                pred_scalar = model(x_preproc, return_temporal=False)

            if end_event is not None:
                end_event.record()
                torch.cuda.synchronize()
                inference_times.append(start_event.elapsed_time(end_event) / 1000.0)

            # Store SCALAR predictions (primary task)
            scalar_predictions.append(pred_scalar.cpu().numpy())
            scalar_targets.append(y_scalar.cpu().numpy())
            
            # Optionally store temporal predictions (for analysis only)
            if save_temporal or save_uncertainty:
                temporal_predictions.append(pred_temporal.cpu().numpy())
                temporal_targets.append(y_temporal.cpu().numpy())
                waveforms.append(x.cpu().numpy())

    # Concatenate scalar predictions
    pred_final = np.concatenate(scalar_predictions)  # (n_waveforms,)
    target_final = np.concatenate(scalar_targets)    # (n_waveforms,)

    # Compute metrics on SCALAR predictions only
    mse = mean_squared_error(target_final, pred_final)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(target_final, pred_final)
    r2 = r2_score(target_final, pred_final)

    avg_inference_time = np.mean(inference_times) if inference_times else 0
    total_inference_time = np.sum(inference_times) if inference_times else 0

    # Print results
    print("\n" + "=" * 60)
    print("UMAMBA V3 EVALUATION RESULTS (SCALAR PREDICTIONS)")
    print("=" * 60)
    print(f"Number of waveforms: {len(pred_final)}")
    print(f"Model parameters: {total_params:,}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    if inference_times:
        print(f"Average inference time: {avg_inference_time:.4f}s per batch")
        print(f"Total inference time: {total_inference_time:.2f}s")
    print("=" * 60)

    results = {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'n_samples': len(pred_final),
        'model_params': total_params,
        'scalar_predictions': pred_final,
        'scalar_targets': target_final,
    }

    # Add temporal analysis if collected
    if save_temporal or save_uncertainty:
        temporal_pred_all = np.concatenate(temporal_predictions)
        temporal_targ_all = np.concatenate(temporal_targets)
        waveforms_all = np.concatenate(waveforms)
        
        results['temporal_predictions'] = temporal_pred_all
        results['temporal_targets'] = temporal_targ_all
        results['waveforms'] = waveforms_all
        
        # Add uncertainty if collected
        if save_uncertainty and use_uncertainty:
            uncertainty_all = np.concatenate(uncertainty_predictions)
            results['uncertainty'] = uncertainty_all
            
            # Convert log variance to std dev for interpretability
            std_dev = np.exp(uncertainty_all / 2)
            results['std_dev'] = std_dev
            
            print(f"\nUNCERTAINTY HEAD ANALYSIS:")
            print(f"Mean log variance: {uncertainty_all.mean():.4f}")
            print(f"Mean std dev: {std_dev.mean():.4f}")
            print(f"Min/Max std dev: {std_dev.min():.4f} / {std_dev.max():.4f}")
        
        # Compute temporal metrics for analysis (not official evaluation)
        temporal_mse = mean_squared_error(
            temporal_targ_all.flatten(), 
            temporal_pred_all.flatten()
        )
        temporal_mae = mean_absolute_error(
            temporal_targ_all.flatten(), 
            temporal_pred_all.flatten()
        )
        
        results['temporal_mse'] = temporal_mse
        results['temporal_mae'] = temporal_mae
        
        print("\nTEMPORAL HEAD ANALYSIS (auxiliary task - not used for evaluation):")
        print(f"Temporal MSE (flattened): {temporal_mse:.4f}")
        print(f"Temporal MAE (flattened): {temporal_mae:.4f}")
        print("NOTE: These are computed on per-sample predictions and are for analysis only")
        print("=" * 60)

    # Create plots
    if plot_examples or save_temporal or save_uncertainty:
        if output_dir is None:
            output_dir = os.path.dirname(model_path)
        
        os.makedirs(output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Plot 1: Summary plots (scalar predictions)
        plot_scalar_summary(
            pred_final, 
            target_final, 
            mse, 
            rmse, 
            mae, 
            r2, 
            output_dir, 
            timestamp
        )
        
        # Plot 2: Triple-head examples (waveform + temporal + uncertainty)
        if (save_temporal or save_uncertainty) and num_examples > 0:
            if save_uncertainty and use_uncertainty:
                # Plot with all three: waveform, temporal, uncertainty
                plot_triple_head_examples(
                    waveforms_all,
                    temporal_pred_all,
                    temporal_targ_all,
                    uncertainty_all,
                    pred_final,
                    target_final,
                    num_examples,
                    output_dir,
                    timestamp
                )
            else:
                # Plot dual-head (waveform + temporal only)
                plot_temporal_examples(
                    waveforms_all,
                    temporal_pred_all,
                    temporal_targ_all,
                    pred_final,
                    target_final,
                    num_examples,
                    output_dir,
                    timestamp
                )

    return results


def plot_scalar_summary(pred, target, mse, rmse, mae, r2, output_dir, timestamp):
    """Plot summary of scalar predictions (4 subplots)."""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Scatter: predicted vs true
    axes[0, 0].scatter(target, pred, alpha=0.6, s=10)
    axes[0, 0].plot([target.min(), target.max()], [target.min(), target.max()], "r--", lw=2)
    axes[0, 0].set_xlabel("True Magnitude")
    axes[0, 0].set_ylabel("Predicted Magnitude")
    axes[0, 0].set_title(f"UMamba V3: Predicted vs True (R² = {r2:.3f})")
    axes[0, 0].grid(True, alpha=0.3)

    # Residual plot
    residuals = pred - target
    axes[0, 1].scatter(target, residuals, alpha=0.6, s=10)
    axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[0, 1].set_xlabel("True Magnitude")
    axes[0, 1].set_ylabel("Residual (Predicted - True)")
    axes[0, 1].set_title(f"Residual Plot (RMSE = {rmse:.3f})")
    axes[0, 1].grid(True, alpha=0.3)

    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, density=True, edgecolor='black')
    axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
    axes[1, 0].set_xlabel("Residual (Predicted - True)")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title(f"Residual Distribution (MAE = {mae:.3f})")
    axes[1, 0].grid(True, alpha=0.3)

    # Magnitude distribution comparison
    axes[1, 1].hist(target, bins=30, alpha=0.7, label="True", density=True, edgecolor='black')
    axes[1, 1].hist(pred, bins=30, alpha=0.7, label="Predicted", density=True, edgecolor='black')
    axes[1, 1].set_xlabel("Magnitude")
    axes[1, 1].set_ylabel("Density")
    axes[1, 1].set_title("Magnitude Distribution Comparison")
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(output_dir, f"umamba_v3_evaluation_{timestamp}.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"\nSaved scalar evaluation plots to: {save_path}")
    plt.close()


def plot_temporal_examples(waveforms, temporal_pred, temporal_targ, 
                          scalar_pred, scalar_targ, num_examples, 
                          output_dir, timestamp):
    """Plot temporal prediction examples."""
    num_examples = min(num_examples, len(waveforms))
    
    for idx in range(num_examples):
        fig, axes = plt.subplots(4, 1, figsize=(15, 10))
        
        waveform = waveforms[idx]
        temp_pred = temporal_pred[idx]
        temp_targ = temporal_targ[idx]
        scal_pred = scalar_pred[idx]
        scal_targ = scalar_targ[idx]
        
        time = np.arange(len(temp_pred)) / 100  # Assuming 100 Hz
        
        # Plot 3-component waveform
        for i, component in enumerate(['E', 'N', 'Z']):
            axes[0].plot(time, waveform[i], label=component, alpha=0.7)
        axes[0].set_ylabel('Amplitude')
        axes[0].set_title(f'Seismic Waveform (Sample {idx+1})')
        axes[0].legend(loc='upper right')
        axes[0].grid(True, alpha=0.3)
        
        # Plot temporal predictions vs targets
        axes[1].plot(time, temp_targ, 'b-', label='Target', linewidth=2)
        axes[1].plot(time, temp_pred, 'r--', label='Temporal Prediction', linewidth=2)
        axes[1].axhline(y=scal_targ, color='b', linestyle=':', 
                        linewidth=2, label=f'True Magnitude: {scal_targ:.2f}')
        axes[1].axhline(y=scal_pred, color='r', linestyle=':', 
                        linewidth=2, label=f'Predicted Magnitude: {scal_pred:.2f}')
        axes[1].set_ylabel('Magnitude')
        axes[1].set_title('Temporal Predictions vs Target (Auxiliary Head)')
        axes[1].legend(loc='upper right')
        axes[1].grid(True, alpha=0.3)
        
        # Plot temporal prediction error
        error = temp_pred - temp_targ
        axes[2].plot(time, error, 'purple', linewidth=1)
        axes[2].axhline(y=0, color='k', linestyle='--', linewidth=1)
        axes[2].fill_between(time, 0, error, alpha=0.3, color='purple')
        axes[2].set_ylabel('Error')
        axes[2].set_title(f'Temporal Prediction Error (MAE: {np.abs(error).mean():.3f})')
        axes[2].grid(True, alpha=0.3)
        
        # Plot prediction consistency
        axes[3].plot(time, temp_pred, 'r-', label='Temporal Prediction')
        axes[3].fill_between(time, 
                             temp_pred - np.std(temp_pred), 
                             temp_pred + np.std(temp_pred),
                             alpha=0.3, color='red', label=f'Std Dev: {np.std(temp_pred):.3f}')
        axes[3].axhline(y=scal_pred, color='darkred', linestyle='--', 
                        linewidth=2, label='Scalar Prediction (Primary Head)')
        axes[3].set_xlabel('Time (s)')
        axes[3].set_ylabel('Magnitude')
        axes[3].set_title('Temporal Prediction with Uncertainty')
        axes[3].legend(loc='upper right')
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"temporal_example_{idx+1}_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"Saved {num_examples} temporal example plots to: {output_dir}")


def plot_triple_head_examples(waveforms, temporal_pred, temporal_targ, 
                              uncertainty, scalar_pred, scalar_targ, 
                              num_examples, output_dir, timestamp):
    """
    Plot triple-head prediction examples with 3 subplots:
    1. Original 3-phase waveform (all 3 channels on one plot)
    2. Predicted temporal magnitude (per-timestep)
    3. Uncertainty (per-timestep, converted to std dev)
    """
    num_examples = min(num_examples, len(waveforms))
    
    # Convert log variance to std dev for better interpretability
    std_dev = np.exp(uncertainty / 2)
    
    for idx in range(num_examples):
        fig, axes = plt.subplots(3, 1, figsize=(15, 10))
        
        waveform = waveforms[idx]
        temp_pred = temporal_pred[idx]
        temp_targ = temporal_targ[idx]
        std = std_dev[idx]
        scal_pred = scalar_pred[idx]
        scal_targ = scalar_targ[idx]
        
        time = np.arange(len(temp_pred)) / 100  # Assuming 100 Hz
        
        # Subplot 1: 3-component waveform (all on one plot)
        colors = ['blue', 'green', 'red']
        labels = ['E (East)', 'N (North)', 'Z (Vertical)']
        for i in range(3):
            axes[0].plot(time, waveform[i], label=labels[i], alpha=0.8, 
                        linewidth=1.5, color=colors[i])
        axes[0].set_ylabel('Amplitude', fontsize=12, fontweight='bold')
        axes[0].set_title(f'Seismic Waveform - Sample {idx+1}\n'
                         f'True Magnitude: {scal_targ:.2f} | '
                         f'Predicted: {scal_pred:.2f} | '
                         f'Error: {abs(scal_pred - scal_targ):.3f}',
                         fontsize=13, fontweight='bold')
        axes[0].legend(loc='upper right', fontsize=10)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim(time[0], time[-1])
        
        # Subplot 2: Temporal magnitude predictions (per-timestep)
        axes[1].plot(time, temp_targ, 'b-', label='Target (Ground Truth)', 
                    linewidth=2.5, alpha=0.8)
        axes[1].plot(time, temp_pred, 'r-', label='Temporal Prediction', 
                    linewidth=2, alpha=0.9)
        
        # Add horizontal lines for scalar predictions
        axes[1].axhline(y=scal_targ, color='blue', linestyle=':', 
                       linewidth=2, alpha=0.6, label=f'Scalar Target: {scal_targ:.2f}')
        axes[1].axhline(y=scal_pred, color='red', linestyle=':', 
                       linewidth=2, alpha=0.6, label=f'Scalar Prediction: {scal_pred:.2f}')
        
        # Fill area between prediction and target
        axes[1].fill_between(time, temp_pred, temp_targ, alpha=0.2, color='purple',
                            label=f'Error (MAE: {np.abs(temp_pred - temp_targ).mean():.3f})')
        
        axes[1].set_ylabel('Magnitude', fontsize=12, fontweight='bold')
        axes[1].set_title('Temporal Magnitude Predictions (Per-Timestep)', 
                         fontsize=13, fontweight='bold')
        axes[1].legend(loc='upper right', fontsize=9, ncol=2)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim(time[0], time[-1])
        
        # Subplot 3: Uncertainty (per-timestep)
        # Plot std dev as primary metric
        axes[2].plot(time, std, 'purple', linewidth=2, label='Uncertainty (Std Dev)')
        axes[2].fill_between(time, 0, std, alpha=0.4, color='purple')
        
        # Add horizontal line for mean uncertainty
        mean_std = std.mean()
        axes[2].axhline(y=mean_std, color='darkviolet', linestyle='--', 
                       linewidth=2, label=f'Mean Std Dev: {mean_std:.3f}')
        
        # Add reference bands
        axes[2].axhspan(0, 0.1, alpha=0.1, color='green', label='Low Uncertainty')
        axes[2].axhspan(0.1, 0.3, alpha=0.1, color='yellow', label='Medium Uncertainty')
        axes[2].axhspan(0.3, std.max() * 1.1, alpha=0.1, color='red', label='High Uncertainty')
        
        axes[2].set_xlabel('Time (s)', fontsize=12, fontweight='bold')
        axes[2].set_ylabel('Std Dev (Magnitude Units)', fontsize=12, fontweight='bold')
        axes[2].set_title(f'Model Uncertainty (Per-Timestep)\n'
                         f'Min: {std.min():.3f} | Mean: {mean_std:.3f} | Max: {std.max():.3f}',
                         fontsize=13, fontweight='bold')
        axes[2].legend(loc='upper right', fontsize=9, ncol=2)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xlim(time[0], time[-1])
        axes[2].set_ylim(0, std.max() * 1.1)
        
        plt.tight_layout()
        save_path = os.path.join(output_dir, f"triple_head_sample_{idx+1}_{timestamp}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    
    print(f"\nSaved {num_examples} triple-head example plots to: {output_dir}")
