"""
Training script for UMamba V3 magnitude estimation model.

Implements triple-head training with:
- Scalar head: Global magnitude prediction (primary task)
- Temporal head: Per-timestep magnitude prediction (auxiliary task)
- Uncertainty head: Automatic sample weighting via learned uncertainty
- Loss: Kendall & Gal 2017 uncertainty-weighted MSE with configurable weights

Important Notes:
    The uncertainty head outputs log_var (log variance) which is used to weight
    the loss via precision = exp(-log_var). To prevent numerical instability:
    - log_var is clamped to [-3, 3] in the model's forward pass
    - This keeps precision in [exp(-3), exp(3)] ≈ [0.05, 20]
    - The uncertainty head is initialized with zero bias (log_var ≈ 0 initially)
    
    Without clamping, log_var can decrease indefinitely during training, causing
    precision to explode and the loss to become unstable. The tighter [-3, 3]
    range (vs the original [-5, 3]) prevents immediate saturation at the bounds.

Training History Format:
    Unlike other models (PhaseNet, EQTransformer, UMamba V1/V2) which save simple
    history with keys like 'train_losses' and 'val_losses', this model saves
    detailed metrics for each head:
    
    Keys in history dict:
        - train_loss, val_loss: Combined weighted loss
        - train_loss_scalar, val_loss_scalar: Scalar head loss (global magnitude)
        - train_loss_temporal, val_loss_temporal: Temporal head loss (per-timestep)
        - train_uncertainty, val_uncertainty: Log variance (if use_uncertainty=True)
        - learning_rates: Learning rate per epoch
    
    To plot training history:
        from my_project.utils.utils import plot_training_history
        
        # Simple view (compatible with all models):
        plot_training_history("path/to/training_history.pt")
        
        # Detailed view (shows scalar/temporal/uncertainty breakdown):
        plot_training_history("path/to/training_history.pt", detailed_metrics=True)
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

import seisbench.data as sbd
from my_project.models.UMamba_mag_v3.model import UMambaMag
from my_project.loaders import data_loader as dl


def train_umamba_mag_v3(
    model_name: str,
    model: UMambaMag,
    data: sbd.BenchmarkDataset,
    learning_rate: float = 1e-3,
    epochs: int = 100,
    batch_size: int = 64,
    optimizer_name: str = "AdamW",
    weight_decay: float = 1e-5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    save_every: int = 5,
    gradient_clip: float = 1.0,
    early_stopping_patience: int = 15,
    warmup_epochs: int = 5,
    scalar_weight: float = 0.7,
    temporal_weight: float = 0.25,
    quiet: bool = False,
):
    """
    Train UMamba V3 model with triple-head architecture.
    
    Architecture:
        - Scalar head: Global magnitude (primary task)
        - Temporal head: Per-timestep magnitude (auxiliary task)
        - Uncertainty head: Learned sample weighting (if model.use_uncertainty=True)
    
    Loss:
        If uncertainty head enabled:
            L = w_s * (0.5 * precision * MSE_scalar + 0.5 * log_var) + 
                w_t * (0.5 * precision * MSE_temporal + 0.5 * log_var)
            where precision = exp(-log_var)
            
            The 0.5 factor is from Kendall & Gal 2017:
            - Prevents negative loss when log_var < 0
            - Balances error term and regularization term
        
        If uncertainty head disabled:
            L = w_s * MSE_scalar + w_t * MSE_temporal
    
    Args:
        model_name: Name for saving checkpoints
        model: UMambaMag V3 model instance
        data: Training dataset
        learning_rate: Initial learning rate
        epochs: Number of training epochs
        batch_size: Batch size
        optimizer_name: Optimizer type ('Adam', 'AdamW', or 'SGD')
        weight_decay: Weight decay for optimizer
        scheduler_patience: Patience for ReduceLROnPlateau
        scheduler_factor: Factor for learning rate reduction
        save_every: Save checkpoint every N epochs
        gradient_clip: Gradient clipping value
        early_stopping_patience: Stop if no improvement for N epochs
        warmup_epochs: Number of warmup epochs for learning rate
        scalar_weight: Weight for scalar loss (recommended: 0.6-0.8)
        temporal_weight: Weight for temporal loss (recommended: 0.2-0.4)
        quiet: If True, disable tqdm progress bars
    
    Returns:
        dict: Training history
    
    Note:
        Loss weights should sum to ≤ 1.0. Remaining weight (1 - scalar - temporal)
        is implicitly used by the uncertainty head's log_var regularization term.
    """
    print("\n" + "=" * 60)
    use_uncertainty = model.use_uncertainty if hasattr(model, 'use_uncertainty') else False
    if use_uncertainty:
        print("TRAINING UMAMBA V3 (TRIPLE-HEAD ARCHITECTURE)")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Scalar weight: {scalar_weight} (primary task)")
        print(f"Temporal weight: {temporal_weight} (auxiliary task)")
        print(f"Uncertainty: ENABLED (automatic sample weighting)")
    else:
        print("TRAINING UMAMBA V3 (DUAL-HEAD ARCHITECTURE)")
        print("=" * 60)
        print(f"Model: {model_name}")
        print(f"Scalar weight: {scalar_weight} (primary task)")
        print(f"Temporal weight: {temporal_weight} (auxiliary task)")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Weight decay: {weight_decay}")
    print(f"Gradient clip: {gradient_clip}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    print("=" * 60)
    
    # Create save directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"src/trained_weights/{model_name}_{timestamp}"
    os.makedirs(save_dir, exist_ok=True)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Load data
    train_generator, train_loader, _ = dl.load_dataset(
        data, model, "train", batch_size=batch_size
    )
    dev_generator, dev_loader, _ = dl.load_dataset(
        data, model, "dev", batch_size=batch_size
    )
    
    print(f"Train samples: {len(train_generator)}")
    print(f"Dev samples: {len(dev_generator)}")
    print("=" * 60 + "\n")
    
    # Setup optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay, momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")
    
    # Setup learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=scheduler_factor, patience=scheduler_patience, verbose=True
    )
    
    # Training history
    history = {
        "train_loss": [],
        "train_loss_scalar": [],
        "train_loss_temporal": [],
        "val_loss": [],
        "val_loss_scalar": [],
        "val_loss_temporal": [],
        "learning_rates": [],
    }
    
    if use_uncertainty:
        history["train_uncertainty"] = []
        history["val_uncertainty"] = []
    
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    # Track training time
    import time
    training_start_time = time.time()
    
    # Training loop
    for epoch in range(1, epochs + 1):
        # Print epoch header
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*60}")
        
        # Warmup learning rate
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor
        
        current_lr = optimizer.param_groups[0]['lr']
        history["learning_rates"].append(current_lr)
        
        # Train
        model.train()
        train_losses = {'total': [], 'scalar': [], 'temporal': []}
        if use_uncertainty:
            train_losses['uncertainty'] = []
            train_losses['max_precision'] = []
            train_losses['min_log_var'] = []
            epoch_warned = False  # Track if we've warned this epoch
        
        pbar = tqdm(train_loader, desc="Training", disable=quiet, leave=False)
        for batch in pbar:
            x = batch["X"].to(model.device)
            y_temporal = batch["magnitude"].to(model.device)  # (batch, samples)
            # True magnitude is the max value (after P-pick it's constant at source_magnitude)
            y_scalar = y_temporal.max(dim=1)[0]  # (batch,)
            
            optimizer.zero_grad()
            
            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            
            if use_uncertainty:
                # Triple-head: returns (scalar, temporal, log_var)
                pred_scalar, pred_temporal, log_var = model(x_preproc, return_all=True)
                
                # Kendall & Gal 2017 uncertainty-weighted loss
                # Correct formulation: L = 0.5 * precision * error + 0.5 * log_var
                # For multi-task: L = sum_i [ w_i * (0.5 * precision * error_i + 0.5 * log_var) ]
                precision = torch.exp(-log_var)  # (batch,)
                
                # Track statistics for epoch summary
                train_losses['max_precision'].append(precision.max().item())
                train_losses['min_log_var'].append(log_var.min().item())
                
                # Monitor for numerical instability (once per epoch)
                if not epoch_warned and precision.max() > 15:
                    print(f"\n⚠ Warning: High precision detected (max={precision.max().item():.2f}). "
                          f"log_var range: [{log_var.min().item():.3f}, {log_var.max().item():.3f}]")
                    epoch_warned = True
                
                # Scalar loss with uncertainty weighting
                # L_scalar = 0.5 * precision * MSE + 0.5 * log_var
                scalar_error = (pred_scalar - y_scalar) ** 2  # (batch,)
                loss_scalar = (0.5 * precision * scalar_error + 0.5 * log_var).mean()
                
                # Temporal loss with uncertainty weighting
                # L_temporal = 0.5 * precision * MSE + 0.5 * log_var
                temporal_error = ((pred_temporal - y_temporal) ** 2).mean(dim=1)  # (batch,)
                loss_temporal = (0.5 * precision * temporal_error + 0.5 * log_var).mean()
                
                # Combined loss with task weighting
                loss = scalar_weight * loss_scalar + temporal_weight * loss_temporal
                
                # Record raw losses (without uncertainty weighting) for monitoring
                train_losses['scalar'].append(scalar_error.mean().item())
                train_losses['temporal'].append(temporal_error.mean().item())
                train_losses['uncertainty'].append(log_var.mean().item())
            else:
                # Dual-head: returns (scalar, temporal)
                pred_scalar, pred_temporal = model(x_preproc)
                
                # Standard MSE losses
                loss_scalar = F.mse_loss(pred_scalar, y_scalar)
                loss_temporal = F.mse_loss(pred_temporal, y_temporal)
                
                # Combined loss with weighting
                loss = scalar_weight * loss_scalar + temporal_weight * loss_temporal
                
                train_losses['scalar'].append(loss_scalar.item())
                train_losses['temporal'].append(loss_temporal.item())
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Record total loss
            train_losses['total'].append(loss.item())
            
            # Update progress bar
            postfix = {
                'loss': f"{loss.item():.4f}",
                'scalar': f"{train_losses['scalar'][-1]:.4f}",
                'temporal': f"{train_losses['temporal'][-1]:.4f}",
            }
            if use_uncertainty:
                postfix['log_var'] = f"{train_losses['uncertainty'][-1]:.3f}"
            pbar.set_postfix(postfix)
        
        # Average training losses
        train_loss = np.mean(train_losses['total'])
        train_loss_scalar = np.mean(train_losses['scalar'])
        train_loss_temporal = np.mean(train_losses['temporal'])
        
        history["train_loss"].append(train_loss)
        history["train_loss_scalar"].append(train_loss_scalar)
        history["train_loss_temporal"].append(train_loss_temporal)
        
        if use_uncertainty:
            train_uncertainty = np.mean(train_losses['uncertainty'])
            history["train_uncertainty"].append(train_uncertainty)
        
        # Validation
        model.eval()
        val_losses = {'total': [], 'scalar': [], 'temporal': []}
        if use_uncertainty:
            val_losses['uncertainty'] = []
        
        with torch.no_grad():
            pbar = tqdm(dev_loader, desc="Validation", disable=quiet, leave=False)
            for batch in pbar:
                x = batch["X"].to(model.device)
                y_temporal = batch["magnitude"].to(model.device)
                # True magnitude is the max value (after P-pick it's constant at source_magnitude)
                y_scalar = y_temporal.max(dim=1)[0]
                
                x_preproc = model.annotate_batch_pre(x, {})
                
                if use_uncertainty:
                    # Triple-head validation
                    pred_scalar, pred_temporal, log_var = model(x_preproc, return_all=True)
                    
                    # Uncertainty-weighted loss (same as training)
                    precision = torch.exp(-log_var)
                    scalar_error = (pred_scalar - y_scalar) ** 2
                    loss_scalar = (0.5 * precision * scalar_error + 0.5 * log_var).mean()
                    temporal_error = ((pred_temporal - y_temporal) ** 2).mean(dim=1)
                    loss_temporal = (0.5 * precision * temporal_error + 0.5 * log_var).mean()
                    loss = scalar_weight * loss_scalar + temporal_weight * loss_temporal
                    
                    # Record raw losses
                    val_losses['scalar'].append(scalar_error.mean().item())
                    val_losses['temporal'].append(temporal_error.mean().item())
                    val_losses['uncertainty'].append(log_var.mean().item())
                else:
                    # Dual-head validation
                    pred_scalar, pred_temporal = model(x_preproc, return_temporal=True)
                    
                    loss_scalar = F.mse_loss(pred_scalar, y_scalar)
                    loss_temporal = F.mse_loss(pred_temporal, y_temporal)
                    loss = scalar_weight * loss_scalar + temporal_weight * loss_temporal
                    
                    val_losses['scalar'].append(loss_scalar.item())
                    val_losses['temporal'].append(loss_temporal.item())
                
                val_losses['total'].append(loss.item())
                
                # Update progress bar
                postfix = {
                    'loss': f"{loss.item():.4f}",
                    'scalar': f"{val_losses['scalar'][-1]:.4f}",
                    'temporal': f"{val_losses['temporal'][-1]:.4f}",
                }
                if use_uncertainty:
                    postfix['log_var'] = f"{val_losses['uncertainty'][-1]:.3f}"
                pbar.set_postfix(postfix)
        
        val_loss = np.mean(val_losses['total'])
        val_loss_scalar = np.mean(val_losses['scalar'])
        val_loss_temporal = np.mean(val_losses['temporal'])
        
        history["val_loss"].append(val_loss)
        history["val_loss_scalar"].append(val_loss_scalar)
        history["val_loss_temporal"].append(val_loss_temporal)
        
        if use_uncertainty:
            val_uncertainty = np.mean(val_losses['uncertainty'])
            history["val_uncertainty"].append(val_uncertainty)
        
        # Compute metrics for logging
        train_rmse_scalar = np.sqrt(train_loss_scalar)
        val_rmse_scalar = np.sqrt(val_loss_scalar)
        train_rmse_temporal = np.sqrt(train_loss_temporal)
        val_rmse_temporal = np.sqrt(val_loss_temporal)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} Summary")
        print(f"{'='*60}")
        if use_uncertainty:
            max_precision_train = np.max(train_losses['max_precision'])
            min_log_var_train = np.min(train_losses['min_log_var'])
            print(f"Training   -> Loss: {train_loss:.6f} | Scalar: {train_loss_scalar:.6f} (RMSE: {train_rmse_scalar:.4f}) | Temporal: {train_loss_temporal:.6f} (RMSE: {train_rmse_temporal:.4f}) | log_var: {train_uncertainty:.3f}")
            print(f"               Uncertainty: log_var_mean={train_uncertainty:.3f}, log_var_min={min_log_var_train:.3f}, precision_max={max_precision_train:.2f}")
            print(f"Validation -> Loss: {val_loss:.6f} | Scalar: {val_loss_scalar:.6f} (RMSE: {val_rmse_scalar:.4f}) | Temporal: {val_loss_temporal:.6f} (RMSE: {val_rmse_temporal:.4f}) | log_var: {val_uncertainty:.3f}")
        else:
            print(f"Training   -> Loss: {train_loss:.6f} | Scalar: {train_loss_scalar:.6f} (RMSE: {train_rmse_scalar:.4f}) | Temporal: {train_loss_temporal:.6f} (RMSE: {train_rmse_temporal:.4f})")
            print(f"Validation -> Loss: {val_loss:.6f} | Scalar: {val_loss_scalar:.6f} (RMSE: {val_rmse_scalar:.4f}) | Temporal: {val_loss_temporal:.6f} (RMSE: {val_rmse_temporal:.4f})")
        print(f"{'='*60}")
        
        # Learning rate scheduling (based on total validation loss)
        if epoch > warmup_epochs:
            scheduler.step(val_loss)
        
        # Check for improvement and save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(f"✓ New best model saved! Val Loss: {val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        if epoch % save_every == 0 or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_loss_scalar": val_loss_scalar,
                "val_loss_temporal": val_loss_temporal,
                "best_val_loss": best_val_loss,
                "history": history,
            }
            
            if is_best:
                save_path = os.path.join(save_dir, "best_model.pt")
                torch.save(checkpoint, save_path)
            
            if epoch % save_every == 0:
                save_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch}.pt")
                torch.save(checkpoint, save_path)
        
        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"\nEarly stopping triggered after {early_stopping_patience} epochs without improvement")
            print(f"Best validation loss: {best_val_loss:.6f}")
            break
    
    # Save final model and history
    final_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "best_val_loss": best_val_loss,
        "history": history,
    }
    
    final_path = os.path.join(save_dir, "final_model.pt")
    torch.save(final_checkpoint, final_path)
    
    history_path = os.path.join(save_dir, "training_history.pt")
    torch.save(history, history_path)
    
    # Calculate and display total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation RMSE: {best_val_loss**0.5:.6f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Models saved to: {save_dir}")
    print("=" * 60 + "\n")
    
    return history
