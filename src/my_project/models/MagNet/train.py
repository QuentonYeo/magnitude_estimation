"""
Training script for MagNet magnitude estimation model.

Simple BiLSTM-based architecture with scalar magnitude prediction.
Loss: MSE on scalar magnitude (max of temporal labels)

Training is simpler than UMamba V3:
- Single head: Scalar magnitude prediction only
- Single loss: MSE on true magnitude
- No auxiliary tasks or uncertainty estimation

Training History Format:
    Keys in history dict:
        - train_loss: Training MSE loss
        - val_loss: Validation MSE loss
        - train_rmse: Training RMSE (sqrt of MSE)
        - val_rmse: Validation RMSE
        - learning_rates: Learning rate per epoch
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from datetime import datetime

import seisbench.data as sbd
from my_project.models.MagNet.model import MagNet
from my_project.loaders import data_loader as dl


def train_magnet(
    model_name: str,
    model: MagNet,
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
    quiet: bool = False,
):
    """
    Train MagNet model with scalar magnitude prediction.
    
    Architecture:
        Conv(64, k=3) → Dropout → MaxPool(4) →
        Conv(32, k=3) → Dropout → MaxPool(4) →
        BiLSTM(100) → FC(1)
    
    Loss:
        MSE on scalar magnitude (max of temporal labels)
    
    Args:
        model_name: Name for saving checkpoints
        model: MagNet model instance
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
        quiet: If True, disable tqdm progress bars
    
    Returns:
        dict: Training history
    """
    print("\n" + "=" * 60)
    print("TRAINING MAGNET (BiLSTM MAGNITUDE ESTIMATION)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Architecture: Conv64 → Conv32 → BiLSTM100 → FC1")
    print(f"Loss: MSE on scalar magnitude")
    print(f"Learning rate: {learning_rate}")
    print(f"Batch size: {batch_size}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Weight decay: {weight_decay}")
    print(f"Dropout: {model.dropout_rate}")
    print(f"LSTM hidden: {model.lstm_hidden}")
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
        "val_loss": [],
        "train_rmse": [],
        "val_rmse": [],
        "learning_rates": [],
    }
    
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
        train_losses = []
        
        pbar = tqdm(train_loader, desc="Training", disable=quiet, leave=False)
        for batch in pbar:
            x = batch["X"].to(model.device)
            y_temporal = batch["magnitude"].to(model.device)  # (batch, samples)
            
            # True magnitude is the max value (after P-pick it's constant at source_magnitude)
            y_scalar = y_temporal.max(dim=1)[0]  # (batch,)
            
            optimizer.zero_grad()
            
            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            pred_scalar = model(x_preproc).squeeze(-1)  # (batch,) - remove last dim
            
            # MSE loss
            loss = F.mse_loss(pred_scalar, y_scalar)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
            
            optimizer.step()
            
            # Record loss
            train_losses.append(loss.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'rmse': f"{np.sqrt(loss.item()):.4f}",
            })
        
        # Average training loss
        train_loss = np.mean(train_losses)
        train_rmse = np.sqrt(train_loss)
        
        history["train_loss"].append(train_loss)
        history["train_rmse"].append(train_rmse)
        
        # Validation
        model.eval()
        val_losses = []
        
        with torch.no_grad():
            pbar = tqdm(dev_loader, desc="Validation", disable=quiet, leave=False)
            for batch in pbar:
                x = batch["X"].to(model.device)
                y_temporal = batch["magnitude"].to(model.device)
                
                # True magnitude is the max value
                y_scalar = y_temporal.max(dim=1)[0]
                
                # Forward pass
                x_preproc = model.annotate_batch_pre(x, {})
                pred_scalar = model(x_preproc).squeeze(-1)
                
                # MSE loss
                loss = F.mse_loss(pred_scalar, y_scalar)
                val_losses.append(loss.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{loss.item():.4f}",
                    'rmse': f"{np.sqrt(loss.item()):.4f}",
                })
        
        val_loss = np.mean(val_losses)
        val_rmse = np.sqrt(val_loss)
        
        history["val_loss"].append(val_loss)
        history["val_rmse"].append(val_rmse)
        
        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} Summary")
        print(f"{'='*60}")
        print(f"Training   -> Loss: {train_loss:.6f} | RMSE: {train_rmse:.4f}")
        print(f"Validation -> Loss: {val_loss:.6f} | RMSE: {val_rmse:.4f}")
        print(f"{'='*60}")
        
        # Learning rate scheduling
        if epoch > warmup_epochs:
            scheduler.step(val_loss)
        
        # Check for improvement and save best model
        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            print(f"✓ New best model saved! Val Loss: {val_loss:.6f} | Val RMSE: {val_rmse:.4f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save checkpoints
        if epoch % save_every == 0 or is_best:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "val_loss": val_loss,
                "train_rmse": train_rmse,
                "val_rmse": val_rmse,
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
            print(f"Best validation RMSE: {np.sqrt(best_val_loss):.4f}")
            break
    
    # Save final model and history
    final_checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
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
    print(f"Best validation RMSE: {np.sqrt(best_val_loss):.4f}")
    print(f"Final train loss: {history['train_loss'][-1]:.6f}")
    print(f"Final train RMSE: {history['train_rmse'][-1]:.4f}")
    print(f"Final val loss: {history['val_loss'][-1]:.6f}")
    print(f"Final val RMSE: {history['val_rmse'][-1]:.4f}")
    print(f"Total Training Time: {hours:02d}:{minutes:02d}:{seconds:02d}")
    print(f"Models saved to: {save_dir}")
    print("=" * 60 + "\n")
    
    return history
