import os
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import seisbench.data as sbd
from seisbench.models.base import WaveformModel
from tqdm import tqdm

from my_project.models.EQTransformer_v2.model import EQTransformerMagV2
from my_project.loaders import data_loader as dl


def train_eqtransformer_mag(
    model_name: str,
    model: WaveformModel,
    data: sbd.BenchmarkDataset,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,  # Match UMamba V3 batch size
    optimizer_name: str = "AdamW",
    weight_decay: float = 1e-5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    save_every: int = 5,
    gradient_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 15,
    warmup_epochs: int = 5,  # Transformer benefits from warmup
    quiet: bool = False,
    checkpoint_path: Optional[str] = None,  # Path to checkpoint to resume from
):
    """
    Train EQTransformerMagV2 model for scalar magnitude regression.
    
    Updated to follow UMamba V3 training approach:
    - Uses temporal magnitude labels (label[0:onset]=0, label[onset:]=magnitude)
    - Extracts scalar target as max of temporal labels
    - Single scalar head prediction vs scalar target
    - MSE loss on scalar output

    Args:
        model_name: Name for saving model checkpoints
        model: EQTransformerMagV2 model instance
        data: Dataset to train on
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Batch size for training
        optimizer_name: Optimizer type ("Adam", "AdamW", or "SGD")
        weight_decay: Weight decay for optimizer
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor to reduce learning rate by
        save_every: Save model every N epochs
        gradient_clip: Maximum gradient norm (None to disable)
        early_stopping_patience: Epochs to wait before early stopping
        warmup_epochs: Number of epochs for learning rate warmup
        quiet: If True, disable tqdm progress bars
        checkpoint_path: Path to checkpoint file to resume training from
    """
    print("\n" + "=" * 60)
    print("TRAINING EQTRANSFORMERMAG V2 (SCALAR HEAD)")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {data.__class__.__name__}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Early stopping patience: {early_stopping_patience}")
    if checkpoint_path:
        print(f"Resuming from checkpoint: {checkpoint_path}")
    print("=" * 60)

    # Get the device where the model is already placed
    device = next(model.parameters()).device
    print(f"Model is on device: {device}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Load data
    print(f"\nLoading data...")
    train_generator, train_loader, _ = dl.load_dataset(
        data, model, "train", batch_size=batch_size
    )
    dev_generator, dev_loader, _ = dl.load_dataset(
        data, model, "dev", batch_size=batch_size
    )

    print(f"Train batches: {len(train_loader)}")
    print(f"Dev batches: {len(dev_loader)}")

    # Setup optimizer
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    elif optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            momentum=0.9,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Loss function - MSE for scalar regression
    criterion = nn.MSELoss()

    # Setup learning rate scheduler (ReduceLROnPlateau only, warmup handled manually)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
    )

    # Load checkpoint if provided
    start_epoch = 1
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    learning_rates = []
    epochs_without_improvement = 0
    save_dir = None
    
    if checkpoint_path:
        print(f"\nLoading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        
        # Load model state
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✓ Model state loaded from epoch {checkpoint.get('epoch', 'unknown')}")
        
        # Load optimizer state
        if "optimizer_state_dict" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print("✓ Optimizer state loaded")
        
        # Load scheduler state if available
        if "scheduler_state_dict" in checkpoint:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            print("✓ Scheduler state loaded")
        
        # Resume training parameters
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", checkpoint.get("val_loss", float("inf")))
        
        # Load training history if available
        if "train_losses" in checkpoint:
            train_losses = checkpoint["train_losses"]
            val_losses = checkpoint["val_losses"]
            train_maes = checkpoint.get("train_maes", [])
            val_maes = checkpoint.get("val_maes", [])
            learning_rates = checkpoint.get("learning_rates", [])
            print(f"✓ Training history loaded ({len(train_losses)} epochs)")
        
        # Determine save directory from checkpoint path
        # Extract directory from checkpoint path (e.g., "src/trained_weights/EQTransformerMagV2_STEAD_20251116_111103/")
        checkpoint_dir = os.path.dirname(checkpoint_path)
        if os.path.exists(checkpoint_dir) and "trained_weights" in checkpoint_dir:
            save_dir = checkpoint_dir
            print(f"✓ Will continue saving to: {save_dir}")
        
        # Override config parameters from checkpoint if they exist
        if "config" in checkpoint:
            config = checkpoint["config"]
            # Only override if not explicitly provided by user
            if learning_rate == 1e-3 and "learning_rate" in config:
                learning_rate = config["learning_rate"]
                print(f"✓ Using learning rate from checkpoint: {learning_rate}")
            if batch_size == 64 and "batch_size" in config:
                batch_size = config["batch_size"]
                print(f"✓ Using batch size from checkpoint: {batch_size}")
                # Note: Need to reload data with correct batch size
                print("⚠ Warning: Batch size from checkpoint may differ from loaded data")
            if "warmup_epochs" in config:
                warmup_epochs = config["warmup_epochs"]
            if "early_stopping_patience" in config:
                early_stopping_patience = config["early_stopping_patience"]
        
        print(f"\nResuming from epoch {start_epoch}/{epochs}")
        print(f"Best validation loss so far: {best_val_loss:.6f}")
        
        # Validate that we haven't already exceeded target epochs
        if start_epoch > epochs:
            print(f"\n⚠ WARNING: Checkpoint is at epoch {start_epoch-1}, but target epochs is {epochs}")
            print(f"Already completed the requested training. Use --epochs {start_epoch + 10} or higher to continue.")
            print(f"Exiting without further training.")
            return history  # Return existing history
    
    # Create save directory if not resuming from checkpoint
    if save_dir is None:
        script_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = f"src/trained_weights/{model_name}_{script_datetime}"
        os.makedirs(save_dir, exist_ok=True)
        print(f"\nSaving models to: {save_dir}\n")
    else:
        print()  # Empty line for formatting


    def train_loop(dataloader, epoch):
        """Training loop for one epoch."""
        model.train()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        num_batches = len(dataloader)

        # Create progress bar
        pbar = tqdm(dataloader, desc="Training", leave=False, disable=quiet)
        
        for batch in pbar:
            # Get input and temporal labels
            x = batch["X"].to(device)
            y_temporal = batch["magnitude"].to(device)  # Shape: (batch, time_steps)
            
            # Extract scalar magnitude as max of temporal labels (follows UMamba V3 approach)
            # After P-arrival, label is constant at source_magnitude
            y_scalar = y_temporal.max(dim=1)[0]  # Shape: (batch,)

            # Forward pass - model now outputs scalar magnitude
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred_scalar = model(x_preproc)  # Shape: (batch,)

            # Calculate loss on scalar predictions
            loss = criterion(y_pred_scalar, y_scalar)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping (important for transformers)
            grad_norm = 0.0
            if gradient_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )

            optimizer.step()

            # Track metrics
            batch_loss = loss.item()
            total_loss += batch_loss
            total_mse += batch_loss
            with torch.no_grad():
                mae = torch.abs(y_pred_scalar - y_scalar).mean().item()
                total_mae += mae

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{batch_loss:.4f}',
                'mae': f'{mae:.4f}',
                'grad': f'{grad_norm:.2f}' if gradient_clip else 'N/A'
            })

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_mse = total_mse / num_batches
        avg_rmse = avg_mse ** 0.5
        
        return avg_loss, avg_mae, avg_mse, avg_rmse

    def validation_loop(dataloader):
        """Validation loop."""
        model.eval()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        num_batches = len(dataloader)

        # Create progress bar
        pbar = tqdm(dataloader, desc="Validation", leave=False, disable=quiet)
        
        with torch.no_grad():
            for batch in pbar:
                x = batch["X"].to(device)
                y_temporal = batch["magnitude"].to(device)  # Shape: (batch, time_steps)
                
                # Extract scalar magnitude as max of temporal labels
                y_scalar = y_temporal.max(dim=1)[0]  # Shape: (batch,)

                # Forward pass - model outputs scalar magnitude
                x_preproc = model.annotate_batch_pre(x, {})
                y_pred_scalar = model(x_preproc)  # Shape: (batch,)

                # Calculate metrics on scalar predictions
                loss = criterion(y_pred_scalar, y_scalar)
                mae = torch.abs(y_pred_scalar - y_scalar).mean()

                batch_loss = loss.item()
                total_loss += batch_loss
                total_mae += mae.item()
                total_mse += batch_loss
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f'{batch_loss:.4f}',
                    'mae': f'{mae.item():.4f}'
                })

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_mse = total_mse / num_batches
        avg_rmse = avg_mse ** 0.5

        return avg_loss, avg_mae, avg_mse, avg_rmse

    # Training loop
    # (Variables already initialized above in checkpoint loading section)
    
    # Track total training time
    training_start_time = time.time()
    
    # Initialize epoch variable to handle case where loop doesn't execute
    epoch = start_epoch - 1  # Will be start_epoch-1 if loop never runs

    for epoch in range(start_epoch, epochs + 1):  # Resume from start_epoch
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs}")
        
        # Manual warmup learning rate (like UMamba V3)
        # Only apply warmup if we're in the warmup period
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for param_group in optimizer.param_groups:
                param_group['lr'] = learning_rate * warmup_factor
            print(f"Warmup phase: {epoch}/{warmup_epochs}")
        
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.6f}")
        print(f"{'='*60}")

        # Train
        train_loss, train_mae, train_mse, train_rmse = train_loop(train_loader, epoch)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        learning_rates.append(current_lr)

        # Validate
        val_loss, val_mae, val_mse, val_rmse = validation_loop(dev_loader)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{epochs} Summary")
        print(f"{'='*60}")
        print(f"Training   -> Loss: {train_loss:.6f} | MSE: {train_mse:.6f} | RMSE: {train_rmse:.6f} | MAE: {train_mae:.6f}")
        print(f"Validation -> Loss: {val_loss:.6f} | MSE: {val_mse:.6f} | RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f}")
        print(f"{'='*60}")

        # Learning rate scheduling (ReduceLROnPlateau) after warmup
        if epoch > warmup_epochs:
            scheduler.step(val_loss)

        # Check for improvement and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_path = os.path.join(save_dir, f"model_best.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_maes": train_maes,
                    "val_maes": val_maes,
                    "learning_rates": learning_rates,
                    "config": {
                        "model_name": model_name,
                        "learning_rate": learning_rate,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "optimizer": optimizer_name,
                        "weight_decay": weight_decay,
                        "scheduler_patience": scheduler_patience,
                        "scheduler_factor": scheduler_factor,
                        "gradient_clip": gradient_clip,
                        "early_stopping_patience": early_stopping_patience,
                        "warmup_epochs": warmup_epochs,
                    },
                },
                best_model_path,
            )
            print(f"\n✓ New best model saved! Val Loss: {val_loss:.6f}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epochs")

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"\nEarly stopping triggered after {early_stopping_patience} epochs without improvement"
            )
            print(f"Best validation loss: {best_val_loss:.6f}")
            break

        # Save periodic checkpoint
        if epoch % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch}.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "train_maes": train_maes,
                    "val_maes": val_maes,
                    "learning_rates": learning_rates,
                    "config": {
                        "model_name": model_name,
                        "learning_rate": learning_rate,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "optimizer": optimizer_name,
                        "weight_decay": weight_decay,
                        "scheduler_patience": scheduler_patience,
                        "scheduler_factor": scheduler_factor,
                        "gradient_clip": gradient_clip,
                        "early_stopping_patience": early_stopping_patience,
                        "warmup_epochs": warmup_epochs,
                    },
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    final_model_path = os.path.join(save_dir, "model_final.pt")
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "train_loss": train_losses[-1] if train_losses else None,
            "val_loss": val_losses[-1] if val_losses else None,
            "best_val_loss": best_val_loss,
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_maes": train_maes,
            "val_maes": val_maes,
            "learning_rates": learning_rates,
            "config": {
                "model_name": model_name,
                "learning_rate": learning_rate,
                "epochs": epochs,
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "scheduler_patience": scheduler_patience,
                "scheduler_factor": scheduler_factor,
                "gradient_clip": gradient_clip,
                "early_stopping_patience": early_stopping_patience,
                "warmup_epochs": warmup_epochs,
            },
        },
        final_model_path,
    )
    print(f"\nFinal model saved: {final_model_path}")

    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    # Save training history
    # Extract datetime from save_dir if resuming, otherwise create new timestamp
    if checkpoint_path:
        # Extract timestamp from existing directory name
        dir_name = os.path.basename(save_dir)
        # Try to extract timestamp (format: ModelName_Dataset_YYYYMMDD_HHMMSS)
        parts = dir_name.split('_')
        if len(parts) >= 2 and len(parts[-2]) == 8 and len(parts[-1]) == 6:
            script_datetime = f"{parts[-2]}_{parts[-1]}"
        else:
            script_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        script_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_maes": train_maes,
        "val_maes": val_maes,
        "learning_rates": learning_rates,
        "best_val_loss": best_val_loss,
        "total_training_time": total_training_time,
        "config": {
            "model_name": model_name,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "batch_size": batch_size,
            "optimizer": optimizer_name,
            "weight_decay": weight_decay,
            "scheduler_patience": scheduler_patience,
            "scheduler_factor": scheduler_factor,
            "gradient_clip": gradient_clip,
            "early_stopping_patience": early_stopping_patience,
            "warmup_epochs": warmup_epochs,
        },
    }
    history_path = os.path.join(save_dir, f"training_history_{script_datetime}.pt")
    torch.save(history, history_path)
    print(f"Training history saved: {history_path}")

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation RMSE: {best_val_loss**0.5:.6f}")
    if train_losses:  # Only print if we have training history
        print(f"Final train loss: {train_losses[-1]:.6f}")
        print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print("=" * 60)

    return history


def load_checkpoint(
    model: WaveformModel, checkpoint_path, optimizer=None, device="cpu"
):
    """
    Load a saved checkpoint.

    Args:
        model: EQTransformerMag model instance
        checkpoint_path: Path to checkpoint file
        optimizer: Optional optimizer to load state
        device: Device to load model on

    Returns:
        Loaded model, optimizer (if provided), and checkpoint info
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Validation loss: {checkpoint.get('val_loss', 'N/A')}")

    return model, optimizer, checkpoint


# Example usage
if __name__ == "__main__":

    # Initialize model
    model = EQTransformerMagV2(
        in_channels=3,
        in_samples=3001,  # 30 seconds at 100Hz
        sampling_rate=100,
        lstm_blocks=3,
        drop_rate=0.1,
        norm="std",
    )

    # Load dataset
    data = sbd.ETHZ(sampling_rate=100)

    # Train model
    history = train_eqtransformer_mag(
        model_name="eqtransformermag_v1",
        model=model,
        data=data,
        learning_rate=1e-4,  # Lower LR for transformers
        epochs=50,
        batch_size=64,  # Smaller batch size for memory efficiency
        optimizer_name="AdamW",
        weight_decay=1e-5,
        scheduler_patience=7,
        save_every=5,
        gradient_clip=1.0,
        warmup_epochs=5,
    )
