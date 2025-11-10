import os
import argparse
import time
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional
from tqdm import tqdm

import seisbench.data as sbd
from my_project.models.ViT.model import ViTMagnitudeEstimator
from my_project.loaders import data_loader as dl


def train_vit_magnitude(
    model_name: str,
    model: ViTMagnitudeEstimator,
    data: sbd.BenchmarkDataset,
    learning_rate: float = 1e-3,  # Lower LR typical for transformers
    epochs: int = 100,
    batch_size: int = 64,  # Smaller batch size for ViT due to memory
    optimizer_name: str = "AdamW",
    weight_decay: float = 1e-2,  # Higher weight decay for transformers
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    save_every: int = 5,
    gradient_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 5,
    warmup_epochs: int = 5,  # Longer warmup for transformers
    quiet: bool = False,
):
    """
    Train ViTMagnitudeEstimator model for magnitude regression.

    Args:
        model_name: Name for saving model checkpoints
        model: ViTMagnitudeEstimator model instance
        data: Dataset to train on
        learning_rate: Learning rate for optimizer (lower for transformers)
        epochs: Number of training epochs
        batch_size: Batch size for training (smaller for ViT memory usage)
        optimizer_name: Optimizer type ("Adam", "AdamW", or "SGD")
        weight_decay: Weight decay for optimizer (higher for transformers)
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor to reduce learning rate by
        save_every: Save model every N epochs
        gradient_clip: Maximum gradient norm (None to disable)
        early_stopping_patience: Stop training if no improvement for N epochs
        warmup_epochs: Number of epochs for learning rate warmup (longer for transformers)
        quiet: If True, disable tqdm progress bars
    """
    print("\n" + "=" * 60)
    print("TRAINING VISION TRANSFORMER FOR MAGNITUDE ESTIMATION")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {data.__class__.__name__}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Weight decay: {weight_decay}")
    print(f"Optimizer: {optimizer_name}")
    print(f"Gradient clipping: {gradient_clip}")
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
    print("\nLoading training data...")
    train_generator, train_loader, _ = dl.load_dataset(
        data, model, "train", batch_size=batch_size
    )
    print("\nLoading validation data...")
    dev_generator, dev_loader, _ = dl.load_dataset(
        data, model, "dev", batch_size=batch_size
    )

    print(f"Training samples: {len(train_generator)}")
    print(f"Validation samples: {len(dev_generator)}")

    # Setup optimizer - AdamW is typically best for transformers
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),  # Standard transformer betas
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

    # Loss function - MSE for magnitude regression
    criterion = nn.MSELoss()

    # Alternative loss functions for experimentation:
    # criterion = nn.L1Loss()  # MAE - more robust to outliers
    # criterion = nn.HuberLoss(delta=1.0)  # Robust to outliers
    # criterion = nn.SmoothL1Loss()  # Similar to Huber loss

    # Learning rate scheduler with warmup
    def warmup_lambda(epoch):
        """Linear warmup followed by constant learning rate"""
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / float(max(1, warmup_epochs)))
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda
    )

    # ReduceLROnPlateau for post-warmup adjustment
    reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_factor,
        patience=scheduler_patience,
        verbose=True,
        min_lr=1e-7,  # Minimum learning rate
    )

    def train_loop(dataloader):
        """Training loop for one epoch"""
        model.train()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        num_batches = len(dataloader)

        # Create progress bar
        pbar = tqdm(dataloader, desc="Training", leave=False, disable=quiet)
        
        for batch in pbar:
            # Get input and target
            x = batch["X"].to(device)
            y_true = batch["magnitude"].to(device)  # Target magnitudes

            # Handle target shape - should be (batch,) for scalar regression
            if y_true.dim() == 2:
                # If shape is (batch, samples), take mean or first value
                y_true = y_true.mean(dim=1)  # Average across time for single magnitude
            
            # Ensure y_true is 1D: (batch,)
            y_true = y_true.squeeze()

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)  # Should output (batch,)

            # Ensure y_pred is also 1D
            y_pred = y_pred.squeeze()

            # Calculate loss
            loss = criterion(y_pred, y_true)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            grad_norm = 0.0
            if gradient_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            total_mse += batch_loss
            
            with torch.no_grad():
                mae = torch.abs(y_pred - y_true).mean().item()
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
        """Validation loop"""
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
                y_true = batch["magnitude"].to(device)

                # Handle target shape - should be (batch,) for scalar regression
                if y_true.dim() == 2:
                    # If shape is (batch, samples), take mean or first value
                    y_true = y_true.mean(dim=1)  # Average across time for single magnitude
                
                # Ensure y_true is 1D: (batch,)
                y_true = y_true.squeeze()

                # Forward pass
                x_preproc = model.annotate_batch_pre(x, {})
                y_pred = model(x_preproc)  # Should output (batch,)
                
                # Ensure y_pred is also 1D
                y_pred = y_pred.squeeze()

                # Calculate loss
                loss = criterion(y_pred, y_true)
                mae = torch.abs(y_pred - y_true).mean()
                
                batch_loss = loss.item()
                total_loss += batch_loss
                total_mse += batch_loss
                total_mae += mae.item()
                
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

    # Create save directory with timestamp
    script_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"src/trained_weights/{model_name}_{script_datetime}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving models to: {save_dir}")

    # Training history
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0
    
    # Track total training time
    training_start_time = time.time()

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # Training loop
    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'='*60}")

        # Train
        train_loss, train_mae, train_mse, train_rmse = train_loop(train_loader)
        train_losses.append(train_loss)
        train_maes.append(train_mae)

        # Validation
        val_loss, val_mae, val_mse, val_rmse = validation_loop(dev_loader)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # Print epoch summary
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{epochs} Summary")
        print(f"{'='*60}")
        print(f"Training   -> Loss: {train_loss:.6f} | MSE: {train_mse:.6f} | RMSE: {train_rmse:.6f} | MAE: {train_mae:.6f}")
        print(f"Validation -> Loss: {val_loss:.6f} | MSE: {val_mse:.6f} | RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f}")
        print(f"{'='*60}")

        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            reduce_scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0

            # Save best model
            best_model_path = os.path.join(save_dir, "model_best.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "train_loss": train_loss,
                "best_val_loss": best_val_loss,
            }, best_model_path)
            print(f"✓ New best model saved (val_loss: {val_loss:.6f})")
        else:
            epochs_without_improvement += 1

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(
                save_dir, f"model_epoch_{epoch+1}_{script_datetime}.pt"
            )
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "best_val_loss": best_val_loss,
                },
                checkpoint_path,
            )
            print(f"✓ Checkpoint saved: {checkpoint_path}")

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(
                f"\nEarly stopping after {epoch+1} epochs (no improvement for {early_stopping_patience} epochs)"
            )
            break

    # Save final model
    final_model_path = os.path.join(save_dir, f"model_final_{script_datetime}.pt")
    torch.save({
        "epoch": epoch + 1,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
    }, final_model_path)

    # Calculate total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)

    # Save training history
    history_path = os.path.join(save_dir, f"training_history_{script_datetime}.pt")
    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "train_maes": train_maes,
            "val_maes": val_maes,
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1,
            "total_training_time": total_training_time,
            "model_config": model.get_model_args(),
            "training_config": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "optimizer": optimizer_name,
                "weight_decay": weight_decay,
                "scheduler_patience": scheduler_patience,
                "gradient_clip": gradient_clip,
                "warmup_epochs": warmup_epochs,
            },
        },
        history_path,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation RMSE: {best_val_loss**0.5:.6f}")
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Total epochs: {epoch + 1}")
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print(f"Models saved in: {save_dir}")
    print("=" * 60)

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_maes": train_maes,
        "val_maes": val_maes,
        "best_val_loss": best_val_loss,
        "save_dir": save_dir,
        "final_epoch": epoch + 1,
    }


# Example usage
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ViT Magnitude Estimation Model")
    parser.add_argument(
        "--model_name", default="ViTMag_ETHZ", help="Model name for saving"
    )
    parser.add_argument(
        "--dataset",
        default="ETHZ",
        choices=["ETHZ", "GEOFON", "STEAD"],
        help="Dataset to use",
    )
    parser.add_argument("--epochs", type=int, default=100, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
    parser.add_argument(
        "--learning_rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight_decay", type=float, default=1e-2, help="Weight decay")
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

    # Initialize model
    model = ViTMagnitudeEstimator(
        in_channels=3,
        sampling_rate=100,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_transformer_blocks=args.num_blocks,
        patch_size=args.patch_size,
        norm="std",
    )

    # Train model
    history = train_vit_magnitude(
        model_name=args.model_name,
        model=model,
        data=data,
        learning_rate=args.learning_rate,
        epochs=args.epochs,
        batch_size=args.batch_size,
        weight_decay=args.weight_decay,
        optimizer_name="AdamW",
        gradient_clip=1.0,
    )

    print(f"\nTraining completed! Best validation loss: {history['best_val_loss']:.6f}")
