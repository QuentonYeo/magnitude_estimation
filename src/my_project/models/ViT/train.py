import os
import argparse
import torch
import torch.nn as nn
from datetime import datetime
from typing import Optional

import seisbench.data as sbd
from my_project.models.ViT.model import ViTMagnitudeEstimator
from my_project.loaders import data_loader as dl


def train_vit_magnitude(
    model_name: str,
    model: ViTMagnitudeEstimator,
    data: sbd.BenchmarkDataset,
    learning_rate: float = 1e-4,  # Lower LR typical for transformers
    epochs: int = 100,
    batch_size: int = 64,  # Smaller batch size for ViT due to memory
    optimizer_name: str = "AdamW",
    weight_decay: float = 1e-2,  # Higher weight decay for transformers
    scheduler_patience: int = 10,
    scheduler_factor: float = 0.5,
    save_every: int = 10,
    gradient_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 20,
    warmup_epochs: int = 10,  # Longer warmup for transformers
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

    # Move model to device
    model.to_preferred_device(verbose=True)

    # Get the actual device where the model ended up
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
        num_batches = len(dataloader)

        for batch_id, batch in enumerate(dataloader):
            # Get input and target
            x = batch["X"].to(device)
            y_true = batch["magnitude"].to(device)  # Target magnitudes

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)  # Shape: (batch, 1, 3001)

            # Reshape for loss calculation - same as PhaseNetMag
            y_pred = y_pred.squeeze(1)  # Remove channel dimension: (batch, 3001)

            # Calculate loss - same as PhaseNetMag
            loss = criterion(y_pred, y_true)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)

            optimizer.step()

            total_loss += loss.item()

            # Progress logging
            if batch_id % 20 == 0:
                current = batch_id * x.shape[0]
                print(
                    f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}] "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

        return total_loss / num_batches

    def validation_loop(dataloader):
        """Validation loop"""
        model.eval()
        total_loss = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(device)
                y_true = batch["magnitude"].to(device)

                # Forward pass
                x_preproc = model.annotate_batch_pre(x, {})
                y_pred = model(x_preproc)  # Shape: (batch, 1, 3001)

                # Reshape for loss calculation - same as PhaseNetMag
                y_pred = y_pred.squeeze(1)  # Remove channel dimension: (batch, 3001)

                # Calculate loss - same as PhaseNetMag
                loss = criterion(y_pred, y_true)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Validation avg loss: {avg_loss:>8f}")
        return avg_loss

    # Create save directory with timestamp
    script_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"src/trained_weights/{model_name}_{script_datetime}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving models to: {save_dir}")

    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    # Training loop
    for epoch in range(epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 30)

        # Train
        train_loss = train_loop(train_loader)
        train_losses.append(train_loss)

        # Validation
        val_loss = validation_loop(dev_loader)
        val_losses.append(val_loss)

        print(
            f"Epoch {epoch+1}: Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
        )

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
            torch.save(model.state_dict(), best_model_path)
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
    torch.save(model.state_dict(), final_model_path)

    # Save training history
    history_path = os.path.join(save_dir, f"training_history_{script_datetime}.pt")
    torch.save(
        {
            "train_losses": train_losses,
            "val_losses": val_losses,
            "best_val_loss": best_val_loss,
            "final_epoch": epoch + 1,
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
    print(f"Final training loss: {train_losses[-1]:.6f}")
    print(f"Final validation loss: {val_losses[-1]:.6f}")
    print(f"Total epochs: {epoch + 1}")
    print(f"Models saved in: {save_dir}")

    return {
        "train_losses": train_losses,
        "val_losses": val_losses,
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
