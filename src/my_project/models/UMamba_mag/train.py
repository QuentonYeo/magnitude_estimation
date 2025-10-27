import os
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
import seisbench.data as sbd
from seisbench.models.base import WaveformModel

from my_project.models.UMamba_mag.model import UMambaMag
from my_project.loaders import data_loader as dl


def train_umamba_mag(
    model_name: str,
    model: WaveformModel,
    data: sbd.BenchmarkDataset,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    batch_size: int = 64,  # Smaller batch size due to Mamba complexity
    optimizer_name: str = "AdamW",
    weight_decay: float = 1e-5,
    scheduler_patience: int = 5,
    scheduler_factor: float = 0.5,
    save_every: int = 5,
    gradient_clip: Optional[float] = 1.0,
    early_stopping_patience: int = 10,
    warmup_epochs: int = 5,
):
    """
    Train UMambaMag model for magnitude regression.

    Args:
        model_name: Name for saving model checkpoints
        model: UMambaMag model instance
        data: Dataset to train on
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Batch size for training (smaller for Mamba)
        optimizer_name: Optimizer type ("Adam", "AdamW", or "SGD")
        weight_decay: Weight decay for optimizer
        scheduler_patience: Patience for learning rate scheduler
        scheduler_factor: Factor to reduce learning rate by
        save_every: Save model every N epochs
        gradient_clip: Maximum gradient norm (None to disable)
        early_stopping_patience: Epochs to wait before early stopping
        warmup_epochs: Number of epochs for learning rate warmup
    """
    print("\n" + "=" * 60)
    print("TRAINING UMAMBAMAG")
    print("=" * 60)
    print(f"Model: {model_name}")
    print(f"Dataset: {data.__class__.__name__}")
    print(f"Batch size: {batch_size}")
    print(f"Learning rate: {learning_rate}")
    print(f"Epochs: {epochs}")
    print(f"Warmup epochs: {warmup_epochs}")
    print("=" * 60)

    # Move model to device
    model.to("cuda:1")
    # Get the actual device where the model ended up
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

    # Loss function - MSE for regression
    criterion = nn.MSELoss()

    # Learning rate scheduler with warmup and decay
    base_lr = learning_rate

    def warmup_lambda(epoch):
        # Warmup from 10% to 100% linearly over warmup_epochs
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
    )

    def train_loop(dataloader, epoch):
        """Training loop for one epoch."""
        model.train()
        total_loss = 0
        total_mae = 0
        num_batches = len(dataloader)

        for batch_id, batch in enumerate(dataloader):
            # Get input and target
            x = batch["X"].to(device)
            y_true = batch["magnitude"].to(device)

            # Handle different target shapes
            if y_true.dim() == 1:
                y_true = y_true.unsqueeze(1).expand(-1, x.shape[-1])
            elif y_true.dim() == 2 and y_true.shape[1] == 1:
                y_true = y_true.expand(-1, x.shape[-1])

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)

            # Calculate loss
            loss = criterion(y_pred, y_true)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()

            # Gradient clipping
            if gradient_clip is not None:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    model.parameters(), gradient_clip
                )
            else:
                grad_norm = 0.0

            optimizer.step()

            # Track metrics
            total_loss += loss.item()
            with torch.no_grad():
                mae = torch.abs(y_pred - y_true).mean().item()
                total_mae += mae

            # Print progress
            if batch_id % 10 == 0:
                current = batch_id * batch["X"].shape[0]
                print(
                    f"Batch {batch_id:4d}/{num_batches} | "
                    f"Loss: {loss.item():.4f} | "
                    f"MAE: {mae:.4f} | "
                    f"Grad: {grad_norm:.2f} | "
                    f"[{current:>5d}/{len(dataloader.dataset):>5d}]"
                )

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        return avg_loss, avg_mae

    def validation_loop(dataloader):
        """Validation loop."""
        model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(device)
                y_true = batch["magnitude"].to(device)

                # Handle different target shapes
                if y_true.dim() == 1:
                    y_true = y_true.unsqueeze(1).expand(-1, x.shape[-1])
                elif y_true.dim() == 2 and y_true.shape[1] == 1:
                    y_true = y_true.expand(-1, x.shape[-1])

                # Forward pass
                x_preproc = model.annotate_batch_pre(x, {})
                y_pred = model(x_preproc)

                # Calculate metrics
                loss = criterion(y_pred, y_true)
                mae = torch.abs(y_pred - y_true).mean()

                total_loss += loss.item()
                total_mae += mae.item()

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches

        print(f"\nValidation Results:")
        print(f"  Avg Loss (MSE): {avg_loss:.6f}")
        print(f"  Avg MAE: {avg_mae:.6f}")
        print(f"  RMSE: {avg_loss**0.5:.6f}")

        return avg_loss, avg_mae

    # Create save directory with timestamp
    script_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"src/trained_weights/{model_name}_{script_datetime}"
    os.makedirs(save_dir, exist_ok=True)
    print(f"\nSaving models to: {save_dir}\n")

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    learning_rates = []
    epochs_without_improvement = 0

    for epoch in range(epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{epochs}")

        # Report current learning rate
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate: {current_lr:.2e}")
        print(f"{'='*60}")

        # Train
        train_loss, train_mae = train_loop(train_loader, epoch)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        learning_rates.append(current_lr)

        print(f"\nTraining Results:")
        print(f"  Avg Loss (MSE): {train_loss:.6f}")
        print(f"  Avg MAE: {train_mae:.6f}")
        print(f"  RMSE: {train_loss**0.5:.6f}")

        # Validate
        val_loss, val_mae = validation_loop(dev_loader)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # Step warmup scheduler
        warmup_scheduler.step()

        # Step reduce scheduler after warmup
        if epoch >= warmup_epochs:
            reduce_scheduler.step(val_loss)

        # Check for improvement and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            best_model_path = os.path.join(save_dir, f"model_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_mae": val_mae,
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
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"model_epoch_{epoch+1}.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_loss": train_loss,
                    "val_loss": val_loss,
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
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_loss": train_losses[-1],
            "val_loss": val_losses[-1],
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

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_maes": train_maes,
        "val_maes": val_maes,
        "learning_rates": learning_rates,
        "best_val_loss": best_val_loss,
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
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print("=" * 60)

    return history


def load_checkpoint(
    model: WaveformModel, checkpoint_path, optimizer=None, device="cpu"
):
    """
    Load a saved checkpoint.

    Args:
        model: UMambaMag model instance
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

    # Train model
    history = train_umamba_mag(
        model_name="umambamag_v1",
        model=model,
        data=data,
        learning_rate=1e-3,
        epochs=50,
        batch_size=64,
        optimizer_name="AdamW",
        weight_decay=1e-5,
        scheduler_patience=7,
        scheduler_factor=0.5,
        save_every=5,
        gradient_clip=1.0,
        early_stopping_patience=10,
        warmup_epochs=5,
    )
