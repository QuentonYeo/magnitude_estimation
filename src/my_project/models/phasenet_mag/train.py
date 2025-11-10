import os
import argparse
import time
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm

import seisbench.data as sbd
from my_project.models.phasenet_mag.model import PhaseNetMag
from my_project.loaders import data_loader as dl


def train_phasenet_mag(
    model_name: str,
    model: PhaseNetMag,
    data: sbd.BenchmarkDataset,
    learning_rate=1e-3,
    epochs=50,
    batch_size=256,
    optimizer_name="Adam",
    weight_decay=1e-5,
    scheduler_patience=5,
    save_every=5,
    early_stopping_patience=5,
    warmup_epochs=5,
    quiet=False,
):
    """
    Train PhaseNetMag model for magnitude regression.

    Args:
        model_name: Name for saving model checkpoints
        model: PhaseNetMag model instance
        data: Dataset to train on
        learning_rate: Learning rate for optimizer
        epochs: Number of training epochs
        batch_size: Batch size for training
        optimizer_name: Optimizer type ("Adam" or "AdamW")
        weight_decay: Weight decay for optimizer
        scheduler_patience: Patience for learning rate scheduler
        save_every: Save model every N epochs
        early_stopping_patience: Stop training if no improvement for N epochs
        warmup_epochs: Number of epochs for learning rate warmup
        quiet: If True, disable tqdm progress bars
    """
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    print(f"Training PhaseNetMag on {data.__class__.__name__}")

    # Load data
    train_generator, train_loader, _ = dl.load_dataset(
        data, model, "train", batch_size=batch_size
    )
    dev_generator, dev_loader, _ = dl.load_dataset(
        data, model, "dev", batch_size=batch_size
    )

    # Setup optimizer and loss function
    if optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    criterion = nn.MSELoss()  # Mean Squared Error for regression

    # Learning rate scheduler with warmup (LambdaLR) and ReduceLROnPlateau for decay
    def warmup_lambda(epoch):
        # from 10% -> 100% linearly over warmup_epochs
        if epoch < warmup_epochs:
            return 0.1 + 0.9 * (epoch / float(max(1, warmup_epochs)))
        return 1.0

    warmup_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=warmup_lambda
    )

    # ReduceLROnPlateau for post-warmup adjustment
    reduce_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience, verbose=True
    )

    def train_loop(dataloader):
        model.train()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        num_batches = len(dataloader)

        # Create progress bar
        pbar = tqdm(dataloader, desc="Training", leave=False, disable=quiet)
        
        for batch in pbar:
            # Get input and target
            x = batch["X"].to(model.device)
            y_true = batch["magnitude"].to(model.device)  # Target magnitudes

            # Forward pass
            x_preproc = model.annotate_batch_pre(x, {})
            y_pred = model(x_preproc)  # Shape: (batch, 1, samples)

            # Reshape for loss calculation
            y_pred = y_pred.squeeze(1)  # Remove channel dimension: (batch, samples)

            # Calculate loss
            loss = criterion(y_pred, y_true)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
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
                'mae': f'{mae:.4f}'
            })

        avg_loss = total_loss / num_batches
        avg_mae = total_mae / num_batches
        avg_mse = total_mse / num_batches
        avg_rmse = avg_mse ** 0.5
        
        return avg_loss, avg_mae, avg_mse, avg_rmse

    def validation_loop(dataloader):
        model.eval()
        total_loss = 0
        total_mae = 0
        total_mse = 0
        num_batches = len(dataloader)

        # Create progress bar
        pbar = tqdm(dataloader, desc="Validation", leave=False, disable=quiet)
        
        with torch.no_grad():
            for batch in pbar:
                x = batch["X"].to(model.device)
                y_true = batch["magnitude"].to(model.device)

                # Forward pass
                x_preproc = model.annotate_batch_pre(x, {})
                y_pred = model(x_preproc)
                y_pred = y_pred.squeeze(1)

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
    print(f"Saving models to: {save_dir}")

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    train_maes = []
    val_maes = []
    learning_rates = []
    epochs_without_improvement = 0
    
    # Track total training time
    training_start_time = time.time()

    for epoch in range(epochs):
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"{'='*50}")

        # Train
        train_loss, train_mae, train_mse, train_rmse = train_loop(train_loader)
        train_losses.append(train_loss)
        train_maes.append(train_mae)
        learning_rates.append(optimizer.param_groups[0]["lr"])

        # Validate
        val_loss, val_mae, val_mse, val_rmse = validation_loop(dev_loader)
        val_losses.append(val_loss)
        val_maes.append(val_mae)

        # Print epoch summary
        print(f"\n{'='*50}")
        print(f"Epoch {epoch + 1}/{epochs} Summary")
        print(f"{'='*50}")
        print(f"Training   -> Loss: {train_loss:.6f} | MSE: {train_mse:.6f} | RMSE: {train_rmse:.6f} | MAE: {train_mae:.6f}")
        print(f"Validation -> Loss: {val_loss:.6f} | MSE: {val_mse:.6f} | RMSE: {val_rmse:.6f} | MAE: {val_mae:.6f}")
        print(f"{'='*50}")

        # Step warmup LambdaLR ONCE per epoch AFTER optimizer steps
        warmup_scheduler.step()

        # Learning rate scheduling (ReduceLROnPlateau) after warmup
        if epoch >= warmup_epochs:
            reduce_scheduler.step(val_loss)

        # Check for improvement and early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # Save best model checkpoint
            best_model_path = os.path.join(save_dir, "model_best.pt")
            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "best_val_loss": best_val_loss,
                    "config": {
                        "model_name": model_name,
                        "learning_rate": learning_rate,
                        "epochs": epochs,
                        "batch_size": batch_size,
                        "optimizer": optimizer_name,
                        "weight_decay": weight_decay,
                        "scheduler_patience": scheduler_patience,
                        "early_stopping_patience": early_stopping_patience,
                    },
                },
                best_model_path,
            )
            print(f"✓ New best model saved! Val Loss: {val_loss:.6f}")
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
                        "early_stopping_patience": early_stopping_patience,
                    },
                },
                checkpoint_path,
            )
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model checkpoint
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
                "early_stopping_patience": early_stopping_patience,
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
            "early_stopping_patience": early_stopping_patience,
        },
    }
    history_path = os.path.join(save_dir, f"training_history_{script_datetime}.pt")
    torch.save(history, history_path)
    print(f"Training history saved: {history_path}")

    # Print final summary
    print("\n" + "=" * 50)
    print("TRAINING COMPLETED")
    print("=" * 50)
    print(f"Best validation loss: {best_val_loss:.6f}")
    print(f"Best validation RMSE: {best_val_loss**0.5:.6f}")
    print(f"Final train loss: {train_losses[-1]:.6f}")
    print(f"Final val loss: {val_losses[-1]:.6f}")
    print(f"Total training time: {hours:02d}h {minutes:02d}m {seconds:02d}s")
    print("=" * 50)

    return {
        "best_val_loss": best_val_loss,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "train_maes": train_maes,
        "val_maes": val_maes,
        "save_dir": save_dir,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train PhaseNetMag for magnitude regression"
    )
    parser.add_argument(
        "--dataset", type=str, default="ETHZ", help="Dataset name (ETHZ, GEOFON, etc.)"
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--save_every", type=int, default=5, help="Save model every N epochs"
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="Adam",
        choices=["Adam", "AdamW"],
        help="Optimizer type",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=1e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--scheduler_patience",
        type=int,
        default=5,
        help="Patience for learning rate scheduler",
    )
    parser.add_argument(
        "--filter_factor",
        type=int,
        default=1,
        help="Filter factor for model architecture",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=10,
        help="Stop training if no improvement for N epochs",
    )

    args = parser.parse_args()

    # Load dataset
    if args.dataset == "ETHZ":
        data = sbd.ETHZ(sampling_rate=100)
    elif args.dataset == "GEOFON":
        data = sbd.GEOFON(sampling_rate=100)
    elif args.dataset == "STEAD":
        data = sbd.STEAD(sampling_rate=100)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"Loaded dataset: {data}")

    # Model name for saving
    model_name = f"PhaseNetMag_{args.dataset}"

    # Create and train model
    model = PhaseNetMag(
        in_channels=3,
        sampling_rate=100,
        norm="std",
        filter_factor=args.filter_factor,
    )
    model.to_preferred_device(verbose=True)

    # Train model
    train_phasenet_mag(
        model_name=model_name,
        model=model,
        data=data,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        optimizer_name=args.optimizer,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        save_every=args.save_every,
        early_stopping_patience=args.early_stopping_patience,
    )


if __name__ == "__main__":
    main()
