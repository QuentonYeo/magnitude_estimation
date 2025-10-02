import os
import argparse
import numpy as np
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
    save_every=5,
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
        save_every: Save model every N epochs
    """
    print(f"Training PhaseNetMag on {data.__class__.__name__}")
    print(f"Model device: {model.device}")

    # Load data
    train_generator, train_loader, _ = dl.load_dataset(
        data, model, "train", batch_size=batch_size
    )
    dev_generator, dev_loader, _ = dl.load_dataset(
        data, model, "dev", batch_size=batch_size
    )

    print("Data successfully loaded")
    print(f"Training samples: {len(train_generator)}")
    print(f"Validation samples: {len(dev_generator)}")

    # Setup optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error for regression

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    def train_loop(dataloader):
        model.train()
        total_loss = 0
        num_batches = len(dataloader)

        for batch_id, batch in enumerate(dataloader):
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

            total_loss += loss.item()

            if batch_id % 10 == 0:
                current = batch_id * batch["X"].shape[0]
                print(
                    f"loss: {loss.item():>7f}  [{current:>5d}/{len(dataloader.dataset):>5d}]"
                )

        return total_loss / num_batches

    def validation_loop(dataloader):
        model.eval()
        total_loss = 0
        num_batches = len(dataloader)

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(model.device)
                y_true = batch["magnitude"].to(model.device)

                # Forward pass
                x_preproc = model.annotate_batch_pre(x, {})
                y_pred = model(x_preproc)
                y_pred = y_pred.squeeze(1)

                # Calculate loss
                loss = criterion(y_pred, y_true)
                total_loss += loss.item()

        avg_loss = total_loss / num_batches
        print(f"Validation avg loss: {avg_loss:>8f}")
        return avg_loss

    # Create save directory
    save_dir = f"src/trained_weights/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    # Training loop
    best_val_loss = float("inf")
    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)

        # Train
        train_loss = train_loop(train_loader)
        train_losses.append(train_loss)

        # Validate
        val_loss = validation_loop(dev_loader)
        val_losses.append(val_loss)

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            best_model_path = os.path.join(save_dir, f"model_best_{timestamp}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"New best model saved: {best_model_path}")

        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            checkpoint_path = os.path.join(
                save_dir, f"model_epoch_{epoch+1}_{timestamp}.pt"
            )
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

    # Save final model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    final_model_path = os.path.join(save_dir, f"model_final_{timestamp}.pt")
    torch.save(model.state_dict(), final_model_path)
    print(f"Final model saved: {final_model_path}")

    # Save training history
    history = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "best_val_loss": best_val_loss,
    }
    history_path = os.path.join(save_dir, f"training_history_{timestamp}.pt")
    torch.save(history, history_path)
    print(f"Training history saved: {history_path}")

    print("\nTraining completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")


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

    args = parser.parse_args()

    # Load dataset
    if args.dataset == "ETHZ":
        data = sbd.ETHZ(sampling_rate=100)
    elif args.dataset == "GEOFON":
        data = sbd.GEOFON(sampling_rate=100)
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    print(f"Loaded dataset: {data}")

    # Create model
    model = PhaseNetMag(in_channels=3, sampling_rate=100, norm="std", filter_factor=1)

    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Model moved to device: {device}")

    # Model name for saving
    model_name = f"PhaseNetMag_{args.dataset}"

    # Train model
    train_phasenet_mag(
        model_name=model_name,
        model=model,
        data=data,
        learning_rate=args.lr,
        epochs=args.epochs,
        batch_size=args.batch_size,
        save_every=args.save_every,
    )


if __name__ == "__main__":
    main()
