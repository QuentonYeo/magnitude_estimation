import os
from datetime import datetime
import torch
import torch.nn as nn
import seisbench.data as sbd


def train_model(
    model_name: str,
    model,  # YourModelName instance
    data: sbd.BenchmarkDataset,
    learning_rate=0.001,
    epochs=100,
    batch_size=256,
    optimizer_name="Adam",
    weight_decay=1e-5,
    early_stopping_patience=10,
    save_every=5,
):
    """
    Train model for magnitude regression.

    Args:
        model_name: Name for saving model checkpoints
        model: Model instance
        data: Dataset to train on
        learning_rate: Learning rate for optimizer (default: 0.001)
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size for training
        optimizer_name: Optimizer type ("Adam" or "AdamW")
        weight_decay: Weight decay for optimizer
        early_stopping_patience: Stop if validation loss doesn't decrease for N epochs (default: 10)
        save_every: Save model every N epochs
    """
    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    print(f"Training {model.__class__.__name__} on {data.__class__.__name__}")

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

    # Early stopping tracking
    best_val_loss = float("inf")
    epochs_without_improvement = 0

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

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_without_improvement = 0
            # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # best_model_path = os.path.join(save_dir, f"model_best_{timestamp}.pt")
            # torch.save(model.state_dict(), best_model_path)
            # print(f"New best model saved: {best_model_path}")
        else:
            epochs_without_improvement += 1
            print(f"No improvement for {epochs_without_improvement} epoch(s)")

            if epochs_without_improvement >= early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                print(f"Best validation loss: {best_val_loss:.6f}")
                break

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

    return history


# Example usage
if __name__ == "__main__":
    from my_project.loaders import data_loader as dl
    from my_project.models.AMAG.model import AMAG

    # Initialize model
    model = AMAG(
        in_channels=3,
        classes=1,  # Single magnitude output
        phases="M",  # Magnitude
        sampling_rate=100,
        norm="std",
        encoder_depth=4,
        kernel_size=5,
        leaky_relu_slope=0.01,
    )

    # Move model to GPU if available
    model.to_preferred_device(verbose=True)

    print(f"Model initialized on {model.device}")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load your dataset
    data = sbd.ETHZ(sampling_rate=100)

    # Train the model
    history = train_model(
        model_name="AMAG_test",
        model=model,
        data=data,
        learning_rate=1e-5,
        epochs=50,
        batch_size=256,
        optimizer_name="Adam",
        weight_decay=1e-5,
        scheduler_patience=5,
        save_every=5,
    )
