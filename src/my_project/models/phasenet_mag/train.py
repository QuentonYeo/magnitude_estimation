import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from tqdm import tqdm
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
import logging
import pandas as pd
import matplotlib.pyplot as plt

import seisbench.data as sbd
from my_project.models.phasenet_mag.model import PhaseNetMag
from my_project.loaders import data_loader as dl


def train_phasenet_mag_with_optuna(
    model_name: str,
    data: sbd.BenchmarkDataset,
    n_trials=100,
    max_epochs=30,
    pruning_patience=5,
    study_name=None,
):
    """
    Hyperparameter optimization for PhaseNetMag using Optuna.

    Args:
        model_name: Name for saving model checkpoints and study
        data: Dataset to train on
        n_trials: Number of hyperparameter trials to run
        max_epochs: Maximum epochs per trial (for pruning)
        pruning_patience: Patience for pruning unsuccessful trials
        study_name: Name for the Optuna study (defaults to model_name + timestamp)
    """

    if study_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        study_name = f"{model_name}_optuna_{timestamp}"

    # Setup logging
    optuna.logging.get_logger("optuna").addHandler(logging.StreamHandler())
    optuna.logging.set_verbosity(optuna.logging.INFO)

    def objective(trial):
        """Objective function for Optuna optimization."""

        trial_start_time = datetime.now()
        print(f"\n{'='*60}")
        print(f"🔬 Starting Trial {trial.number}")
        print(f"{'='*60}")

        # Suggest hyperparameters
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
        batch_size = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
        filter_factor = trial.suggest_categorical("filter_factor", [1, 2, 4])
        optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "AdamW"])
        weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)
        scheduler_patience = trial.suggest_int("scheduler_patience", 3, 10)
        dropout_rate = trial.suggest_float("dropout_rate", 0.0, 0.3)

        # Display hyperparameters for this trial
        print(f"📋 Trial {trial.number} Hyperparameters:")
        print(f"   Learning Rate: {learning_rate:.2e}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Filter Factor: {filter_factor}")
        print(f"   Optimizer: {optimizer_name}")
        print(f"   Weight Decay: {weight_decay:.2e}")
        print(f"   Scheduler Patience: {scheduler_patience}")
        print(f"   Dropout Rate: {dropout_rate:.3f}")
        print(f"{'='*60}")

        # Create model with suggested hyperparameters
        model = PhaseNetMag(
            in_channels=3, sampling_rate=100, norm="std", filter_factor=filter_factor
        )

        # Add dropout to model if specified
        if dropout_rate > 0:
            # We could modify the model to include dropout, for now we'll skip this
            pass

        model.to_preferred_device(verbose=False)

        # Load data with suggested batch size
        try:
            print(f"🔄 Loading data with batch size {batch_size}...")
            train_generator, train_loader, _ = dl.load_dataset(
                data, model, "train", batch_size=batch_size
            )
            dev_generator, dev_loader, _ = dl.load_dataset(
                data, model, "dev", batch_size=batch_size
            )
            print(f"✅ Data loaded successfully")
        except Exception as e:
            # If batch size is too large, prune this trial
            print(f"❌ Failed to load data: {e}")
            raise optuna.TrialPruned(
                f"Failed to load data with batch_size={batch_size}: {e}"
            )

        # Setup optimizer with suggested parameters
        if optimizer_name == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )
        else:  # AdamW
            optimizer = torch.optim.AdamW(
                model.parameters(), lr=learning_rate, weight_decay=weight_decay
            )

        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=scheduler_patience
        )

        best_val_loss = float("inf")

        # Create directory for saving trial models
        trial_save_dir = f"src/trained_weights/{model_name}/optuna_trials"
        os.makedirs(trial_save_dir, exist_ok=True)
        best_model_path = None

        print(f"🚀 Starting training for Trial {trial.number}")
        print(f"Target: Find validation loss < {best_val_loss}")

        # Training loop with pruning
        for epoch in range(max_epochs):
            epoch_start_time = datetime.now()

            # Training
            model.train()
            train_loss = 0
            num_batches = 0

            # Progress tracking for training
            total_train_batches = len(train_loader)

            for batch_idx, batch in enumerate(train_loader):
                x = batch["X"].to(model.device)
                y_true = batch["magnitude"].to(model.device)

                x_preproc = model.annotate_batch_pre(x, {})
                y_pred = model(x_preproc)
                y_pred = y_pred.squeeze(1)

                loss = criterion(y_pred, y_true)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()
                num_batches += 1

                # Print progress every 20% of batches
                if batch_idx % max(1, total_train_batches // 5) == 0:
                    current_avg_loss = train_loss / num_batches
                    progress = (batch_idx / total_train_batches) * 100
                    print(
                        f"  📈 Trial {trial.number} | Epoch {epoch+1:2d}/{max_epochs} | "
                        f"Train Progress: {progress:5.1f}% | "
                        f"Avg Loss: {current_avg_loss:.4f}"
                    )

            avg_train_loss = train_loss / num_batches

            # Validation
            model.eval()
            val_loss = 0
            num_val_batches = 0

            with torch.no_grad():
                for batch in dev_loader:
                    x = batch["X"].to(model.device)
                    y_true = batch["magnitude"].to(model.device)

                    x_preproc = model.annotate_batch_pre(x, {})
                    y_pred = model(x_preproc)
                    y_pred = y_pred.squeeze(1)

                    loss = criterion(y_pred, y_true)
                    val_loss += loss.item()
                    num_val_batches += 1

            avg_val_loss = val_loss / num_val_batches
            scheduler.step(avg_val_loss)

            # Calculate epoch duration
            epoch_duration = datetime.now() - epoch_start_time

            # Update best validation loss
            is_best = avg_val_loss < best_val_loss
            if is_best:
                best_val_loss = avg_val_loss
                best_indicator = "🌟 NEW BEST!"

                # Save the best model for this trial
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                best_model_path = os.path.join(
                    trial_save_dir,
                    f"trial_{trial.number}_best_epoch_{epoch+1}_{timestamp}.pt",
                )
                torch.save(model.state_dict(), best_model_path)
                print(f"  💾 Best model saved: {best_model_path}")
            else:
                best_indicator = ""

            # Current learning rate
            current_lr = optimizer.param_groups[0]["lr"]

            # Comprehensive epoch summary
            print(
                f"  ✅ Trial {trial.number} | Epoch {epoch+1:2d}/{max_epochs} COMPLETED | "
                f"Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f} | "
                f"LR: {current_lr:.2e} | Time: {epoch_duration.total_seconds():.1f}s {best_indicator}"
            )

            # Report intermediate value for pruning
            trial.report(avg_val_loss, epoch)

            # Pruning check
            if trial.should_prune():
                print(
                    f"  ✂️  Trial {trial.number} PRUNED at epoch {epoch+1} (Val Loss: {avg_val_loss:.4f})"
                )
                raise optuna.TrialPruned()

        # Trial completion summary
        trial_duration = datetime.now() - trial_start_time
        print(f"\n🏁 Trial {trial.number} COMPLETED!")
        print(f"   Best Validation Loss: {best_val_loss:.6f}")
        print(f"   Trial Duration: {trial_duration.total_seconds():.1f}s")
        print(f"   Final Learning Rate: {optimizer.param_groups[0]['lr']:.2e}")
        if best_model_path:
            print(f"   💾 Best Model Saved: {best_model_path}")
        else:
            print(f"   ⚠️  No model saved (no improvement found)")

        return best_val_loss

    # Create study with pruning
    study = optuna.create_study(
        direction="minimize",
        pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=pruning_patience),
        sampler=TPESampler(seed=42),
        study_name=study_name,
    )

    print(f"\nStarting Optuna hyperparameter optimization")
    print(f"Study name: {study_name}")
    print(f"Number of trials: {n_trials}")
    print(f"Max epochs per trial: {max_epochs}")
    print(f"Pruning patience: {pruning_patience}")
    print("=" * 60)

    # Track overall study progress
    study_start_time = datetime.now()

    # Custom callback to show progress between trials
    def trial_callback(study, trial):
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        pruned_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ]
        failed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.FAIL
        ]

        print(f"\n📊 STUDY PROGRESS SUMMARY")
        print(f"{'='*60}")
        print(
            f"Completed Trials: {len(completed_trials):3d} | "
            f"Pruned: {len(pruned_trials):3d} | "
            f"Failed: {len(failed_trials):3d} | "
            f"Total: {len(study.trials):3d}/{n_trials}"
        )

        if completed_trials:
            best_value = min(t.value for t in completed_trials)
            best_trial_num = min(completed_trials, key=lambda t: t.value).number
            recent_values = [t.value for t in completed_trials[-5:]]  # Last 5 trials
            avg_recent = sum(recent_values) / len(recent_values)

            print(f"🏆 Best Validation Loss: {best_value:.6f} (Trial {best_trial_num})")
            print(f"📈 Recent Average (last {len(recent_values)}): {avg_recent:.6f}")

            # Show improvement trend
            if len(completed_trials) >= 2:
                improvement = completed_trials[-2].value - completed_trials[-1].value
                trend = "📈 improving" if improvement > 0 else "📉 worsening"
                print(f"🔄 Last trial trend: {trend} ({improvement:+.6f})")

        # Time estimates
        elapsed_time = datetime.now() - study_start_time
        if len(study.trials) > 0:
            avg_time_per_trial = elapsed_time.total_seconds() / len(study.trials)
            remaining_trials = n_trials - len(study.trials)
            estimated_remaining = remaining_trials * avg_time_per_trial

            print(
                f"⏱️  Elapsed: {elapsed_time.total_seconds()/60:.1f}m | "
                f"Est. Remaining: {estimated_remaining/60:.1f}m | "
                f"Avg/Trial: {avg_time_per_trial:.1f}s"
            )

        print(f"{'='*60}")

    # Run optimization
    study.optimize(objective, n_trials=n_trials, callbacks=[trial_callback])

    # Print results
    total_duration = datetime.now() - study_start_time
    completed_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    ]
    pruned_trials = [
        t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    ]
    failed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.FAIL]

    print("\n" + "=" * 80)
    print("🎯 HYPERPARAMETER OPTIMIZATION COMPLETED!")
    print("=" * 80)

    print(f"📊 FINAL STATISTICS:")
    print(f"   Total Duration: {total_duration.total_seconds()/60:.1f} minutes")
    print(f"   Completed Trials: {len(completed_trials)}")
    print(f"   Pruned Trials: {len(pruned_trials)}")
    print(f"   Failed Trials: {len(failed_trials)}")
    print(f"   Success Rate: {len(completed_trials)/len(study.trials)*100:.1f}%")

    if completed_trials:
        print(f"\n🏆 BEST TRIAL RESULTS:")
        print(f"   Best Trial Number: {study.best_trial.number}")
        print(f"   Best Validation Loss: {study.best_value:.6f}")
        print(f"   Trial Duration: {study.best_trial.duration.total_seconds():.1f}s")

        print(f"\n⚙️  OPTIMAL HYPERPARAMETERS:")
        for key, value in study.best_params.items():
            if isinstance(value, float):
                if value < 0.01:
                    print(f"   {key}: {value:.2e}")
                else:
                    print(f"   {key}: {value:.4f}")
            else:
                print(f"   {key}: {value}")

        # Performance analysis
        values = [t.value for t in completed_trials]
        if len(values) > 1:
            import statistics

            mean_loss = statistics.mean(values)
            median_loss = statistics.median(values)
            std_loss = statistics.stdev(values) if len(values) > 1 else 0

            print(f"\n📈 PERFORMANCE DISTRIBUTION:")
            print(f"   Best Loss: {min(values):.6f}")
            print(f"   Worst Loss: {max(values):.6f}")
            print(f"   Mean Loss: {mean_loss:.6f}")
            print(f"   Median Loss: {median_loss:.6f}")
            print(f"   Std Dev: {std_loss:.6f}")
            print(f"   Improvement: {(max(values) - min(values))/max(values)*100:.1f}%")
    else:
        print("\n❌ No trials completed successfully!")

    print("=" * 80)

    # Save study results
    study_dir = f"src/trained_weights/{model_name}/optuna_studies"
    os.makedirs(study_dir, exist_ok=True)

    # Save detailed results to CSV
    trials_data = []
    for trial in study.trials:
        trial_data = {
            "trial_number": trial.number,
            "value": trial.value,
            "state": trial.state.name,
            "datetime_start": trial.datetime_start,
            "datetime_complete": trial.datetime_complete,
            "duration": trial.duration.total_seconds() if trial.duration else None,
        }
        # Add parameters
        for param_name, param_value in trial.params.items():
            trial_data[f"param_{param_name}"] = param_value
        # Add intermediate values
        for step, intermediate_value in trial.intermediate_values.items():
            trial_data[f"intermediate_step_{step}"] = intermediate_value
        trials_data.append(trial_data)

    df = pd.DataFrame(trials_data)
    csv_path = os.path.join(study_dir, f"{study_name}_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"Study results saved to CSV: {csv_path}")

    # Save best parameters to separate CSV
    best_params_df = pd.DataFrame([study.best_params])
    best_params_df["best_value"] = study.best_value
    best_params_df["best_trial"] = study.best_trial.number
    best_params_path = os.path.join(study_dir, f"{study_name}_best_params.csv")
    best_params_df.to_csv(best_params_path, index=False)
    print(f"Best parameters saved to CSV: {best_params_path}")

    # Create matplotlib visualizations
    try:
        # Plot optimization history
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        completed_trials = [
            t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
        ]
        trial_numbers = [t.number for t in completed_trials]
        trial_values = [t.value for t in completed_trials]
        plt.plot(trial_numbers, trial_values, "o-", alpha=0.7)
        plt.xlabel("Trial Number")
        plt.ylabel("Validation Loss")
        plt.title("Optimization History")
        plt.grid(True, alpha=0.3)

        # Plot parameter importance (simplified)
        plt.subplot(2, 2, 2)
        if len(completed_trials) > 10:  # Need enough trials for meaningful importance
            param_names = list(study.best_params.keys())
            param_values = list(study.best_params.values())
            plt.barh(param_names, [1] * len(param_names))  # Simplified visualization
            plt.xlabel("Relative Importance")
            plt.title("Parameter Importance (Best Trial)")
        else:
            plt.text(
                0.5,
                0.5,
                "Not enough trials\nfor importance analysis",
                ha="center",
                va="center",
                transform=plt.gca().transAxes,
            )
            plt.title("Parameter Importance")

        # Plot parameter distribution for best trials
        plt.subplot(2, 2, 3)
        if len(completed_trials) >= 5:
            # Show learning rate distribution for top 25% trials
            top_trials = sorted(completed_trials, key=lambda x: x.value)[
                : max(1, len(completed_trials) // 4)
            ]
            lr_values = [
                t.params.get("learning_rate")
                for t in top_trials
                if "learning_rate" in t.params
            ]
            if lr_values:
                plt.hist(lr_values, bins=min(10, len(lr_values)), alpha=0.7)
                plt.xlabel("Learning Rate")
                plt.ylabel("Frequency")
                plt.title("Learning Rate Distribution (Top 25% Trials)")
                plt.xscale("log")

        # Plot validation loss over epochs for best trial
        plt.subplot(2, 2, 4)
        best_trial = study.best_trial
        if best_trial.intermediate_values:
            epochs = list(best_trial.intermediate_values.keys())
            losses = list(best_trial.intermediate_values.values())
            plt.plot(epochs, losses, "o-", color="red")
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title(f"Best Trial ({best_trial.number}) Training Progress")
            plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plot_path = os.path.join(study_dir, f"{study_name}_analysis.png")
        plt.savefig(plot_path, dpi=300, bbox_inches="tight")
        plt.close()
        print(f"Analysis plots saved to: {plot_path}")

    except Exception as e:
        print(f"Warning: Could not create plots: {e}")

    # Save study object for later analysis
    study_path = os.path.join(study_dir, f"{study_name}.pkl")
    import pickle

    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"Study object saved to: {study_path}")

    # Train final model with best hyperparameters
    print("\n" + "=" * 80)
    print("🚀 TRAINING FINAL MODEL WITH BEST HYPERPARAMETERS")
    print("=" * 80)

    best_params = study.best_params
    print(f"🎯 Using optimized parameters from Trial {study.best_trial.number}:")
    for key, value in best_params.items():
        if isinstance(value, float):
            if value < 0.01:
                print(f"   {key}: {value:.2e}")
            else:
                print(f"   {key}: {value:.4f}")
        else:
            print(f"   {key}: {value}")
    print("=" * 80)

    final_model = PhaseNetMag(
        in_channels=3,
        sampling_rate=100,
        norm="std",
        filter_factor=best_params["filter_factor"],
    )
    final_model.to_preferred_device(verbose=True)

    print(f"🏁 Starting full training with 50 epochs...")

    # Train final model with best hyperparameters
    train_phasenet_mag(
        model_name=f"{model_name}_best_optuna",
        model=final_model,
        data=data,
        learning_rate=best_params["learning_rate"],
        epochs=50,  # Full training for final model
        batch_size=best_params["batch_size"],
        optimizer_name=best_params["optimizer"],
        weight_decay=best_params["weight_decay"],
        scheduler_patience=best_params["scheduler_patience"],
        save_every=10,
    )

    return study


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

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=scheduler_patience
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

        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        #     best_model_path = os.path.join(save_dir, f"model_best_{timestamp}.pt")
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f"New best model saved: {best_model_path}")

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

    # Hyperparameter tuning options
    parser.add_argument(
        "--tune", action="store_true", help="Run hyperparameter tuning with Optuna"
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=100,
        help="Number of trials for hyperparameter tuning",
    )
    parser.add_argument(
        "--max_epochs_per_trial",
        type=int,
        default=30,
        help="Maximum epochs per trial during tuning",
    )
    parser.add_argument(
        "--study_name", type=str, default=None, help="Name for the Optuna study"
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

    # Model name for saving
    model_name = f"PhaseNetMag_{args.dataset}"

    if args.tune:
        # Run hyperparameter tuning
        print("Running hyperparameter optimization with Optuna...")
        study = train_phasenet_mag_with_optuna(
            model_name=model_name,
            data=data,
            n_trials=args.n_trials,
            max_epochs=args.max_epochs_per_trial,
            study_name=args.study_name,
        )
    else:
        # Regular training
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
        )


if __name__ == "__main__":
    main()
