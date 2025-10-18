import argparse

import seisbench.data as sbd
import seisbench.models as sbm
from seisbench.data import BenchmarkDataset

from my_project.tutorial.tutorial import (
    test_load_data,
    test_generator,
    train_phasenet,
    evaluate_phase_model,
)
from my_project.models.phasenet_mag.train import train_phasenet_mag
from my_project.models.phasenet_mag.evaluate import evaluate_phasenet_mag
from my_project.models.phasenet_mag.model import PhaseNetMag

from my_project.models.phasenetLSTM.model import PhaseNetLSTM
from my_project.models.phasenetLSTM.modelv2 import PhaseNetConvLSTM
from my_project.models.phasenetLSTM.train import (
    train_phasenetLSTM,
    train_phasenet_lstm_model,
    train_phasenet_lstm_default,
)

from my_project.models.AMAG_v2.model import MagnitudeNet
from my_project.models.AMAG_v2.train import train_magnitude_net

from my_project.utils.utils import plot_training_history


def mag_train_phasenet(data: BenchmarkDataset, epochs: int = 5):
    """Train PhaseNetMag for magnitude regression"""
    print(f"Training PhaseNetMag on {data.name} for {epochs} epochs...")

    # Create model
    model = PhaseNetMag(in_channels=3, sampling_rate=100, norm="std", filter_factor=1)
    model.to_preferred_device(verbose=True)

    # Model name for saving
    model_name = f"PhaseNetMag_{data.name}"

    # Train model
    train_phasenet_mag(
        model_name=model_name,
        model=model,
        data=data,
        learning_rate=1e-4,
        epochs=epochs,
        batch_size=256,
        save_every=5,
    )


def magnitude_evaluate(
    data: BenchmarkDataset, model_path: str, plot_examples: bool = False
):
    """Evaluate PhaseNetMag for magnitude regression"""
    print(f"Evaluating PhaseNetMag on {data.name}...")

    # Create model
    model = PhaseNetMag(in_channels=3, sampling_rate=100, norm="std", filter_factor=1)

    model.to_preferred_device(verbose=True)
    print(f"Model moved to device: {model.device}")

    # Evaluate model
    results = evaluate_phasenet_mag(
        model=model,
        model_path=model_path,
        data=data,
        batch_size=256,
        plot_examples=plot_examples,
        num_examples=5,
    )

    return results


def tutorial_test_load_and_generator(data: BenchmarkDataset):
    """Test data loading and generator functionality"""
    print("\n" + "=" * 50)
    print("TUTORIAL: TESTING DATA LOADING AND GENERATOR")
    print("=" * 50)

    test_load_data()
    test_generator()

    print("Data loading and generator tests completed!")


def tutorial_train_phasenet(data: BenchmarkDataset, epochs: int = 5):
    """Train PhaseNet model for tutorial"""
    print("\n" + "=" * 50)
    print("TUTORIAL: TRAINING PHASENET")
    print("=" * 50)

    # Load standard PhaseNet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    model.to_preferred_device(verbose=True)

    model_name = f"PhaseNet_{data.name}"

    train_phasenet(
        model=model, model_name=model_name, data=data, learning_rate=1e-2, epochs=epochs
    )

    print(f"PhaseNet training completed! Model saved as: {model_name}")


def tutorial_evaluate_phasenet(data: BenchmarkDataset, model_path: str):
    """Evaluate PhaseNet model for tutorial"""
    print("\n" + "=" * 50)
    print("TUTORIAL: EVALUATING PHASENET")
    print("=" * 50)

    if not model_path:
        print("Error: --model_path is required for tutorial evaluation")
        print("Usage: --model_path path/to/model.pt")
        exit(1)

    # model = sbm.PhaseNet(
    #     phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    # )

    # Load PhaseNetConvLSTM for evaluation
    model = PhaseNetConvLSTM()
    model.to_preferred_device(verbose=True)

    evaluate_phase_model(model=model, model_path=model_path, data=data)

    print("PhaseNet evaluation completed!")


def tutorial_tests(data: BenchmarkDataset, model_path: str = ""):
    """Original PhaseNet tutorial tests (deprecated - use individual tutorial functions)"""
    print("\n" + "=" * 50)
    print("TUTORIAL: RUNNING ALL TESTS (DEPRECATED)")
    print("=" * 50)
    print(
        "Warning: This function is deprecated. Consider using individual tutorial functions:"
    )
    print("  - tutorial_test_load_and_generator")
    print("  - tutorial_train_phasenet")
    print("  - tutorial_evaluate_phasenet")
    print("=" * 50)

    # # Run evaluation only (as in original)
    # model = PhaseNetConvLSTM()
    # model.to_preferred_device(verbose=True)

    # evaluate_phase_model(model=model, model_path=model_path, data=data)


def train_phasenet_lstm(
    data: BenchmarkDataset,
    epochs: int = 50,
    filter_factor: int = 1,
    lstm_hidden_size: int = None,
    lstm_num_layers: int = 1,
    lstm_bidirectional: bool = True,
    learning_rate: float = 1e-3,
    batch_size: int = 256,
    num_workers: int = 4,
):
    """Train PhaseNet-LSTM model with configurable parameters"""
    print("\n" + "=" * 50)
    print("TRAINING PHASENET-LSTM")
    print("=" * 50)

    # Train the model using the abstracted function
    model, best_loss = train_phasenet_lstm_model(
        data=data,
        epochs=epochs,
        filter_factor=filter_factor,
        lstm_hidden_size=lstm_hidden_size,
        lstm_num_layers=lstm_num_layers,
        lstm_bidirectional=lstm_bidirectional,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    return model, best_loss


def train_amag_v2(data: BenchmarkDataset, epochs: int = 50):
    """Train AMAG_v2 MagnitudeNet model"""
    print("\n" + "=" * 50)
    print("TRAINING AMAG_V2 MAGNITUDENET")
    print("=" * 50)

    # Initialize model with default parameters
    model = MagnitudeNet(
        in_channels=3,
        sampling_rate=100,
        filter_factor=1,
        lstm_hidden=128,
        lstm_layers=2,
        dropout=0.2,
    )

    # Train the model
    history = train_magnitude_net(
        model_name="magnitudenet_test",
        model=model,
        data=data,
        learning_rate=1e-3,
        epochs=epochs,
        batch_size=256,
        optimizer_name="AdamW",
        weight_decay=1e-5,
        scheduler_patience=5,
        save_every=5,
        gradient_clip=1.0,
    )

    return history


if __name__ == "__main__":
    """
    Main script for PhaseNet tutorials and magnitude prediction workflows

    Usage:
        # Tutorial: Test data loading and generator
        python src/my_project/main.py --mode tutorial_test_load_and_generator --dataset ETHZ

        # Tutorial: Train PhaseNet model
        python src/my_project/main.py --mode tutorial_train_phasenet --dataset ETHZ --epochs 5

        # Tutorial: Evaluate PhaseNet model
        python src/my_project/main.py --mode tutorial_evaluate_phasenet --dataset ETHZ --model_path path/to/model.pt

        # Train PhaseNet-LSTM model
        python src/my_project/main.py --mode train_phasenetLSTM --dataset ETHZ --epochs 5

        # Train magnitude model (PhaseNetMag)
        python src/my_project/main.py --mode magnitude_train --dataset ETHZ --epochs 5

        # Train AMAG_v2 magnitude model
        python src/my_project/main.py --mode train_AMAG_v2 --dataset ETHZ --epochs 5

        # Evaluate magnitude model
        python src/my_project/main.py --mode magnitude_eval --dataset ETHZ --model_path path/to/model.pt

        # Evaluate magnitude model with plots
        python src/my_project/main.py --mode magnitude_eval --dataset ETHZ --model_path path/to/model.pt --plot

        # Plot training history (saves PNG, no display)
        python src/my_project/main.py --mode plot_history --model_path path/to/training_history.pt

        # Plot training history (saves PNG and displays plot)
        python src/my_project/main.py --mode plot_history --model_path path/to/training_history.pt --plot

        # Original PhaseNet tutorial (deprecated - use individual tutorial functions)
        python src/my_project/main.py --mode tutorial --model_path path/to/model.pt
    """

    parser = argparse.ArgumentParser(
        description="PhaseNet and Magnitude Prediction Workflows"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETHZ",
        choices=["ETHZ", "STEAD", "GEOFON", "MLAAPDE"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to model checkpoint file (.pt) or training history file (.pt). Required for tutorial_evaluate_phasenet, magnitude_eval, and plot_history modes",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tutorial_test_load_and_generator",
        choices=[
            "tutorial",
            "tutorial_test_load_and_generator",
            "tutorial_train_phasenet",
            "tutorial_evaluate_phasenet",
            "train_phasenetLSTM",
            "train_AMAG_v2",
            "magnitude_train",
            "magnitude_eval",
            "plot_history",
        ],
        help="Workflow mode",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Display plots on screen and save PNG files (for magnitude_eval and plot_history modes)",
    )

    args = parser.parse_args()

    if args.mode != "plot_history":
        # Load dataset
        print("\n" + "=" * 50)
        print("LOADING DATA")
        print("=" * 50)
        if args.dataset == "ETHZ":
            data = sbd.ETHZ(sampling_rate=100)
        elif args.dataset == "STEAD":
            data = sbd.STEAD(sampling_rate=100)
        elif args.dataset == "GEOFON":
            data = sbd.GEOFON(sampling_rate=100)
        elif args.dataset == "MLAAPDE":
            data = sbd.MLAAPDE(sampline_rate=100)
        else:
            raise ValueError(f"Unknown dataset: {args.dataset}")

        print(f"{data.name} dataset loaded: {len(data)} samples")

    # Run appropriate workflow
    if args.mode == "tutorial":
        # if not args.model_path:
        #     print("Error: --model_path is required for tutorial mode")
        #     exit(1)
        tutorial_tests(data, model_path=args.model_path)
    elif args.mode == "tutorial_test_load_and_generator":
        tutorial_test_load_and_generator(data)
    elif args.mode == "tutorial_train_phasenet":
        tutorial_train_phasenet(data, epochs=args.epochs)
    elif args.mode == "tutorial_evaluate_phasenet":
        tutorial_evaluate_phasenet(data, model_path=args.model_path)
    elif args.mode == "train_phasenetLSTM":
        train_phasenet_lstm(data, epochs=args.epochs)
    elif args.mode == "train_AMAG_v2":
        train_amag_v2(data, epochs=args.epochs)
    elif args.mode == "magnitude_train":
        mag_train_phasenet(data, epochs=args.epochs)
    elif args.mode == "magnitude_eval":
        if not args.model_path:
            print("Error: --model_path is required for magnitude_eval mode")
            exit(1)
        magnitude_evaluate(data, model_path=args.model_path, plot_examples=args.plot)
    elif args.mode == "plot_history":
        if not args.model_path:
            print("Error: --model_path is required for plot_history mode")
            print("Usage: --model_path path/to/training_history_*.pt")
            exit(1)
        plot_training_history(args.model_path, show_plot=args.plot)
    else:
        print(f"Unknown mode: {args.mode}")

    print("=" * 50 + "\n")
