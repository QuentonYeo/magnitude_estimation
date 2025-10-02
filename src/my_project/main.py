import argparse

import seisbench.data as sbd
import seisbench.models as sbm
from seisbench.data import BenchmarkDataset

from my_project.tutorial.tutorial import (
    test_load_data,
    test_generator,
    train_phasenet,
    evaluate_phasenet,
)
from my_project.models.phasenet_mag.train import train_phasenet_mag
from my_project.models.phasenet_mag.evaluate import evaluate_phasenet_mag
from my_project.models.phasenet_mag.model import PhaseNetMag

from my_project.utils.utils import plot_training_history


def magnitude_train(data: BenchmarkDataset, epochs: int = 5):
    """Train PhaseNetMag for magnitude regression"""
    print(f"Training PhaseNetMag on {data.name} for {epochs} epochs...")

    # Create model
    model = PhaseNetMag(in_channels=3, sampling_rate=100, norm="std", filter_factor=1)

    model.to_preferred_device(verbose=True)
    print(f"Model moved to device: {model.device}")

    # Model name for saving
    model_name = f"PhaseNetMag_{data.name}"

    # Train model
    train_phasenet_mag(
        model_name=model_name,
        model=model,
        data=data,
        learning_rate=1e-3,
        epochs=epochs,
        batch_size=256,
        save_every=5,
    )


def magnitude_evaluate(data: BenchmarkDataset, model_path: str):
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
        plot_examples=True,
        num_examples=5,
    )

    return results


def tutorial_tests(data: BenchmarkDataset, model_path: str = ""):
    """Original PhaseNet tutorial tests"""
    # test_load_data()
    # test_generator()

    # Load standard Phasenet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    model.to_preferred_device(verbose=True)

    # model_name = f"PhaseNet_{data.name}"

    # train_phasenet(
    #     model=model, model_name=model_name, data=data, learning_rate=1e-2, epochs=5
    # )

    # run with: uv run src/my_project/main.py --model_path <root/path-to-model>
    evaluate_phasenet(model=model, model_path=model_path, data=data)


if __name__ == "__main__":
    """
    Main script for PhaseNet tutorials and magnitude prediction workflows

    Usage:
        # Train magnitude model
        python src/my_project/main.py --mode magnitude_train --dataset ETHZ --epochs 5

        # Evaluate magnitude model
        python src/my_project/main.py --mode magnitude_eval --dataset ETHZ --model_path path/to/model.pt

        # Plot training history
        python src/my_project/main.py --mode plot_history --model_path path/to/training_history.pt

        # Original PhaseNet tutorial
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
        help="Path to model checkpoint file (.pt) or training history file (.pt). Required for tutorial, magnitude_eval, and plot_history modes",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tutorial",
        choices=["tutorial", "magnitude_train", "magnitude_eval", "plot_history"],
        help="Workflow mode",
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of epochs for training"
    )

    args = parser.parse_args()

    # Load dataset
    print(f"Loading {args.dataset} dataset...")
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
        if not args.model_path:
            print("Error: --model_path is required for tutorial mode")
            exit(1)
        tutorial_tests(data, model_path=args.model_path)
    elif args.mode == "magnitude_train":
        magnitude_train(data, epochs=args.epochs)
    elif args.mode == "magnitude_eval":
        if not args.model_path:
            print("Error: --model_path is required for magnitude_eval mode")
            exit(1)
        magnitude_evaluate(data, model_path=args.model_path)
    elif args.mode == "plot_history":
        if not args.model_path:
            print("Error: --model_path is required for plot_history mode")
            print("Usage: --model_path path/to/training_history_*.pt")
            exit(1)
        plot_training_history(args.model_path)
    else:
        print(f"Unknown mode: {args.mode}")

    print("Script completed!")
