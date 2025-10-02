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
    Main script for both original PhaseNet tutorials and new magnitude prediction

    Usage:
        # Original PhaseNet tutorial
        python src/my_project/main.py
    """

    parser = argparse.ArgumentParser(
        description="PhaseNet and Magnitude Prediction Workflows"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="ETHZ",
        choices=["ETHZ", "STEAD", "GEOFON"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="tutorial",
        choices=["tutorial"],
        help="Workflow mode",
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
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    print(f"Dataset loaded: {len(data)} samples")

    # Run appropriate workflow
    if args.mode == "tutorial":
        tutorial_tests(data, model_path=args.model_path)
    else:
        print(f"Unknown mode: {args.mode}")

    print("Script completed!")
