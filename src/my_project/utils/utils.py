import os
import csv
import matplotlib.pyplot as plt

from seisbench.data import BenchmarkDataset


def dump_metadata_to_csv(sbd: BenchmarkDataset, filename="metadata.csv"):
    """
    Dumps metadata values from the sbd object to a CSV file in /metadata/.
    Args:
        sbd: The seisbench dataset object with metadata attribute (assumed to be a DataFrame or dict-like).
        filename: Name of the CSV file to write (default: metadata.csv).
    """
    metadata_dir = os.path.join(os.path.dirname(__file__), "../metadata")
    os.makedirs(metadata_dir, exist_ok=True)
    csv_path = os.path.join(metadata_dir, filename)

    metadata = getattr(sbd, "metadata", None)
    if metadata is None:
        raise ValueError("sbd object does not have a 'metadata' attribute.")

    # If pandas DataFrame
    try:
        import pandas as pd

        if isinstance(metadata, pd.DataFrame):
            metadata.to_csv(csv_path, index=False)
            return csv_path
    except ImportError:
        pass

    # If dict-like
    if isinstance(metadata, dict):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["key", "value"])
            for k, v in metadata.items():
                writer.writerow([k, v])
        return csv_path

    # If list of dicts
    if isinstance(metadata, list) and all(isinstance(item, dict) for item in metadata):
        with open(csv_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=metadata[0].keys())
            writer.writeheader()
            writer.writerows(metadata)
        return csv_path

    raise TypeError(
        "Unsupported metadata format. Must be pandas DataFrame, dict, or list of dicts."
    )


def plot_magnitude_distribution(data: BenchmarkDataset) -> None:
    magnitudes = data.metadata["source_magnitude"]

    # Plot histogram: frequency vs magnitude bins (0-9, step 0.5)
    bins = [x * 0.5 for x in range(19)]  # 0, 0.5, ..., 10
    plt.figure()
    plt.hist(magnitudes, bins=bins, edgecolor="black")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.title("Magnitude Distribution in DummyDataset")
    plt.xticks(bins)
    plt.show()
