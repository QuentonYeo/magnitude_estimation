import seisbench
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.data import BenchmarkDataset
from my_project.loaders.magnitude_labellers import MagnitudeLabeller

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import torch
import random
from sklearn.metrics import (
    mean_squared_error,
    r2_score,
    mean_absolute_error,
    precision_score,
    recall_score,
    f1_score,
)
from scipy.signal import find_peaks

from my_project.loaders import data_loader as dl
from my_project.utils import utils

phase_dict = {
    "trace_p_arrival_sample": "P",
    "trace_pP_arrival_sample": "P",
    "trace_P_arrival_sample": "P",
    "trace_P1_arrival_sample": "P",
    "trace_Pg_arrival_sample": "P",
    "trace_Pn_arrival_sample": "P",
    "trace_PmP_arrival_sample": "P",
    "trace_pwP_arrival_sample": "P",
    "trace_pwPm_arrival_sample": "P",
    "trace_s_arrival_sample": "S",
    "trace_S_arrival_sample": "S",
    "trace_S1_arrival_sample": "S",
    "trace_Sg_arrival_sample": "S",
    "trace_SmS_arrival_sample": "S",
    "trace_Sn_arrival_sample": "S",
}


def test_load_data():
    data = sbd.ETHZ(sampling_rate=100)
    print(data)

    print("Cache root:", seisbench.cache_root)
    print("Contents:", os.listdir(seisbench.cache_root))
    print("datasets:", os.listdir(seisbench.cache_root / "datasets"))
    print(
        "dummydataset:", os.listdir(seisbench.cache_root / "datasets" / "dummydataset")
    )

    dummy_from_disk = sbd.WaveformDataset(
        seisbench.cache_root / "datasets" / "dummydataset"
    )
    print(dummy_from_disk)
    print(data.metadata)

    waveforms = data.get_waveforms(3)
    print("waveforms.shape:", waveforms.shape)

    plt.plot(waveforms.T)

    # # Filtering the dataset
    # mask = (
    #     data.metadata["source_magnitude"] > 2.5
    # )  # Only select events with magnitude above 2.5
    # data.filter(mask)

    # print(data)
    # print(data.metadata)

    magnitudes = data.metadata["source_magnitude"]
    indicies = data.metadata["index"]

    print(indicies)

    # Plot histogram: frequency vs magnitude bins (0-10, step 0.5)
    bins = [x * 1 for x in range(11)]  # 0, 0.5, ..., 10
    plt.figure()
    plt.hist(magnitudes, bins=bins, edgecolor="black")
    plt.xlabel("Magnitude")
    plt.ylabel("Frequency")
    plt.title("Magnitude Distribution in DummyDataset")
    plt.xticks(bins)
    plt.show()


def test_generator():
    data = sbd.ETHZ(sampling_rate=100)

    generator = sbg.GenericGenerator(data)

    print(generator)

    import matplotlib.pyplot as plt

    print("Number of examples:", len(generator))
    sample = generator[200]
    print("Example:", sample)

    plt.plot(sample["X"].T)

    generator.augmentation(sbg.RandomWindow(windowlen=6000))
    generator.augmentation(sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1))
    generator.augmentation(
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
    )
    generator.add_augmentations([MagnitudeLabeller(phase_dict=phase_dict)])

    print(generator)

    utils.plot_samples(generator, single=True)


def train_phasenet(
    model_name: str,
    model: sbm.WaveformModel,
    data: BenchmarkDataset,
    learning_rate=1e-2,
    epochs=5,
):
    train_generator, train_loader, _ = dl.load_dataset(data, model, "train")
    dev_generator, dev_loader, _ = dl.load_dataset(data, model, "dev")

    print("Data successfully loaded")

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    def loss_fn(y_pred, y_true, eps=1e-5):
        # vector cross entropy loss
        h = y_true * torch.log(y_pred + eps)
        h = h.mean(-1).sum(
            -1
        )  # Mean along sample dimension and sum along pick dimension
        h = h.mean()  # Mean over batch axis
        return -h

    def train_loop(dataloader):
        size = len(dataloader.dataset)
        for batch_id, batch in enumerate(dataloader):
            # Compute prediction and loss
            x = batch["X"].to(model.device)
            x_preproc = model.annotate_batch_pre(
                x, {}
            )  # Remove mean and normalize amplitude
            pred = model(x_preproc)
            loss = loss_fn(pred, batch["y"].float().to(model.device))

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_id % 5 == 0:
                loss, current = loss.item(), batch_id * batch["X"].shape[0]
                print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    def test_loop(dataloader):
        num_batches = len(dataloader)
        test_loss = 0

        model.eval()  # close the model for evaluation

        with torch.no_grad():
            for batch in dataloader:
                x = batch["X"].to(model.device)
                x_preproc = model.annotate_batch_pre(
                    x, {}
                )  # Remove mean and normalize amplitude
                pred = model(x_preproc)
                test_loss += loss_fn(pred, batch["y"].float().to(model.device)).item()

        model.train()  # re-open model for training stage

        test_loss /= num_batches
        print(f"Test avg loss: {test_loss:>8f} \n")

    # Train model with checkpoint each 5 epochs
    save_dir = f"src/trained_weights/{model_name}"
    os.makedirs(save_dir, exist_ok=True)

    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loop(train_loader)
        test_loop(dev_loader)
        if (t + 1) % 5 == 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"model_epoch_{t+1}_{timestamp}.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to {save_path}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"model_final_{timestamp}.pt")
    torch.save(model.state_dict(), save_path)
    print(f"Training complete, model saved to {save_path}")


def calculate_pick_metrics(true_picks, pred_picks, tolerance=30):
    """Calculate precision, recall, F1 score, and MAE for phase picks."""
    if not true_picks and not pred_picks:
        return {"precision": 1.0, "recall": 1.0, "f1_score": 1.0, "mae": 0.0}
    if not true_picks:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "mae": np.nan}
    if not pred_picks:
        return {"precision": 0.0, "recall": 0.0, "f1_score": 0.0, "mae": np.nan}

    # Match predictions to true picks within tolerance
    matched_errors = []
    used_true = set()

    for pred_pick in pred_picks:
        best_match = None
        best_distance = float("inf")

        for i, true_pick in enumerate(true_picks):
            if i not in used_true:
                distance = abs(pred_pick - true_pick)
                if distance <= tolerance and distance < best_distance:
                    best_match = i
                    best_distance = distance

        if best_match is not None:
            matched_errors.append(best_distance)
            used_true.add(best_match)

    # Calculate metrics
    tp = len(matched_errors)
    fp = len(pred_picks) - tp
    fn = len(true_picks) - tp

    precision = tp / len(pred_picks)
    recall = tp / len(true_picks)
    f1_score = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )
    mae = np.mean(matched_errors) if matched_errors else np.nan

    return {"precision": precision, "recall": recall, "f1_score": f1_score, "mae": mae}


def find_pick_times(data, threshold, data_format="channels_first"):
    """
    Find the first pick time above threshold for each sample.

    Args:
        data: Array of shape [n_samples, channels, time] or [n_samples, time, channels]
        threshold: Threshold for detecting picks
        data_format: "channels_first" if shape is [n_samples, channels, time]

    Returns:
        List of pick times for P and S channels (None if no pick found)
    """
    p_picks = []
    s_picks = []

    for i, sample in enumerate(data):
        if data_format == "channels_first":
            # Shape is [channels, time] - transpose to [time, channels]
            sample = sample.T

        # Find first point above threshold for P and S channels
        p_channel = sample[:, 0]
        s_channel = sample[:, 1]

        # Find pick time - use peak detection for better accuracy
        if threshold >= 1.0:
            # For true labels, find center of regions where value = 1.0
            p_indices = np.where(p_channel >= threshold)[0]
            s_indices = np.where(s_channel >= threshold)[0]

            # Use center of the pick region instead of first point
            p_pick = int(np.mean(p_indices)) if len(p_indices) > 0 else None
            s_pick = int(np.mean(s_indices)) if len(s_indices) > 0 else None
        else:
            # For predictions, find the peak (maximum value point)
            p_indices = np.where(p_channel > threshold)[0]
            s_indices = np.where(s_channel > threshold)[0]

            if len(p_indices) > 0:
                # Find the peak within the above-threshold region
                p_values = p_channel[p_indices]
                p_peak_idx = p_indices[np.argmax(p_values)]
                p_pick = p_peak_idx
            else:
                p_pick = None

            if len(s_indices) > 0:
                # Find the peak within the above-threshold region
                s_values = s_channel[s_indices]
                s_peak_idx = s_indices[np.argmax(s_values)]
                s_pick = s_peak_idx
            else:
                s_pick = None

        p_picks.append(p_pick)
        s_picks.append(s_pick)

    return p_picks, s_picks


def calculate_pick_mae(true_picks, pred_picks):
    """Calculate MAE between true and predicted pick times, ignoring samples with missing picks."""
    errors = []

    for true_pick, pred_pick in zip(true_picks, pred_picks):
        # Only calculate error if both picks exist
        if true_pick is not None and pred_pick is not None:
            errors.append(abs(true_pick - pred_pick))

    # Return 0 if no valid comparisons (instead of NaN)
    return np.mean(errors) if errors else 0.0


def evaluate_phase_model(
    model: sbm.WaveformModel, model_path: str, data: sbd.BenchmarkDataset
):
    # Load model weights
    state_dict = torch.load(model_path, map_location=model.device, weights_only=True)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {model_path}")

    test_generator, _, _ = dl.load_dataset(data=data, model=model, type="test")

    # Visual check: plot random sample
    sample = test_generator[np.random.randint(len(test_generator))]
    fig, axs = plt.subplots(3, 1, figsize=(15, 10), sharex=True)
    axs[0].plot(sample["X"].T)
    axs[1].plot(sample["y"].T)

    model.eval()
    with torch.no_grad():
        x = torch.tensor(sample["X"]).to(model.device).unsqueeze(0)
        x_preproc = model.annotate_batch_pre(x, {})
        pred = model(x_preproc)[0].cpu().numpy()
    axs[2].plot(pred.T)

    # Evaluate all test samples
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for i in tqdm(range(len(test_generator)), desc="Evaluating"):
            sample = test_generator[i]
            x = torch.tensor(sample["X"]).to(model.device).unsqueeze(0)
            x_preproc = model.annotate_batch_pre(x, {})
            pred = model(x_preproc)[0].cpu().numpy()
            label = sample["y"]

            all_preds.append(pred)
            all_labels.append(label)

    # Convert to numpy arrays and flatten
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Overall waveform metrics (existing)
    y_true = all_labels.reshape(-1, all_labels.shape[-1])
    y_pred = all_preds.reshape(-1, all_preds.shape[-1])

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    # Phase-specific metrics using sklearn
    # Extract P and S channels
    p_true = y_true[:, 0]  # P channel
    s_true = y_true[:, 1]  # S channel
    p_pred = y_pred[:, 0]  # P channel
    s_pred = y_pred[:, 1]  # S channel

    # Convert to binary classifications using thresholds
    p_true_binary = (p_true > 0.5).astype(int)
    s_true_binary = (s_true > 0.5).astype(int)
    p_pred_binary = (p_pred > 0.3).astype(int)
    s_pred_binary = (s_pred > 0.3).astype(int)

    # Calculate metrics using sklearn
    p_precision = precision_score(p_true_binary, p_pred_binary, zero_division=0)
    p_recall = recall_score(p_true_binary, p_pred_binary, zero_division=0)
    p_f1 = f1_score(p_true_binary, p_pred_binary, zero_division=0)

    s_precision = precision_score(s_true_binary, s_pred_binary, zero_division=0)
    s_recall = recall_score(s_true_binary, s_pred_binary, zero_division=0)
    s_f1 = f1_score(s_true_binary, s_pred_binary, zero_division=0)

    # Calculate MAE based on pick times
    # Find pick times for true labels (where label = 1) and predictions (first point > threshold)
    true_p_picks, true_s_picks = find_pick_times(
        all_labels, threshold=1.0, data_format="channels_first"
    )
    pred_p_picks, pred_s_picks = find_pick_times(
        all_preds, threshold=0.3, data_format="channels_first"
    )

    p_mae = calculate_pick_mae(true_p_picks, pred_p_picks)
    s_mae = calculate_pick_mae(true_s_picks, pred_s_picks)

    # Convert MAE from samples to seconds (sampling rate = 100 Hz)
    sampling_rate = 100  # Hz
    p_mae_seconds = p_mae / sampling_rate
    s_mae_seconds = s_mae / sampling_rate

    # Print results
    print(f"\nOverall Metrics:")
    print(f"MSE: {mse:.6f}, RMSE: {rmse:.6f}, R²: {r2:.6f}")

    print(f"\nP-wave Metrics:")
    print(f"Precision: {p_precision:.4f}, Recall: {p_recall:.4f}")
    print(f"F1-score: {p_f1:.4f}, MAE: {p_mae:.2f} samples ({p_mae_seconds:.3f} sec)")

    print(f"\nS-wave Metrics:")
    print(f"Precision: {s_precision:.4f}, Recall: {s_recall:.4f}")
    print(f"F1-score: {s_f1:.4f}, MAE: {s_mae:.2f} samples ({s_mae_seconds:.3f} sec)")

    plt.show()
