import seisbench
import seisbench.data as sbd
import seisbench.generate as sbg
import seisbench.models as sbm

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm
import torch
from sklearn.metrics import mean_squared_error, r2_score

from my_project.loaders import data_loader as dl


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
    plt.show()

    generator.augmentation(sbg.RandomWindow(windowlen=3000))
    generator.augmentation(sbg.Normalize(detrend_axis=-1, amp_norm_axis=-1))

    print(generator)

    sample = generator[200]
    print("Example:", sample)

    plt.plot(sample["X"].T)
    plt.show()

    generator.augmentation(
        sbg.ProbabilisticLabeller(
            label_columns=["trace_P1_arrival_sample"], sigma=50, dim=-2
        )
    )

    print(generator)

    sample = generator[200]
    print("Sample keys:", sample.keys())

    fig = plt.figure(figsize=(10, 7))
    axs = fig.subplots(2, 1)
    axs[0].plot(sample["X"].T)
    axs[1].plot(sample["y"].T)

    plt.show()


def train_phasenet(model_name: str, learning_rate=1e-2, epochs=5):
    # Load standard Phasenet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    model.to_preferred_device(verbose=True)

    train_generator, train_loader, _ = dl.load_dataset(model, "train")
    dev_generator, dev_loader, _ = dl.load_dataset(model, "dev")

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


def evaluate_phasenet():
    parser = argparse.ArgumentParser(description="Evaluate PhaseNet model")
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to model checkpoint file"
    )
    args = parser.parse_args()

    # Load standard Phasenet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    model.to_preferred_device(verbose=True)

    # Load weights from specified checkpoint
    state_dict = torch.load(args.model_path, map_location=model.device)
    model.load_state_dict(state_dict)
    print(f"Loaded model weights from {args.model_path}")

    test_generator, _, _ = dl.load_dataset(model, "test")

    # Visual check: plot a random test sample and prediction
    sample = test_generator[np.random.randint(len(test_generator))]
    fig = plt.figure(figsize=(15, 10))
    axs = fig.subplots(
        3, 1, sharex=True, gridspec_kw={"hspace": 0, "height_ratios": [3, 1, 1]}
    )
    axs[0].plot(sample["X"].T)
    axs[1].plot(sample["y"].T)

    model.eval()
    with torch.no_grad():
        x = torch.tensor(sample["X"]).to(model.device).unsqueeze(0)
        x_preproc = model.annotate_batch_pre(x, {})
        pred = model(x_preproc)[0].cpu().numpy()
    axs[2].plot(pred.T)
    plt.show()

    # Evaluate on all test samples with progress bar
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

    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    # Flatten for metrics (samples, features)
    y_true = all_labels.reshape(-1, all_labels.shape[-1])
    y_pred = all_preds.reshape(-1, all_preds.shape[-1])

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)

    print(f"Test MSE: {mse:.6f}")
    print(f"Test RMSE: {rmse:.6f}")
    print(f"Test R^2 Score: {r2:.6f}")
