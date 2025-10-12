import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from seisbench.models.base import WaveformModel
from seisbench.data.base import BenchmarkDataset
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

from my_project.utils.utils import plot_magnitude_distribution, dump_metadata_to_csv
from my_project.loaders.magnitude_labellers import MagnitudeLabellerPhaseNet
from my_project.models.phasenet_mag.model import PhaseNetMag
from my_project.models.AMAG.model import AMAG

# Only training for S and P picks, map the labels
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


def get_magnitude_augmentation(windowlen=3001):
    """Define training and validation generator for magnitude regression only:

    - Long window around pick
    - Random window of specified length (default 3001 for PhaseNet, 600 for AMAG)
    - Change datatype to float32 for pytorch
    - Magnitude labeller only (no phase labels)

    Args:
        windowlen (int): Length of the random window. Default is 3001 for PhaseNet.
                        Use 600 for AMAG model.
    """
    augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=3000,
            windowlen=6000,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(windowlen=windowlen, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        MagnitudeLabellerPhaseNet(phase_dict=phase_dict),
    ]
    return augmentations


def get_augmentation(model: WaveformModel):
    """Define training and validation generator with the following augmentations:

    - Long window around pick
    - Random window of 3001 samples (Phasenet input length)
    - Change datatype to float32 for pytorch
    - Probablistic label
    """
    augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=3000,
            windowlen=6000,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(windowlen=3001, strategy="pad"),
        sbg.ChangeDtype(np.float32),
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict, model_labels=model.labels, sigma=30, dim=0
        ),
    ]
    return augmentations


def load_dataset(
    data: BenchmarkDataset,
    model: WaveformModel,
    type: str,
    batch_size: int = 256,
    num_workers: int = 8,
) -> tuple[sbg.GenericGenerator, DataLoader, BenchmarkDataset]:
    train, dev, test = data.train_dev_test()

    if type == "train":
        train.preload_waveforms()
        dataset = train
    elif type == "dev":
        dev.preload_waveforms()
        dataset = dev
    else:
        test.preload_waveforms()
        dataset = test

    ds_generator = sbg.GenericGenerator(dataset)

    if isinstance(model, PhaseNetMag):
        ds_generator.add_augmentations(get_magnitude_augmentation(windowlen=3001))
    elif isinstance(model, AMAG):
        ds_generator.add_augmentations(get_magnitude_augmentation(windowlen=601))
    elif isinstance(model, (sbm.PhaseNet)):
        ds_generator.add_augmentations(get_augmentation(model))
        ds_generator.add_augmentations(
            [MagnitudeLabellerPhaseNet(phase_dict=phase_dict)]
        )
    else:
        print("load dataset failed")
        return

    loader = DataLoader(
        ds_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=2,
        # worker_init_fn=worker_seeding,
        pin_memory=True,
    )

    return ds_generator, loader, data


if __name__ == "__main__":
    # Load standard Phasenet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    model.to_preferred_device(verbose=True)

    train_generator, _, data = load_dataset(model, "train")

    # Example training input
    sample = train_generator[np.random.randint(len(train_generator))]

    print(sample)

    fig = plt.figure(figsize=(15, 12))
    axs = fig.subplots(
        3, 1, sharex=True, gridspec_kw={"hspace": 0.2, "height_ratios": [3, 1, 1]}
    )
    # Plot Z, N, E waveforms
    channel_names = ["Z", "N", "E"]
    for i in range(sample["X"].shape[0]):
        axs[0].plot(sample["X"][i], label=channel_names[i])
    axs[0].set_ylabel("Waveform")
    axs[0].legend()

    # Plot P, S, Noise phase labels
    phase_names = ["P", "S", "Noise"]
    colors = ["tab:blue", "tab:green", "tab:orange"]
    for i in range(sample["y"].shape[0]):
        axs[1].plot(sample["y"][i], label=phase_names[i], color=colors[i])
    axs[1].set_ylabel("Phase Label")
    axs[1].legend()

    # Plot magnitude label
    if "magnitude" in sample:
        mag_data = sample["magnitude"]
        axs[2].plot(mag_data, color="tab:orange")
        axs[2].set_ylabel("Magnitude Label")
    else:
        axs[2].text(0.5, 0.5, "No magnitude label", ha="center", va="center")
    axs[2].set_xlabel("Sample Index")

    plot_magnitude_distribution(data)
    # dump_metadata_to_csv(data, "GEOFON_metadata.csv")

    plt.show()
