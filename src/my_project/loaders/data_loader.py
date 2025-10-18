import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from seisbench.models.base import WaveformModel
from seisbench.data.base import BenchmarkDataset
import seisbench.generate as sbg
import seisbench.models as sbm
from seisbench.util import worker_seeding

from my_project.utils.utils import (
    plot_magnitude_distribution,
    dump_metadata_to_csv,
    plot_samples,
)
from my_project.loaders.magnitude_labellers import (
    MagnitudeLabeller,
    MagnitudeLabellerAMAG,
)
from my_project.models.phasenet_mag.model import PhaseNetMag
from my_project.models.phasenetLSTM.model import PhaseNetLSTM
from my_project.models.phasenetLSTM.modelv2 import PhaseNetConvLSTM
from my_project.models.AMAG.model import AMAG
from my_project.models.AMAG_v2.model import MagnitudeNet

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


def get_random_window(windowlen) -> list:
    augmentations = [
        sbg.WindowAroundSample(
            list(phase_dict.keys()),
            samples_before=windowlen,
            windowlen=windowlen * 2,
            selection="random",
            strategy="variable",
        ),
        sbg.RandomWindow(windowlen=windowlen, strategy="pad"),
        sbg.ChangeDtype(np.float32),
    ]

    return augmentations


def get_magnitude_augmentation(windowlen=3001) -> list:
    """Define training and validation augmentation for magnitude regression only:

    Args:
        windowlen (int): Length of the random window. Default is 3001 for PhaseNet.
                        Use 600 for AMAG model.
    """

    augmentations = get_random_window(windowlen=windowlen)
    augmentations.append(MagnitudeLabeller(phase_dict=phase_dict))

    return augmentations


def get_phase_augmentation(windowlen=3001):
    """Define training and validation augmentation for magnitude regression only:

    Returns:
        windowlen (int): Length of the random window. Default is 3001 for PhaseNet.
                        Use 600 for AMAG model.
    """
    augmentations = get_random_window(windowlen=windowlen)
    augmentations.append(
        sbg.ProbabilisticLabeller(
            label_columns=phase_dict, model_labels=["P", "S", "N"], sigma=30, dim=0
        )
    )

    return augmentations


def get_magnitude_and_phase_augmentation(windowlen=3001):
    """Define training and validation augmentation with both phase and magnitude labels:

    - Random window of specified length
    - Both probabilistic phase labels AND magnitude labels

    Args:
        windowlen (int): Length of the random window. Default is 3001 for PhaseNet, 600 for AMAG.
    """
    augmentations = get_random_window(windowlen=windowlen)
    augmentations.extend(
        [
            sbg.ProbabilisticLabeller(
                label_columns=phase_dict, model_labels=["P", "S", "N"], sigma=30, dim=0
            ),
            MagnitudeLabeller(phase_dict=phase_dict),
        ]
    )
    return augmentations


def get_amag_augmentation(windowlen=600):
    """
    Define augmentation pipeline for AMAG model following the paper's preprocessing:

    1. Bandpass filter 1-20 Hz (mentioned in paper)
    2. Window around first P pick
    3. Random window of 600 samples (6 seconds at 100 Hz)
    4. Change dtype to float32
    5. AMAG magnitude labelling with equation (11): label = mag + 1 for signal

    Note: Detrending and demeaning are handled in the model's annotate_batch_pre method.

    Args:
        windowlen (int): Length of the window in samples (default 600 for 6s at 100Hz)

    Returns:
        list: Augmentation pipeline for SeisBench generator
    """
    p_phase_keys = list(phase_dict.keys())

    augmentations = [
        # 1. Bandpass filter 1-20 Hz (paper's preprocessing step)
        sbg.Filter(
            N=4,  # Filter order
            Wn=[1.0, 20.0],  # 1-20 Hz as specified in paper
            btype="bandpass",
        ),
        # 2. Window around first P pick with flexibility
        sbg.WindowAroundSample(
            p_phase_keys,
            samples_before=int(windowlen * 0.5),  # Center around P arrival
            windowlen=windowlen * 1.5,  # Larger initial window for flexibility
            selection="first",  # Always use first available P pick
            strategy="pad",  # Pad if needed
        ),
        # 3. Random window to get exactly 600 samples
        # sbg.RandomWindow(windowlen=windowlen, strategy="pad"),
        # 4. Change dtype to float32 for PyTorch
        sbg.ChangeDtype(np.float32),
        # 5. AMAG magnitude labeller with equation (11): mag + 1
        # MagnitudeLabellerAMAG(phase_dict=phase_dict, debug=False),
        MagnitudeLabellerPhaseNet(phase_dict=phase_dict),
        sbg.ProbabilisticLabeller(label_columns=phase_dict, sigma=30, dim=0),
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

    if isinstance(model, (PhaseNetMag, MagnitudeNet)):
        ds_generator.add_augmentations(get_magnitude_and_phase_augmentation(3000))
    elif isinstance(model, AMAG):
        ds_generator.add_augmentations(get_amag_augmentation())
    elif isinstance(model, (sbm.PhaseNet, PhaseNetLSTM, PhaseNetConvLSTM)):
        ds_generator.add_augmentations(get_phase_augmentation())
    else:
        print("load dataset failed")
        return

    loader = DataLoader(
        ds_generator,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=worker_seeding,
    )

    return ds_generator, loader, data


if __name__ == "__main__":
    # Load dataset first
    import seisbench.data as sbd

    data = sbd.ETHZ(sampling_rate=100)  # Or whatever dataset you want to use

    # Load standard Phasenet
    model = sbm.PhaseNet(
        phases="PSN", norm="std", default_args={"blinding": (200, 200)}
    )
    # model = PhaseNetMag(in_channels=3, sampling_rate=100, norm="std", filter_factor=1)
    model.to_preferred_device(verbose=True)

    train_generator, _, data = load_dataset(data, model, "train")

    plot_samples(train_generator)

    plot_magnitude_distribution(data)
    # dump_metadata_to_csv(data, "GEOFON_metadata.csv")

    plt.show()
