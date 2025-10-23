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
from my_project.models.AMAG_v2.model import MagnitudeNet
from my_project.models.EQTransformer.model import EQTransformerMag
from my_project.models.ViT.model import ViTMagnitudeEstimator

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
    elif isinstance(model, (EQTransformerMag, ViTMagnitudeEstimator)):
        # EQTransformerMag and ViT use 30-second windows (3001 samples at 100Hz)
        ds_generator.add_augmentations(get_magnitude_and_phase_augmentation(3001))
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
