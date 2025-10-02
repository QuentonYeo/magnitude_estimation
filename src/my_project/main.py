import seisbench.data as sbd
import seisbench.models as sbm
from seisbench.data import BenchmarkDataset

from my_project.tutorial.tutorial import (
    test_load_data,
    test_generator,
    train_phasenet,
    evaluate_phasenet,
)


def tutorial_tests(data: BenchmarkDataset):
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
    evaluate_phasenet(model=model, data=data)


if __name__ == "__main__":
    """
    Get dataset @100Hz for sampling rate and use defined training splits
    if gots memory for it use cache="trace" if need to redownload use force=True
    NOTE: These dataset are located in ~/.seisbench for transfer or removal
    """
    data = sbd.ETHZ(sampling_rate=100)
    # data = sbd.STEAD(sampling_rate=100)
    # data = sbd.MLAAPDE(sampling_rate=100)
    # data = sbd.GEOFON(sampling_rate=100)

    tutorial_tests(data)
