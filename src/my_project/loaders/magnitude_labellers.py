import numpy as np
from seisbench.generate.labeling import SupervisedLabeller


class MagnitudeLabellerPhaseNet(SupervisedLabeller):
    """
    Labeller for magnitude regression: sets all values to zero before first P pick,
    and to the event's source magnitude after the first P pick.
    """

    def __init__(
        self,
        phase_dict,
        magnitude_column="source_magnitude",
        key=("X", "magnitude"),
    ):
        super().__init__(label_type="multi_label", dim=1, key=key)
        self.phase_dict = phase_dict
        self.magnitude_column = magnitude_column
        self.label_columns = list(phase_dict.keys()) + [magnitude_column]

    def label(self, X, metadata):
        length = X.shape[-1]
        mag = metadata.get(self.magnitude_column, 0.0)
        # Find the earliest pick time from phase_dict keys
        pick_times = []
        for pick_key in self.phase_dict.keys():
            pick = metadata.get(pick_key, np.nan)
            if not np.isnan(pick):
                pick_times.append(pick)
        if pick_times:
            onset = int(min(pick_times))
        else:
            onset = None
        label = np.zeros(length, dtype=np.float32)
        if onset is not None and onset < length:
            label[onset:] = mag
        # Debug print
        # print(
        #     f"[MagnitudeLabeller] mag: {mag}, onset: {onset}, label (nonzero count): {np.count_nonzero(label)}, label (unique): {np.unique(label)}"
        # )
        return label
