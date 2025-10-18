import numpy as np
from seisbench.generate.labeling import SupervisedLabeller
import logging

# Set up logger for debugging magnitude labelling issues
logger = logging.getLogger(__name__)


class MagnitudeLabeller(SupervisedLabeller):
    """
    Labeller for magnitude regression: sets all values to zero before first P pick,
    and to the event's source magnitude after the first P pick.

    Fixed to handle noise-only samples deterministically.
    """

    def __init__(
        self,
        phase_dict,
        magnitude_column="source_magnitude",
        key=("X", "magnitude"),
        debug=False,
    ):
        super().__init__(label_type="multi_label", dim=1, key=key)
        self.phase_dict = phase_dict
        self.magnitude_column = magnitude_column
        self.label_columns = list(phase_dict.keys()) + [magnitude_column]
        self.debug = debug

    def label(self, X, metadata):
        length = X.shape[-1]
        mag = metadata.get(self.magnitude_column, 0.0)

        # Find the earliest pick time from phase_dict keys
        valid_pick_times = []
        for pick_key in self.phase_dict.keys():
            pick = metadata.get(pick_key, np.nan)
            # More robust validation: check for NaN, negative values, and picks beyond signal length
            if not np.isnan(pick) and pick >= 0 and pick < length:
                valid_pick_times.append(pick)

        # Initialize label array with zeros
        label = np.zeros(length, dtype=np.float32)

        # Determine if this is a noise-only sample or a real event
        has_valid_picks = len(valid_pick_times) > 0
        has_valid_magnitude = mag > 0

        # Only apply magnitude if we have BOTH valid picks AND valid magnitude
        if has_valid_picks and has_valid_magnitude:
            onset = int(min(valid_pick_times))
            label[onset:] = mag
            sample_type = "event"
        else:
            sample_type = "noise"

        # Enhanced debugging output
        if self.debug:
            logger.info(
                f"[MagnitudeLabeller] Sample type: {sample_type}, "
                f"mag: {mag}, valid_picks: {len(valid_pick_times)}, "
                f"pick_times: {valid_pick_times}, "
                f"onset: {min(valid_pick_times) if valid_pick_times else None}, "
                f"label_nonzero_count: {np.count_nonzero(label)}, "
                f"label_unique_values: {np.unique(label)}"
            )

        return label


class MagnitudeLabellerAMAG(SupervisedLabeller):
    """
    Labeller for AMAG model following equation (11) from the paper:

    label(t) = {
        0,        if t < tp (noise)
        mag + 1,  if t >= tp (signal)
    }

    The +1 offset serves two purposes:
    1. Prevents underestimation of magnitude predictions
    2. Creates clear separation between noise (0) and signal (mag+1)

    Note: When evaluating predictions, subtract 1 from signal predictions
    to get the actual magnitude estimate.
    """

    def __init__(
        self,
        phase_dict,
        magnitude_column="source_magnitude",
        key=("X", "magnitude"),
        debug=False,
    ):
        super().__init__(label_type="multi_label", dim=1, key=key)
        self.phase_dict = phase_dict
        self.magnitude_column = magnitude_column
        self.label_columns = list(phase_dict.keys()) + [magnitude_column]
        self.debug = debug

    def label(self, X, metadata):
        length = X.shape[-1]
        mag = metadata.get(self.magnitude_column, 0.0)

        # Find the earliest pick time from phase_dict keys
        valid_pick_times = []
        for pick_key in self.phase_dict.keys():
            pick = metadata.get(pick_key, np.nan)
            if not np.isnan(pick) and pick >= 0 and pick < length:
                valid_pick_times.append(pick)

        # Initialize label array with zeros (noise label)
        label = np.zeros(length, dtype=np.float32)

        # Determine if this is a noise-only sample or a real event
        has_valid_picks = len(valid_pick_times) > 0
        has_valid_magnitude = mag > 0

        # Apply equation (11): mag + 1 for signal portion
        if has_valid_picks and has_valid_magnitude:
            onset = int(min(valid_pick_times))
            label[onset:] = mag + 1.0  # KEY DIFFERENCE: Add 1 to magnitude
            sample_type = "event"
        else:
            sample_type = "noise"

        # Enhanced debugging output
        if self.debug:
            logger.info(
                f"[AMAG MagnitudeLabeller] Sample type: {sample_type}, "
                f"raw_mag: {mag}, label_value: {mag + 1.0 if has_valid_picks and has_valid_magnitude else 0.0}, "
                f"valid_picks: {len(valid_pick_times)}, "
                f"pick_times: {valid_pick_times}, "
                f"onset: {min(valid_pick_times) if valid_pick_times else None}, "
                f"label_nonzero_count: {np.count_nonzero(label)}, "
                f"label_unique_values: {np.unique(label)}"
            )

        return label
