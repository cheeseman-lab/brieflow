import numpy as np

from lib.shared.image_utils import remove_channels


def compute_standard_deviation(log_filtered_data, remove_index=None):
    """
    Use standard deviation over cycles, followed by mean across channels to estimate sequencing read locations.
    If only 1 cycle is present, takes standard deviation across channels.

    Parameters:
        log_filtered_data (numpy.ndarray): LoG-ed SBS image data with expected dimensions of (CYCLE, CHANNEL, I, J).
        remove_index (None or int, optional): Index of data to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI).

    Returns:
        consensus (numpy.ndarray): Standard deviation score for each pixel, dimensions of (I, J).
    """
    # Remove specified index channel if needed
    if remove_index is not None:
        log_filtered_data = remove_channels(log_filtered_data, remove_index)

    # If only one cycle present, add a new dimension
    if len(log_filtered_data.shape) == 3:
        log_filtered_data = log_filtered_data[:, None, ...]

    # Compute standard deviation across cycles and mean across channels
    consensus = np.std(log_filtered_data, axis=0).mean(axis=0)

    return consensus
