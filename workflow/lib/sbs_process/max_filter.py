"""Apply a maximum filter to an image."""

from scipy.ndimage.filters import maximum_filter

from lib.shared.image_utils import remove_channels


def max_filter(log_filtered_data, width, remove_index=None):
    """Apply a maximum filter in a window of `width`.

    Conventionally operates on Laplacian-of-Gaussian filtered SBS data,
    dilating sequencing channels to compensate for single-pixel alignment error.

    Args:
        log_filtered_data (numpy.ndarray): Log filtered image data with expected dimensions of (..., I, J) with up to 4 total dimensions.
        width (int): Neighborhood size for max filtering.
        remove_index (None or int, optional): Index of data to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI).

    Returns:
        maxed (numpy.ndarray): Maxed `data` with preserved dimensions.
    """
    # Ensure data has at least 3 dimensions
    if log_filtered_data.ndim == 2:
        log_filtered_data = log_filtered_data[None, None]
    elif log_filtered_data.ndim == 3:
        log_filtered_data = log_filtered_data[None]

    # Remove specified index channel if needed
    if remove_index is not None:
        log_filtered_data = remove_channels(log_filtered_data, remove_index)

    # Apply maximum filter to the data with specified window size
    maxed = maximum_filter(log_filtered_data, size=(1, 1, width, width))

    return maxed
