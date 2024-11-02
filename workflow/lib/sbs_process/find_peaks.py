"""Find SBS peaks."""

import numpy as np

from lib.shared.image_utils import remove_channels


def find_neighborhood_peaks(data, n=5):
    """Finds local maxima in the input data.

    At a maximum, the value is max - min in a neighborhood of width `n`.
    Elsewhere, it is zero.

    Parameters:
        data (numpy.ndarray): Input data.
        n (int, optional): Width of the neighborhood for finding local maxima. Default is 5.

    Returns:
        peaks (numpy.ndarray): Local maxima scores.
    """
    # Import necessary modules and functions
    from scipy import ndimage as ndi
    import numpy as np

    # Define the maximum and minimum filters for finding local maxima
    filters = ndi.filters

    # Define the neighborhood size based on the input data dimensions
    neighborhood_size = (1,) * (data.ndim - 2) + (n, n)

    # Apply maximum and minimum filters to the data to find local maxima
    data_max = filters.maximum_filter(data, neighborhood_size)
    data_min = filters.minimum_filter(data, neighborhood_size)

    # Calculate the difference between maximum and minimum values to identify peaks
    peaks = data_max - data_min

    # Set values to zero where the original data is not equal to the maximum values
    peaks[data != data_max] = 0

    # Remove peaks close to the edge
    mask = np.ones(peaks.shape, dtype=bool)
    mask[..., n:-n, n:-n] = False
    peaks[mask] = 0

    return peaks


def find_peaks(standard_deviation_data, width=5, remove_index=None):
    """Find local maxima and label by difference to next-highest neighboring pixel.

    Conventionally used to estimate SBS read locations by inputting the standard deviation score.

    Parameters:
        standard_deviation_data (numpy.ndarray): 2D image data of sbs standard deviation.
        width (int, optional): Neighborhood size for finding local maxima. Default is 5.
        remove_index (None or int, optional): Index of data to remove from subsequent analysis, generally any non-SBS channels (e.g., DAPI).

    Returns:
        peaks (numpy.ndarray): Local maxima scores, dimensions same as data. At a maximum, the value is max - min in the defined neighborhood, elsewhere zero.
    """
    # Remove specified index channel if needed
    if remove_index is not None:
        standard_deviation_data = remove_channels(standard_deviation_data, remove_index)

    # If data is 2D, convert it to a list
    if standard_deviation_data.ndim == 2:
        standard_deviation_data = [standard_deviation_data]

    # Find peaks in each image with a defined neighborhood size
    peaks = [
        find_neighborhood_peaks(x, n=width) if x.max() > 0 else x
        for x in standard_deviation_data
    ]

    # Convert the list of peaks to a numpy array and squeeze it to remove singleton dimensions
    peaks = np.array(peaks).squeeze()

    return peaks
