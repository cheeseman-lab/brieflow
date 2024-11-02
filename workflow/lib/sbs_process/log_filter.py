"""Filter data with Laplacian of Gaussian filter."""

import warnings

import numpy as np
from scipy import ndimage
import skimage

from lib.shared.image_utils import applyIJ


@applyIJ
def log_ndi(data, sigma=1, *args, **kwargs):
    """Apply Laplacian of Gaussian to each image in a stack of shape (..., I, J).

    Args:
        data (numpy.ndarray): Input data.
        sigma (float, optional): Standard deviation of the Gaussian kernel. Default is 1.
        *args: Additional positional arguments passed to scipy.ndimage.filters.gaussian_laplace.
        **kwargs: Additional keyword arguments passed to scipy.ndimage.filters.gaussian_laplace.

    Returns:
        numpy.ndarray: Resulting images after applying Laplacian of Gaussian.
    """
    # Define the Laplacian of Gaussian filter function
    f = ndimage.filters.gaussian_laplace

    # Apply the filter to the data and invert the output
    arr_ = -1 * f(data.astype(float), sigma, *args, **kwargs)

    # Clip values to ensure they are within the valid range [0, 65535] and convert back to uint16
    arr_ = np.clip(arr_, 0, 65535) / 65535

    # Suppress precision warning from skimage
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return skimage.img_as_uint(arr_)


def log_filter(aligned_image_data, sigma=1, skip_index=None):
    """Apply Laplacian-of-Gaussian filter from scipy.ndimage to the input data.

    Args:
        aligned_image_data (numpy.ndarray): Aligned SBS image data with expected dimensions of (CYCLE, CHANNEL, I, J).
        sigma (float, optional): Size of the Gaussian kernel used in the Laplacian-of-Gaussian filter. Default is 1.
        skip_index (None or int, optional): If an integer, skips transforming a specific channel (e.g., DAPI with skip_index=0).

    Returns:
        loged (numpy.ndarray): LoG-ed `data`.
    """
    # Convert input data to a numpy array
    aligned_image_data = np.array(aligned_image_data)

    # Apply Laplacian-of-Gaussian filter to the data using ops.process.log_ndi function from ops.process module
    loged = log_ndi(aligned_image_data, sigma=sigma)

    # If skip_index is specified, keep the original values for the corresponding channel
    if skip_index is not None:
        loged[..., skip_index, :, :] = aligned_image_data[..., skip_index, :, :]

    return loged
