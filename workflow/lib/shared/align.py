"""Shared functions for aligning images.

Uses NumPy and scikit-image to provide image alignment between sequencing cycles.
"""

import numpy as np
import skimage

from lib.shared.image_utils import applyIJ, remove_channels


def apply_window(data, window):
    """Apply a window to image data.

    Args:
        data (np.ndarray): Image data.
        window (int): Size of the window to apply.

    Returns:
        np.ndarray: Filtered image data.
    """
    # Extract height and width dimensions from the last two axes of the data shape
    height, width = data.shape[-2:]

    # Define a function to find the border based on the window size
    def find_border(x):
        return int((x / 2.0) * (1 - 1 / float(window)))

    # Calculate border indices
    i, j = find_border(height), find_border(width)

    # Return the data with the border cropped out
    return data[..., i : height - i, j : width - j]


def fill_noise(data, mask, x1, x2):
    """Fill masked areas of data with uniform noise.

    Args:
        data (np.ndarray): Input image data.
        mask (np.ndarray): Boolean mask indicating areas to be replaced with noise.
        x1 (int): Lower threshold value.
        x2 (int): Upper threshold value.

    Returns:
        np.ndarray: Filtered image data.
    """
    # Make a copy of the original data
    filtered = data.copy()
    # Initialize a random number generator with seed 0
    rs = np.random.RandomState(0)
    # Replace the masked values with uniform noise generated in the range [x1, x2]
    filtered[mask] = rs.uniform(x1, x2, mask.sum()).astype(data.dtype)
    # Return the filtered data
    return filtered


def calculate_offsets(data_, upsample_factor):
    """Calculate offsets between images using phase cross-correlation.

    Args:
        data_ (np.ndarray): Image data.
        upsample_factor (int): Upsampling factor for cross-correlation.

    Returns:
        tuple: (offsets, errors) where offsets is np.ndarray of offset values
            and errors is np.ndarray of correlation errors (0 = perfect correlation).
    """
    # Set the target frame as the first frame in the data
    target = data_[0]

    # Initialize empty lists to store offsets and errors
    offsets = []
    errors = []
    # Iterate through each frame in the data
    for i, src in enumerate(data_):
        # If it's the first frame, add a zero offset and zero error
        if i == 0:
            offsets += [(0, 0)]
            errors += [0.0]
        else:
            # Calculate the offset between the current frame and the target frame
            offset, error, _ = skimage.registration.phase_cross_correlation(
                src, target, upsample_factor=upsample_factor, normalization=None
            )
            # Add the offset and error to the lists
            offsets += [offset]
            errors += [error]
    # Convert the lists to numpy arrays and return
    return np.array(offsets), np.array(errors)


@applyIJ
def filter_percentiles(data, q1, q2):
    """Replace data outside of the percentile range [q1, q2] with uniform noise.

    Args:
        data (np.ndarray): Input image data.
        q1 (int): Lower percentile threshold.
        q2 (int): Upper percentile threshold.

    Returns:
        np.ndarray: Filtered image data.
    """
    # Calculate the q1th and q2th percentiles of the input data
    x1, x2 = np.percentile(data, [q1, q2])
    # Create a mask where values are outside the range [x1, x2]
    mask = (x1 > data) | (x2 < data)
    # Fill the masked values with uniform noise in the range [x1, x2] using the fill_noise function
    return fill_noise(data, mask, x1, x2)


def apply_offsets(data_, offsets):
    """Apply offsets to image data.

    Args:
        data_ (np.ndarray): Image data.
        offsets (np.ndarray): Offset values to be applied.

    Returns:
        np.ndarray: Warped image data.
    """
    # Initialize an empty list to store warped frames
    warped = []
    # Iterate through each frame and its corresponding offset
    for frame, offset in zip(data_, offsets):
        # If the offset is zero, add the frame as it is
        if offset[0] == 0 and offset[1] == 0:
            warped += [frame]
        else:
            # Otherwise, apply a similarity transform to warp the frame based on the offset
            st = skimage.transform.SimilarityTransform(translation=offset[::-1])
            frame_ = skimage.transform.warp(frame, st, preserve_range=True)
            # Add the warped frame to the list
            warped += [frame_.astype(data_.dtype)]
    # Convert the list of warped frames to a numpy array and return
    return np.array(warped)


def normalize_by_percentile(data_, q_norm=70):
    """Normalize data by the specified percentile.

    Args:
        data_ (np.ndarray): Input image data.
        q_norm (int, optional): Percentile value for normalization. Defaults to 70.

    Returns:
        np.ndarray: Normalized image data.
    """
    # Get the shape of the input data
    shape = data_.shape
    # Replace the last two dimensions with a single dimension to allow percentile calculation
    shape = shape[:-2] + (-1,)
    # Calculate the q_normth percentile along the last two dimensions of the data
    p = np.percentile(data_, q_norm, axis=(-2, -1))[..., None, None]
    # Normalize the data by dividing it by the calculated percentile values
    normed = data_ / p
    # Return the normalized data
    return normed


def apply_custom_offsets(data, offsets_dict):
    """Apply custom offsets to specific channels in image data.

    Applies custom offsets to specified channels. Useful for aligning channels acquired
    in different imaging rounds when automatic alignment fails or there is no common channel
    to use as reference.

    Offset directions:
    - To shift left: +x
    - To shift right: -x
    - To shift up: +y
    - To shift down: -y

    Args:
        data (np.ndarray): Input image data.
        offsets_dict (dict): Mapping of channel index to (y, x) offset.

    Returns:
        np.ndarray: Image data with custom offsets applied.
    """
    # Set up full offsets array, initialized with zeros
    offsets = np.array([(0, 0) for _ in range(data.shape[0])])

    # Apply custom offsets for specified channels
    for channel, offset_yx in offsets_dict.items():
        offsets[channel] = offset_yx

    # Apply the calculated offsets to data
    adjusted = apply_offsets(data, offsets)

    return adjusted
