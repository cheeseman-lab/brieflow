"""Shared functions for aligning images.

Uses NumPy and scikit-image to provide image
alignment between sequencing cycles, apply percentile-based filtering, fill masked
areas with noise, and perform various transformations to enhance image data quality.
"""

import numpy as np
import skimage
from scipy.ndimage import shift as ndi_shift

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


def calculate_offsets(data_, upsample_factor, max_offset=None, reference="first"):
    """Calculate offsets between images using phase cross-correlation.

    Args:
        data_ (np.ndarray): Image data with first axis as frames.
        upsample_factor (int): Upsampling factor for cross-correlation.
        max_offset (float, optional): Maximum allowed offset in pixels. Offsets with
            magnitude exceeding this threshold are clamped and a warning is printed.
            If None, defaults to 10% of the smallest image dimension. Defaults to None.
        reference (str, optional): Reference strategy for computing offsets. Options:
            - "first": Align all frames to the first frame (original behavior).
            - "mean": Align all frames to the mean of all frames. More robust when
              individual frames are noisy or have artifacts.
            Defaults to "first".

    Returns:
        np.ndarray: Offset values (y, x) between images, shape (N, 2).
    """
    # Determine reference image
    if reference == "mean":
        target = data_.mean(axis=0)
    else:
        target = data_[0]

    # Auto-compute max_offset based on image dimensions if not provided
    if max_offset is None:
        max_offset = min(data_.shape[-2], data_.shape[-1]) * 0.1

    offsets = []
    for i, src in enumerate(data_):
        # First frame gets zero offset when using "first" reference
        if reference == "first" and i == 0:
            offsets.append((0, 0))
        else:
            offset, _, _ = skimage.registration.phase_cross_correlation(
                src, target, upsample_factor=upsample_factor
            )

            # Validate offset magnitude and clamp if too large
            offset_magnitude = np.sqrt(offset[0] ** 2 + offset[1] ** 2)
            if offset_magnitude > max_offset:
                print(
                    f"Warning: Frame {i} offset ({offset[0]:.2f}, {offset[1]:.2f}) "
                    f"magnitude {offset_magnitude:.2f}px exceeds max_offset "
                    f"{max_offset:.1f}px. Clamping to max."
                )
                scale = max_offset / offset_magnitude
                offset = offset * scale

            offsets.append(tuple(offset))

    return np.array(offsets)


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
    """Apply translation offsets to image data using scipy.ndimage.shift.

    Uses scipy.ndimage.shift for efficient sub-pixel translation with bilinear
    interpolation. This is purpose-built for pure translations and avoids the
    overhead of general-purpose geometric warping.

    Args:
        data_ (np.ndarray): Image data with first axis as frames.
        offsets (np.ndarray): Offset values (y, x) to apply to each frame.

    Returns:
        np.ndarray: Shifted image data.
    """
    warped = []
    for frame, offset in zip(data_, offsets):
        if offset[0] == 0 and offset[1] == 0:
            warped.append(frame)
        else:
            # scipy.ndimage.shift with order=1 (bilinear interpolation) and
            # mode='constant' (fill edges with 0) matches previous warp behavior
            frame_ = ndi_shift(frame, offset, order=1, mode="constant", cval=0)
            warped.append(frame_.astype(data_.dtype))
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
