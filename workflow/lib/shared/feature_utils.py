"""General functions for extracting features from image regions."""

import numpy as np


def correlate_channels_masked(r, first, second):
    """Cross-correlation between non-zero pixels of two channels within a masked region.

    Args:
        r (skimage regionprops object): Region properties object containing intensity images for multiple channels.
        first (int): Index of the first channel.
        second (int): Index of the second channel.

    Returns:
        float: Mean cross-correlation coefficient between the non-zero pixels of the two channels.
    """
    # Extract intensity images for the specified channels from the region
    A = masked(r, first)
    B = masked(r, second)

    # Filter out zero pixels from both channels
    filt = (A > 0) & (B > 0)
    # If no non-zero pixels are found, return NaN
    if filt.sum() == 0:
        return np.nan

    # Filter the intensity values based on the non-zero pixel indices
    A = A[filt]
    B = B[filt]
    # Calculate the cross-correlation coefficient between the two channels
    corr = (A - A.mean()) * (B - B.mean()) / (A.std() * B.std())

    # Return the mean cross-correlation coefficient
    return corr.mean()


def masked(r, index):
    """Extract masked intensity image for a specific channel index from a region.

    Args:
        r (skimage regionprops object): Region properties object containing intensity images for multiple channels.
        index (int): Index of the channel to extract.

    Returns:
        array: Masked intensity image for the specified channel index.
    """
    return r.intensity_image_full[index][r.image]


def correlate_channels_all_multichannel(r):
    """Compute cross-correlation between masked images of all channels within a region.

    Args:
        r (skimage regionprops object): Region properties object containing intensity images for multiple channels.

    Returns:
        array: Array containing cross-correlation values between all pairs of channels.
    """
    # Compute correlation coefficients for all pairs of channels
    R = np.corrcoef(r.intensity_image[r.image].T)

    # Extract upper triangle (excluding the diagonal)
    # same order as itertools.combinations of channel numbers
    return R[np.triu_indices_from(R, k=1)]
