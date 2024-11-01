"""
Functions related to illumination correction in BrieFlow.
Used in preprocessing to calculate the ic field and downstream steps to apply the ic field to images.
"""

import warnings
from typing import List

from joblib import Parallel, delayed
import numpy as np
from skimage import morphology
from skimage import filters
from skimage.io import imread

from lib.shared.image_utils import applyIJ


def accumulate_image(file: str, slicer: slice, data: np.ndarray, N: int) -> np.ndarray:
    """
    Accumulates an image's contribution by adding a sliced version of it to the provided data array.

    Args:
        file (str): Path to the image file to be accumulated.
        slicer (slice): Slice object to select specific parts of the image.
        data (np.ndarray): The numpy array where the accumulated image data is stored.
        N (int): The number of files, used to average the data by dividing each image.

    Returns:
        np.ndarray: Updated image data with the new image accumulated.
    """

    data += imread(file)[slicer] / N
    return data


@applyIJ
def rescale_channels(data: np.ndarray) -> np.ndarray:
    """
    Rescales the image data by dividing by a robust minimum and setting values below 1 to 1.

    Args:
        data (np.ndarray): The input image data to be rescaled.

    Returns:
        np.ndarray: The rescaled image data.
    """

    # Use 2nd percentile for robust minimum
    robust_min = np.quantile(data.reshape(-1), q=0.02)
    robust_min = 1 if robust_min == 0 else robust_min
    data = data / robust_min
    data[data < 1] = 1
    return data


def calculate_ic_field(
    files: List[str],
    smooth: int = None,
    rescale: bool = True,
    threading: bool = False,
    slicer: slice = slice(None),
) -> np.ndarray:
    """
    Calculate illumination correction field for use with the apply_illumination_correction
    Snake method. Equivalent to CellProfiler's CorrectIlluminationCalculate module with
    option "Regular", "All", "Median Filter".

    Note: Algorithm originally benchmarked using ~250 images per plate to calculate plate-wise
    illumination correction functions (Singh et al. J Microscopy, 256(3):231-236, 2014).

    Args:
        files (List[str]): List of file paths to images for which to calculate the illumination correction.
        smooth (int, optional): Smoothing factor for the correction. Default is calculated as 1/20th of the image area.
        rescale (bool, optional): Whether to rescale the correction field. Defaults to True.
        threading (bool, optional): Whether to use threading for parallel processing. Defaults to False.
        slicer (slice, optional): Slice object to select specific parts of the images.

    Returns:
        np.ndarray: The calculated illumination correction field.
    """

    # Initialize data variable
    data = imread(files[0])[slicer] / len(files)

    # Accumulate images using threading or sequential processing, averaging them
    if threading:
        # Accumulate results in parallel and combine them
        results = Parallel(n_jobs=-1, require="sharedmem")(
            delayed(accumulate_image)(file, slicer, np.zeros_like(data), len(files))
            for file in files[1:]
        )
        for result in results:
            data += result  # Aggregate results from parallel processing
    else:
        for file in files[1:]:
            data = accumulate_image(file, slicer, data, len(files))

    # Squeeze and convert data to uint16 (remove any dimensions of size 1)
    data = np.squeeze(data.astype(np.uint16))

    # Calculate default smoothing factor if not provided
    if not smooth:
        smooth = int(np.sqrt((data.shape[-1] * data.shape[-2]) / (np.pi * 20)))

    selem = morphology.disk(smooth)
    median_filter = applyIJ(filters.median)

    # Apply median filter with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed = median_filter(data, selem, behavior="rank")

    # Rescale channels if requested
    if rescale:
        smoothed = rescale_channels(smoothed)

    return smoothed
