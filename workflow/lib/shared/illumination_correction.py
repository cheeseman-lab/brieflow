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


def applyIJ_parallel(f, arr, n_jobs=-2, backend="threading", *args, **kwargs):
    """
    Decorator to apply a function that expects 2D input to the trailing two dimensions of an array,
    parallelizing computation across 2D frames.

    Parameters:
        f (function): The function to be decorated and applied in parallel.
        arr (numpy.ndarray): The input array to apply the function to.
        n_jobs (int): The number of jobs to run in parallel. Default is -2.
        backend (str): The parallelization backend to use. Default is 'threading'.
        *args: Additional positional arguments to be passed to the function.
        **kwargs: Additional keyword arguments to be passed to the function.

    Returns:
        numpy.ndarray: Output array after applying the function in parallel.
    """

    h, w = arr.shape[-2:]
    reshaped = arr.reshape((-1, h, w))

    work = reshaped

    arr_ = Parallel(n_jobs=n_jobs, backend=backend)(
        delayed(f)(frame, *args, **kwargs) for frame in work
    )

    output_shape = arr.shape[:-2] + arr_[0].shape
    return np.array(arr_).reshape(output_shape)


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


@applyIJ
def rolling_ball_background_skimage(
    image, radius=100, ball=None, shrink_factor=None, smooth=None, **kwargs
):
    """
    Apply rolling ball background subtraction to an image using skimage.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image for background subtraction.
    radius : int, default 100
        Radius of the rolling ball.
    ball : numpy.ndarray, optional
        Precomputed ball kernel. If None, it will be generated.
    shrink_factor : int, optional
        Factor by which to shrink the image and ball for faster computation.
        Default is determined based on the radius.
    smooth : float, optional
        Sigma for Gaussian smoothing applied to the background after rolling ball.
    kwargs : dict
        Additional arguments passed to skimage's rolling_ball function.

    Returns:
    --------
    numpy.ndarray
        The calculated background to be subtracted from the original image.
    """
    import skimage.restoration
    import skimage.transform
    import skimage.filters

    # Generate the ball kernel if not provided
    if ball is None:
        ball = skimage.restoration.ball_kernel(radius, ndim=2)

    # Determine shrink factor and trim based on the radius
    if shrink_factor is None:
        if radius <= 10:
            shrink_factor = 1
            trim = 0.12  # Trim 24% in x and y
        elif radius <= 30:
            shrink_factor = 2
            trim = 0.12  # Trim 24% in x and y
        elif radius <= 100:
            shrink_factor = 4
            trim = 0.16  # Trim 32% in x and y
        else:
            shrink_factor = 8
            trim = 0.20  # Trim 40% in x and y

        # Trim the ball kernel
        n = int(ball.shape[0] * trim)
        i0, i1 = n, ball.shape[0] - n
        ball = ball[i0:i1, i0:i1]

    # Rescale the image and ball kernel
    image_rescaled = skimage.transform.rescale(
        image, 1.0 / shrink_factor, preserve_range=True
    ).astype(image.dtype)
    kernel_rescaled = skimage.transform.rescale(
        ball, 1.0 / shrink_factor, preserve_range=True
    ).astype(ball.dtype)

    # Compute the rolling ball background
    background = skimage.restoration.rolling_ball(
        image_rescaled, kernel=kernel_rescaled, **kwargs
    )

    # Apply Gaussian smoothing if specified
    if smooth is not None:
        background = skimage.filters.gaussian(
            background, sigma=smooth / shrink_factor, preserve_range=True
        )

    # Resize the background to the original image size
    background_resized = skimage.transform.resize(
        background, image.shape, preserve_range=True
    ).astype(image.dtype)

    return background_resized


def subtract_background(
    image, radius=100, ball=None, shrink_factor=None, smooth=None, **kwargs
):
    """
    Subtract the background from an image using the rolling ball algorithm.

    Parameters:
    -----------
    image : numpy.ndarray
        Input image from which to subtract the background.
    radius : int, default 100
        Radius of the rolling ball.
    ball : numpy.ndarray, optional
        Precomputed ball kernel. If None, it will be generated.
    shrink_factor : int, optional
        Factor by which to shrink the image and ball for faster computation.
        Default is determined based on the radius.
    smooth : float, optional
        Sigma for Gaussian smoothing applied to the background after rolling ball.
    kwargs : dict
        Additional arguments passed to the rolling_ball_background_skimage function.

    Returns:
    --------
    numpy.ndarray
        The image with the background subtracted.
    """
    # Calculate the background using the rolling ball algorithm
    background = rolling_ball_background_skimage(
        image,
        radius=radius,
        ball=ball,
        shrink_factor=shrink_factor,
        smooth=smooth,
        **kwargs,
    )

    # Ensure that the background does not exceed the image values
    mask = background > image
    background[mask] = image[mask]

    # Subtract the background from the image
    return image - background


def apply_ic_field(
    data,
    correction=None,
    zproject=False,
    rolling_ball=False,
    rolling_ball_kwargs={},
    n_jobs=1,
    backend="threading",
):
    """
    Apply illumination correction to the given data.

    Parameters:
    data (numpy array): The input data to be corrected.
    correction (numpy array, optional): The correction factor to be applied. Default is None.
    zproject (bool, optional): If True, perform a maximum projection along the first axis. Default is False.
    rolling_ball (bool, optional): If True, apply a rolling ball background subtraction. Default is False.
    rolling_ball_kwargs (dict, optional): Additional arguments for the rolling ball background subtraction. Default is an empty dictionary.
    n_jobs (int, optional): The number of parallel jobs to run. Default is 1 (no parallelization).
    backend (str, optional): The parallel backend to use ('threading' or 'multiprocessing'). Default is 'threading'.

    Returns:
    numpy array: The corrected data.
    """

    # If zproject is True, perform a maximum projection along the first axis
    if zproject:
        data = data.max(axis=0)

    # If n_jobs is 1, process the data without parallelization
    if n_jobs == 1:
        # Apply the correction factor if provided
        if correction is not None:
            data = (data / correction).astype(np.uint16)

        # Apply rolling ball background subtraction if specified
        if rolling_ball:
            data = subtract_background(data, **rolling_ball_kwargs).astype(np.uint16)

        return data

    else:
        # If n_jobs is greater than 1, apply illumination correction in parallel
        return applyIJ_parallel(
            apply_ic_field,
            arr=data,
            correction=correction,
            backend=backend,
            n_jobs=n_jobs,
        )
