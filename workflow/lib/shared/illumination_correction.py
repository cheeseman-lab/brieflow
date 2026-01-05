"""Functions related to illumination correction in BrieFlow.

Used in preprocessing to calculate the ic field and downstream steps to apply the ic field to images.
"""

import warnings
from typing import List
import sys

from joblib import Parallel, delayed
import numpy as np
import random
from skimage import morphology
import skimage.restoration
import skimage.transform
import skimage.filters
from lib.shared.io import read_image

# from tifffile import imread

from lib.shared.image_utils import applyIJ


@applyIJ
def rescale_channels(data: np.ndarray) -> np.ndarray:
    """Rescales the image data by dividing by a robust minimum and setting values below 1 to 1.

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
    sample_fraction: float = 1.0,
) -> np.ndarray:
    print(f"DEBUG: calculate_ic_field started with {len(files)} files.", file=sys.stderr)
    # Randomly sample a subset of files if sample_fraction is less than 1.0
    if sample_fraction < 1.0:
        sample_size = int(len(files) * sample_fraction)
        files = random.sample(files, sample_size)
    print(f"DEBUG: Processing {len(files)} files after sampling.", file=sys.stderr)

    # Initialize data variable
    data = read_image(files[0])[slicer] / len(files)
    print(f"DEBUG: Initialized data with shape {data.shape} from {files[0]}.", file=sys.stderr)

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
        for i, file in enumerate(files[1:]):
            print(f"DEBUG: Accumulating file {i+1}/{len(files)-1}: {file}", file=sys.stderr)
            data = accumulate_image(file, slicer, data, len(files))

    print(f"DEBUG: All images accumulated. Final data shape: {data.shape}.", file=sys.stderr)
    # Squeeze and convert data to uint16 (remove any dimensions of size 1)
    data = np.squeeze(data.astype(np.uint16))
    print(f"DEBUG: Data squeezed and cast to uint16. New shape: {data.shape}.", file=sys.stderr)

    # Calculate default smoothing factor if not provided
    if not smooth:
        smooth = int(np.sqrt((data.shape[-1] * data.shape[-2]) / (np.pi * 20)))
    print(f"DEBUG: Smoothing factor: {smooth}.", file=sys.stderr)

    selem = morphology.disk(smooth)
    median_filter = applyIJ(skimage.filters.median)

    # Apply median filter with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        print(f"DEBUG: Applying median filter to data of shape {data.shape}.", file=sys.stderr)
        smoothed = median_filter(data, selem, behavior="rank")
        print(f"DEBUG: Median filter applied. Smoothed data shape: {smoothed.shape}.", file=sys.stderr)

    # Rescale channels if requested
    if rescale:
        print("DEBUG: Rescaling channels.", file=sys.stderr)
        smoothed = rescale_channels(smoothed)
        print(f"DEBUG: Channels rescaled. Smoothed data shape: {smoothed.shape}.", file=sys.stderr)

    print("DEBUG: calculate_ic_field finished.", file=sys.stderr)
    return smoothed


def accumulate_image(file: str, slicer: slice, data: np.ndarray, N: int) -> np.ndarray:
    print(f"DEBUG: accumulate_image: Reading {file}", file=sys.stderr)
    data += read_image(file)[slicer] / N
    print(f"DEBUG: accumulate_image: Finished reading and accumulating {file}", file=sys.stderr)
    return data


def apply_ic_field(
    data,
    correction=None,
    zproject=False,
    rolling_ball=False,
    rolling_ball_kwargs={},
    n_jobs=1,
    backend="threading",
):
    """Apply illumination correction to the given data.

    Based CellProfiler's CorrectIlluminationApply module
    https://github.com/CellProfiler/CellProfiler/blob/fa81fb0f2850c7c6d9cefdf4e71806188f1dc546/src/frontend/cellprofiler/modules/correctilluminationapply.py#L62

    Args:
        data (np.ndarray): Input data to be corrected.
        correction (np.ndarray, optional): Correction factor to be applied. Defaults to None.
        zproject (bool, optional): If True, perform a maximum projection along the first axis.
            Defaults to False.
        rolling_ball (bool, optional): If True, apply rolling ball background subtraction.
            Defaults to False.
        rolling_ball_kwargs (dict, optional): Additional arguments for rolling ball background
            subtraction. Defaults to an empty dictionary.
        n_jobs (int, optional): Number of parallel jobs to run. Defaults to 1 (no parallelization).
        backend (str, optional): Parallel backend to use ('threading' or 'multiprocessing').
            Defaults to 'threading'.

    Returns:
        np.ndarray: Corrected data.
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


def applyIJ_parallel(f, arr, n_jobs=-2, backend="threading", *args, **kwargs):
    """Decorator to apply a function that expects 2D input to the trailing two dimensions of an array, parallelizing computation across 2D frames.

    Args:
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





@applyIJ
def rolling_ball_background_skimage(
    image, radius=100, ball=None, shrink_factor=None, smooth=None, **kwargs
):
    """Apply rolling ball background subtraction to an image using skimage.

    Args:
        image (np.ndarray): Input image for background subtraction.
        radius (int, optional): Radius of the rolling ball. Defaults to 100.
        ball (np.ndarray, optional): Precomputed ball kernel. If None, it will be generated.
        shrink_factor (int, optional): Factor to shrink the image and ball for faster computation.
            Determined based on radius if not provided.
        smooth (float, optional): Sigma for Gaussian smoothing applied to the background after
            rolling ball. Defaults to None.
        **kwargs: Additional arguments passed to skimage's rolling_ball function.

    Returns:
        np.ndarray: Calculated background to be subtracted from the original image.
    """
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
    """Subtract the background from an image using the rolling ball algorithm.

    Args:
        image (np.ndarray): Input image from which to subtract the background.
        radius (int, optional): Radius of the rolling ball. Defaults to 100.
        ball (np.ndarray, optional): Precomputed ball kernel. If None, it will be generated.
        shrink_factor (int, optional): Factor to shrink the image and ball for faster computation.
            Determined based on radius if not provided.
        smooth (float, optional): Sigma for Gaussian smoothing applied to the background after
            rolling ball. Defaults to None.
        **kwargs: Additional arguments passed to the rolling_ball_background_skimage function.

    Returns:
        np.ndarray: Image with the background subtracted.
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


def combine_ic_images(images, indices):
    """Combine illumination correction images using specified indices.

    Args:
        images: List of IC images [extra_channels_image, base_channels_image]
        indices: List of indices [extra_channel_indices, base_channel_indices].
                If base_channel_indices is None, automatically selects all channels
                except the extra_channel_indices for complementary selection.

    Returns:
        Combined IC image with extra channels from first image and base channels from second.
    """
    extra_img = images[0]
    base_img = images[1]

    # Infer target from the larger image (usually has all channels)
    target_channels = max(extra_img.shape[0], base_img.shape[0])

    # Extract extra channels
    if indices[0] is not None:
        extra_channels = extra_img[indices[0]]
        extra_indices_used = (
            indices[0] if isinstance(indices[0], list) else [indices[0]]
        )
    else:
        extra_channels = extra_img
        extra_indices_used = list(range(extra_img.shape[0]))

    # Extract base channels
    if indices[1] is not None:
        # Explicit base indices provided
        base_channels = base_img[indices[1]]
    else:
        # Smart selection: all channels EXCEPT the extra channel indices
        all_indices = list(range(target_channels))
        base_indices = [i for i in all_indices if i not in extra_indices_used]
        # If the base image has fewer channels than needed, adjust indices
        if (
            max(base_indices) >= base_img.shape[0]
        ):  # base indices is out of bounds of base_img
            base_indices = list(
                range(len(base_indices))
            )  # the length of the base indices starting from 0
        # Extract base channels using the adjusted indices
        base_channels = base_img[base_indices]

    # Ensure both are 3D and concatenate
    if extra_channels.ndim == 2:
        extra_channels = extra_channels[np.newaxis]
    if base_channels.ndim == 2:
        base_channels = base_channels[np.newaxis]

    return np.concatenate([extra_channels, base_channels], axis=0)