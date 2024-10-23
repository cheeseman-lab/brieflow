import decorator
import warnings

from joblib import Parallel, delayed
import numpy as np
import skimage.morphology
import skimage.filters
from skimage.io import imread

@decorator.decorator
def applyIJ(f, arr: np.ndarray, *args: tuple, **kwargs: dict) -> np.ndarray:
    """
    Decorator to apply a function that expects 2D input to the trailing two dimensions of a multi-dimensional array.

    Arguments:
        f (Callable): The function to apply to each 2D slice of the input array.
        arr (np.ndarray): The input array with at least two dimensions, where the function will be applied to the last two dimensions.
        *args (tuple): Additional positional arguments to be passed to the function.
        **kwargs (dict): Additional keyword arguments to be passed to the function.

    Returns:
        np.ndarray: The output array with the function applied to the trailing two dimensions of the input array.
    """
    # Get the height and width of the trailing two dimensions of the input array
    h, w = arr.shape[-2:]
    
    # Reshape the input array to a 3D array with shape (-1, h, w), where -1 indicates the product of all other dimensions
    reshaped = arr.reshape((-1, h, w))

    # Apply the function f to each frame in the reshaped array, along with additional arguments and keyword arguments
    arr_ = [f(frame, *args, **kwargs) for frame in reshaped]

    # Determine the output shape based on the input array shape and the shape of the output from the function f
    output_shape = arr.shape[:-2] + arr_[0].shape
    
    # Reshape the resulting list of arrays to the determined output shape
    return np.array(arr_).reshape(output_shape)


def calculate_ic(files, smooth=None, rescale=True, threading=False, slicer=slice(None)):
    """
    Calculate illumination correction field for use with the apply_illumination_correction
    Snake method. Equivalent to CellProfiler's CorrectIlluminationCalculate module with
    option "Regular", "All", "Median Filter".

    Note: Algorithm originally benchmarked using ~250 images per plate to calculate plate-wise
    illumination correction functions (Singh et al. J Microscopy, 256(3):231-236, 2014).

    Parameters:
    -----------
    files : list
        List of file paths to images for which to calculate the illumination correction.
    smooth : int, optional
        Smoothing factor for the correction. Default is calculated as 1/20th of the image area.
    rescale : bool, default True
        Whether to rescale the correction field.
    threading : bool, default False
        Whether to use threading for parallel processing.
    slicer : slice, optional
        Slice object to select specific parts of the images.

    Returns:
    --------
    numpy.ndarray
        The calculated illumination correction field.
    """
    
    N = len(files)
    
    print(f"{N} files passed into image correction module")
    
    # Initialize global data variable
    global data
    data = imread(files[0])[slicer] / N
    
    def accumulate_image(file):
        global data
        data += imread(file)[slicer] / N
    
    # Accumulate images using threading or sequential processing, averaging them
    if threading:
        Parallel(n_jobs=-1, require='sharedmem')(delayed(accumulate_image)(file) for file in files[1:])
    else:
        for file in files[1:]:
            accumulate_image(file)
    
    # Squeeze and convert data to uint16 (remove any dimensions of size 1)
    data = np.squeeze(data.astype(np.uint16))
    
    # Calculate default smoothing factor if not provided
    if not smooth:
        smooth = int(np.sqrt((data.shape[-1] * data.shape[-2]) / (np.pi * 20)))
    print(f"Smoothing factor: {smooth}")
    
    selem = skimage.morphology.disk(smooth)
    median_filter = applyIJ(skimage.filters.median)
    
    # Apply median filter with warning suppression
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        smoothed = median_filter(data, selem, behavior='rank')
    
    # Rescale channels if requested
    if rescale:
        @applyIJ
        def rescale_channels(data):
            # Use 2nd percentile for robust minimum
            robust_min = np.quantile(data.reshape(-1), q=0.02)
            robust_min = 1 if robust_min == 0 else robust_min
            data = data / robust_min
            data[data < 1] = 1
            return data
        
        smoothed = rescale_channels(smoothed)
    
    return smoothed