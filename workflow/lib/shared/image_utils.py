import decorator

import numpy as np


@decorator.decorator
def applyIJ(f, arr: np.ndarray, *args: tuple, **kwargs: dict) -> np.ndarray:
    """
    Decorator to apply a function that expects 2D input to the trailing two dimensions of a multi-dimensional array.

    Args:
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
