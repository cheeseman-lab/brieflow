"""Shared utilties for configuring Brieflow process parameters."""

import numpy as np
import matplotlib
import skimage.morphology

CONFIG_FILE_HEADER = """
# BrieFlow configuration file

# Defining samples:
#   Samples must be defined in the following TSV files with columns:
#     sbs_samples.tsv: sample_fp, well, tile, cycle
#     phenotype_samples.tsv: sample_fp, well, tile

# Paths:
#   Paths are resolved relative to the directory the workflow is run from

# Parameters:
"""


def random_cmap(alpha=0.5, num_colors=256):
    """Create a random colormap for segmentation.

    Args:
        alpha (float, optional): Transparency value for the colors in the colormap, ranging from 0 (transparent)
            to 1 (opaque). Defaults to 0.5.
        num_colors (int, optional): Number of colors to generate in the colormap. Defaults to 256.

    Returns:
        matplotlib.colors.ListedColormap: A colormap object with randomly generated colors, where the first
            color is set to black with full transparency.
    """
    colmat = np.random.rand(num_colors, 4)
    colmat[:, -1] = alpha
    # Set the first color to black with full transparency
    colmat[0, :] = [0, 0, 0, 1]
    cmap = matplotlib.colors.ListedColormap(colmat)
    return cmap


def outline_mask(arr, direction="outer", width=1):
    """Remove interior of label mask in `arr`.

    Args:
        arr (numpy.ndarray): The input label mask array.
        direction (str, optional): The direction of outlining. 'outer' outlines the outer boundary, 'inner' outlines the inner boundary. Default is 'outer'.
        width (int, optional): The width of the structuring element used for erosion and dilation. Default is 1.

    Returns:
        numpy.ndarray: The label mask array with the outlined interior removed.

    Raises:
        ValueError: If `direction` is neither 'outer' nor 'inner'.
    """
    selem = skimage.morphology.disk(
        width
    )  # Create a disk-shaped structuring element with the specified width
    arr = (
        arr.copy()
    )  # Create a copy of the input array to avoid modifying the original array
    if direction == "outer":  # If outlining direction is 'outer'
        mask = skimage.morphology.erosion(
            arr, selem
        )  # Erode the mask using the structuring element
        arr[mask > 0] = 0  # Set interior pixels to 0
        return arr  # Return the modified array
    elif direction == "inner":  # If outlining direction is 'inner'
        mask1 = (
            skimage.morphology.erosion(arr, selem) == arr
        )  # Create a mask for pixels on the inner boundary
        mask2 = (
            skimage.morphology.dilation(arr, selem) == arr
        )  # Create a mask for pixels on the outer boundary
        arr[mask1 & mask2] = (
            0  # Set pixels within the inner boundary and outside the outer boundary to 0
        )
        return arr  # Return the modified array
    else:  # If direction is neither 'outer' nor 'inner'
        raise ValueError(direction)  # Raise a ValueError
