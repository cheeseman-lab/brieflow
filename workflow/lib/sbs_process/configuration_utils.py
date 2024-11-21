"""Utility functions for configuring SBS processing parameters."""

import numpy as np
import skimage.morphology


def annotate_segment_on_sequencing_data(data, nuclei, cells):
    """Annotate outlines of nuclei and cells on sequencing data.

    This function overlays outlines of nuclei and cells on the provided sequencing data.

    Args:
        data (numpy.ndarray): Sequencing data with shape (cycles, channels, height, width).
        nuclei (numpy.ndarray): Array representing nuclei outlines.
        cells (numpy.ndarray): Array representing cells outlines.

    Returns:
        numpy.ndarray: Annotated sequencing data with outlines of nuclei and cells.
    """
    # Ensure data has at least 4 dimensions
    if data.ndim == 3:
        data = data[None]

    # Get dimensions of the sequencing data
    cycles, channels, height, width = data.shape

    # Create an array to store annotated data
    annotated = np.zeros((cycles, channels + 1, height, width), dtype=np.uint16)

    # Generate combined mask for nuclei and cells outlines
    mask = (outline_mask(nuclei, direction="inner") > 0) + (
        outline_mask(cells, direction="inner") > 0
    )

    # Copy original data to annotated data
    annotated[:, :channels] = data

    # Add combined mask to the last channel
    annotated[:, channels] = mask

    return np.squeeze(annotated)


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
