"""Utility functions for configuring SBS processing parameters."""

import numpy as np
import skimage.morphology


def identify_cytoplasm_cellpose(nuclei, cells):
    """Identifies and isolates the cytoplasm region in an image based on the provided nuclei and cells masks.

    Parameters:
    nuclei (ndarray): A 2D array representing the nuclei regions.
    cells (ndarray): A 2D array representing the cells regions.

    Returns:
    ndarray: A 2D array representing the cytoplasm regions.
    """
    # Check if the number of unique labels in nuclei and cells are the same
    if len(np.unique(nuclei)) != len(np.unique(cells)):
        return None  # Break out of the function if the masks are not compatible

    # Create an empty cytoplasmic mask with the same shape as cells
    cytoplasms = np.zeros(cells.shape)

    # Iterate over each unique cell label
    for cell_label in np.unique(cells):
        # Skip if the cell label is 0 (background)
        if cell_label == 0:
            continue

        # Find the corresponding nucleus label for this cell
        nucleus_label = cell_label

        # Get the coordinates of the nucleus and cell regions
        nucleus_coords = np.argwhere(nuclei == nucleus_label)
        cell_coords = np.argwhere(cells == cell_label)

        # Update the cytoplasmic mask with the cell region
        cytoplasms[cell_coords[:, 0], cell_coords[:, 1]] = cell_label

        # Remove the nucleus region from the cytoplasmic mask
        cytoplasms[nucleus_coords[:, 0], nucleus_coords[:, 1]] = 0

    # Calculate the number of identified cytoplasms (excluding background label)
    num_cytoplasm_segmented = len(np.unique(cytoplasms)) - 1
    print(f"Number of cytoplasms identified: {num_cytoplasm_segmented}")

    # Return the final cytoplasm array
    return cytoplasms.astype(int)


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


def annotate_on_phenotyping_data(data, nuclei, cells):
    """Annotate outlines of nuclei and cells on phenotyping data.

    This function overlays outlines of nuclei and cells on the provided phenotyping data.

    Args:
        data (numpy.ndarray): Phenotyping data with shape (channels, height, width).
        nuclei (numpy.ndarray): Array representing nuclei outlines.
        cells (numpy.ndarray): Array representing cells outlines.

    Returns:
        numpy.ndarray: Annotated phenotyping data with outlines of nuclei and cells.
    """
    # Ensure data has at least 3 dimensions
    if data.ndim == 2:
        data = data[None]

    # Get dimensions of the phenotyping data
    channels, height, width = data.shape

    # Create an array to store annotated data
    annotated = np.zeros((channels + 1, height, width), dtype=np.uint16)

    # Generate combined mask for nuclei and cells outlines
    mask = (outline_mask(nuclei, direction="inner") > 0) + (
        outline_mask(cells, direction="inner") > 0
    )

    # Copy original data to annotated data
    annotated[:channels] = data

    # Add combined mask to the last channel
    annotated[channels] = mask

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
