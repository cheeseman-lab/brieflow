"""Shared utilties for configuring Brieflow process parameters."""

import re
import math
from pathlib import Path

import pandas as pd
from microfilm.microplot import Micropanel
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


def create_samples_df(images_fp, sample_pattern, metadata):
    """Generate samples dataframe from a directory of images.

    Samples dataframe includes path to images and image metadata extracted from the file names.

    Args:
        images_fp (Path): Path to the directory containing images.
        sample_pattern (str): Regular expression pattern to extract metadata from image file names.
        metadata (list): List of metadata keys to extract from the file names.

    Returns:
        Samples DataFrame: DataFrame containing sample metadata and file paths.
    """
    if images_fp is None:
        print("No image directory provided, returning an empty sample DataFrame!")
        return pd.DataFrame(columns=["sample_fp"])

    samples_data = []

    # Iterate over files and extract information
    for image_fp in images_fp.rglob("*"):  # Recursively matches all files in subdirs
        match = re.search(sample_pattern, image_fp.name)
        if match:
            # Find sample path and metadata
            sample_data = {"sample_fp": str(image_fp)}
            sample_metadata = {
                key: match.group(i + 1) for i, key in enumerate(metadata)
            }

            # Convert numeric metadta values to integers where applicable
            for key, value in sample_metadata.items():
                if value.isdigit():
                    sample_metadata[key] = int(value)

            # Update sample data with metadata
            sample_data.update(sample_metadata)

            # Append sample data to list
            samples_data.append(sample_data)

    # Create a DataFrame and sort by metadata
    samples_df = pd.DataFrame(samples_data)
    samples_df = samples_df.sort_values(by=metadata)
    samples_df = samples_df.reset_index(drop=True)

    return samples_df


def create_micropanel(microimages, num_cols=2, figscaling=6, add_channel_label=True):
    """Creates a Micropanel from a list of Microimages.

    Dynamically arranges microimages into a grid based on the specified number of columns.

    Args:
        microimages (list): A list of Microimage objects to be displayed in the panel.
        num_cols (int, optional): Number of columns in the grid. Defaults to 2.
        figscaling (int, optional): Scaling factor for the figure size. Defaults to 4.
        add_channel_label (bool, optional): If True, adds channel labels to the microimages. Defaults to True

    Returns:
        Micropanel: A Micropanel object with microimages arranged in a grid.
    """
    # Calculate grid dimensions
    num_images = len(microimages)
    num_rows = math.ceil(num_images / num_cols)

    # Create panel with dynamic rows
    panel = Micropanel(rows=num_rows, cols=num_cols, figscaling=figscaling)

    # Add all microimages to the panel
    for i, microimage in enumerate(microimages):
        row = i // num_cols
        col = i % num_cols
        panel.add_element([row, col], microimage)

    # Add channel labels to the microimages
    if add_channel_label:
        panel.add_channel_label()

    return panel


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


def image_segmentation_annotations(data, nuclei, cells):
    """Annotate outlines of nuclei and cells on image data.

    This function overlays outlines of nuclei and cells on the provided image data.

    Args:
        data (numpy.ndarray): Image data with shape (channels, height, width).
        nuclei (numpy.ndarray): Array representing nuclei outlines.
        cells (numpy.ndarray): Array representing cells outlines.

    Returns:
        numpy.ndarray: Annotated image data with outlines of nuclei and cells.
    """
    # Ensure data has at least 3 dimensions
    if data.ndim == 2:
        data = data[None]

    # Get dimensions of the image data
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
