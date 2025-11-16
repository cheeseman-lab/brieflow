"""Shared utilties for configuring Brieflow process parameters.

This includes:
- Header string for Brieflow config file.
- Function to create the Brieflow samples dataframe with file location and metadata.
- Functions for displaying SBS/phenotype images and segmentations.
- Functions for viewing steps of merge process such as determining tiles to merge and seeing an example merge.
"""

import re
import math
from pathlib import Path

import pandas as pd
from microfilm.microplot import Micropanel
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import skimage.morphology
from scipy.spatial.distance import cdist
import matplotlib.colors as mcolors

from lib.merge.fast_merge import build_linear_model

CONFIG_FILE_HEADER = """
# ~BrieFlow analysis configuration file~

# All paths are resolved relative to the directory the workflow is run from.

# Parameters:
"""


def create_samples_df(images_fp, sample_pattern, metadata, metadata_order_type):
    """Generate samples dataframe from a directory of images.

    Samples dataframe includes path to images and image metadata extracted from file names.
    The function reorders columns, changes column types, and sorts the DataFrame.

    Args:
        images_fp (Path): Path to the directory containing images.
        sample_pattern (str): Regular expression pattern to extract metadata from image file names.
        metadata (list): List of metadata keys to extract from the file names.
        metadata_order_type (dict): Dictionary specifying the order and data types of metadata columns.

    Returns:
        DataFrame: DataFrame containing sample metadata and file paths.
    """
    if images_fp is None:
        print("No image directory provided, returning an empty sample DataFrame!")
        return pd.DataFrame(columns=["sample_fp"] + metadata)

    samples_data = []
    sample_regex = re.compile(sample_pattern)

    # Find and extract metadata from matching files
    for image_fp in Path(images_fp).rglob("*"):
        match = sample_regex.search(str(image_fp))
        if match:
            samples_data.append(
                {"sample_fp": str(image_fp), **dict(zip(metadata, match.groups()))}
            )

    # Create DataFrame
    samples_df = pd.DataFrame(samples_data)

    if samples_df.empty:
        raise ValueError(
            f"No matching files found in {images_fp} with pattern {sample_pattern}"
        )

    # Convert column types according to metadata_order_type
    for column, column_type in metadata_order_type.items():
        if column in samples_df.columns:
            samples_df[column] = samples_df[column].astype(column_type)

    # Reorder columns
    column_order = ["sample_fp"] + list(metadata_order_type.keys())
    samples_df = samples_df[column_order]

    # Sort DataFrame by metadata columns
    sort_columns = list(metadata_order_type.keys())
    samples_df = samples_df.sort_values(by=sort_columns)

    # Reset index
    samples_df.reset_index(drop=True, inplace=True)

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


def convert_tuples_to_lists(obj):
    """Recursively convert all tuples in a data structure to lists."""
    if isinstance(obj, tuple):
        return [convert_tuples_to_lists(i) for i in obj]
    elif isinstance(obj, list):
        return [convert_tuples_to_lists(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
    else:
        return obj

