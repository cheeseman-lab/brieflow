"""Functions for evaluating segmentation results."""

from pathlib import Path

import pandas as pd
import numpy as np
import skimage.morphology

from lib.shared.file_utils import parse_filename
from lib.shared.eval import plot_plate_heatmap


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


def segmentation_overview(segmentation_stats_paths):
    """Compile segmentation statistics across multiple files and aggregate by well.

    Processes each segmentation stats file to extract counts for initial nuclei, initial cells,
    after-edge-removal nuclei, after-edge-removal cells, final cells, and final nuclei.
    Aggregates the counts across tiles for each well.

    Args:
        segmentation_stats_paths (list of str): List of file paths to segmentation statistics files,
            each containing well and tile segmentation data.

    Returns:
        pandas.DataFrame: A DataFrame with aggregated segmentation counts for each well.
            Columns include 'well', 'initial_nuclei', 'initial_cells', 'after_edge_removal_nuclei',
            'after_edge_removal_cells', 'final_cells', and 'final_nuclei', with values summed across tiles.
    """
    # Initialize an empty list to store individual DataFrames
    data_frames = []

    # Process each segmentation stats file
    for segmentation_stats_path in segmentation_stats_paths:
        # Read the segmentation stats file
        segmentation_stats = pd.read_csv(segmentation_stats_path, sep="\t")

        # Parse filename to get well and tile information
        segmentation_filename = Path(segmentation_stats_path).name
        data_location, _, _ = parse_filename(segmentation_filename)
        well = data_location["well"]

        # Add the well information as a column
        segmentation_stats["well"] = well

        # Append this DataFrame to the list
        data_frames.append(segmentation_stats)

    # Concatenate all data frames
    combined_df = pd.concat(data_frames, ignore_index=True)

    # Group by well and sum across tiles
    segmentation_overview = combined_df.groupby("well").sum().reset_index()

    return segmentation_overview


def plot_cell_density_heatmap(df_cells, shape="square", plate="6W", **kwargs):
    """Plot a heatmap of cell density by well and tile in a plate layout.

    This function calculates and visualizes the number of cells per well-tile combination in a plate layout.
    It uses `plot_plate_heatmap` to plot cell density across the specified plate type (e.g., '6W', '24W', '96W').

    Args:
        df_cells (pandas.DataFrame):
            DataFrame containing cell data with columns 'well' and 'tile', where each row represents a cell
            located in a specific well and tile.
        shape (str, optional):
            Shape of the subplot for each well within the plate. Options include 'square', '6W_ph', '6W_sbs',
            or a list defining custom row layouts. Defaults to 'square'.
        plate (str):
            Type of plate for plotting layout. Options are {'6W', '24W', '96W'}, which determine the overall
            layout of wells in the plot.
        **kwargs:
            Additional keyword arguments to customize the appearance, passed directly to `plot_plate_heatmap()`.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: A summary DataFrame with cell counts (`cell count`) per well-tile combination.
            - matplotlib.figure.Figure: The figure object containing the plotted heatmap.

    """
    # Calculate cell counts by well and tile
    df_summary = (
        df_cells.groupby(["well", "tile"]).size().reset_index(name="cell count")
    )

    # Plot heatmap
    fig, _ = plot_plate_heatmap(
        df_summary, metric="cell count", shape=shape, plate=plate, **kwargs
    )

    return df_summary, fig
