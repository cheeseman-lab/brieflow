from pathlib import Path

import pandas as pd

from lib.shared.file_utils import parse_filename
from lib.shared.eval import plot_plate_heatmap


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
