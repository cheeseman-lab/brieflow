from pathlib import Path

import pandas as pd

from lib.shared.file_utils import parse_filename


def get_segmentation_overview(segmentation_stats_paths):
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
