"""Utility functions for handling files during preprocessing."""

import pandas as pd


def get_sample_fps(
    samples_df: pd.DataFrame, well: str = None, tile: int = None, cycle: int = None
) -> list[str]:
    """Filters the samples DataFrame based on optional well, tile, and cycle inputs.

    Args:
        samples_df (pd.DataFrame): DataFrame containing sample data.
        well (str, optional): Well identifier to filter by.
        tile (int, optional): Tile number to filter by.
        cycle (int, optional): Cycle number to filter by.

    Returns:
        list[str]: List of sample file paths that match the filters.
    """
    # Start with the full DataFrame
    filtered_df = samples_df

    # Apply filters if arguments are provided
    if well is not None:
        filtered_df = filtered_df[filtered_df["well"] == well]

    if tile is not None:
        filtered_df = filtered_df[filtered_df["tile"] == int(tile)]

    if cycle is not None:
        filtered_df = filtered_df[filtered_df["cycle"] == int(cycle)]

    # Return the list of file paths as Path objects
    return filtered_df["sample_fp"].tolist()
