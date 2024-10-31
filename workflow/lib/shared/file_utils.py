"""
Utility functions for handling and filtering sample file paths in the BrieFlow pipeline.
"""

import re

import pandas as pd


def get_sample_fps(
    samples_df: pd.DataFrame, well: str = None, tile: int = None, cycle: int = None
) -> list[str]:
    """
    Filters the samples DataFrame based on optional well, tile, and cycle inputs.

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


def extract_tile_from_filename(filepath: str) -> int:
    """
    Extracts the tile number from a given filename.

    Args:
        filepath (str): The path to the file.

    Returns:
        int: The extracted tile number, or None if not found.
    """

    match = re.search(r"Points-(\d+)", filepath)
    if match:
        return int(match.group(1))
    return None


def get_filename(data_location: dict, info_type: str, file_type: str) -> str:
    """
    Generate a structured filename based on data location, information type, and file type.

    Parameters:
    - data_location (dict): Dictionary containing location info like well, tile, and cycle.
    - info_type (str): Type of information (e.g., 'cell_features', 'sbs_reads').
    - file_type (str): File extension/type (e.g., 'tsv', 'hdf5', 'tiff').

    Returns:
    - str: Structured filename.
    """
    print(data_location)

    # Well has no leading zeros
    well_str = f"W{data_location.get('well')}"

    # Tile with 4 digits leading zero padding if numeric
    tile = data_location.get("tile")
    if tile and tile.isdigit():
        tile_str = f"_T{int(tile):04d}"
    else:
        tile_str = f"_T{tile}" if tile else ""

    # Cycle with 2 digits leading zero padding if numeric
    cycle = data_location.get("cycle")
    if cycle and cycle.isdigit():
        cycle_str = f"_C{int(cycle):02d}"
    else:
        cycle_str = f"_C{cycle}" if cycle else ""

    # Construct filename by combining components with info_type and file_type
    filename = f"{well_str}{tile_str}{cycle_str}__{info_type}.{file_type}"
    return filename
