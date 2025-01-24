"""Utility functions for handling and filtering sample file paths in the BrieFlow pipeline."""

import logging

import numpy as np

from tifffile import imread

log = logging.getLogger(__name__)


def get_filename(data_location: dict, info_type: str, file_type: str) -> str:
    """Generate a structured filename based on data location, information type, and file type.

    Args:
        data_location (dict): Dictionary containing location info like well, tile, and cycle.
        info_type (str): Type of information (e.g., 'cell_features', 'sbs_reads').
        file_type (str): File extension/type (e.g., 'tsv', 'hdf5', 'tiff').

    Returns:
        str: Structured filename.
    """
    # Well info
    well = data_location.get("well")
    well_str = f"W{well}" if well else ""

    # Tile info
    tile = data_location.get("tile")
    tile_str = f"_T{tile}" if tile else ""

    # Cycle info
    cycle = data_location.get("cycle")
    cycle_str = f"_C{cycle}" if cycle else ""

    # Channel info
    channel = data_location.get("channel")
    channel_str = f"_CH{channel}" if channel else ""

    # Construct the filename by combining the components with info_type and file type
    if any([well, tile, cycle, channel]):
        filename = (
            f"{well_str}{tile_str}{cycle_str}{channel_str}__{info_type}.{file_type}"
        )
    else:
        filename = f"{info_type}.{file_type}"
    return filename


def parse_filename(filename: str) -> tuple:
    """Parse a structured filename to extract data location, information type, and file type.

    Args:
        filename (str): Structured filename, e.g., 'WA1_T02_C03__cell_features.tsv'.

    Returns:
        tuple: A tuple containing:
            - data_location (dict): Dictionary with keys 'well', 'tile', 'cycle' as applicable.
            - info_type (str): The type of information (e.g., 'cell_features').
            - file_type (str): The file extension/type (e.g., 'tsv').
    """
    # Split the filename into main parts
    base, file_type = filename.rsplit(".", 1)
    parts = base.split("__")

    # Initialize data_location dictionary and variables
    data_location = {}
    info_type = None

    # Parse data location part (e.g., 'WA1_T02_C03')
    if len(parts) == 2:
        location_part, info_type = parts
        elements = location_part.split("_")

        for element in elements:
            if element.startswith("W"):
                data_location["well"] = element[1:]  # remove 'W'
            elif element.startswith("T"):
                data_location["tile"] = int(
                    element[1:]
                )  # remove 'T' and convert to int
            elif element.startswith("C"):
                data_location["cycle"] = int(
                    element[1:]
                )  # remove 'C' and convert to int
            elif element.startswith("CH"):
                data_location["channel"] = element[2:]  # remove 'CH'
    else:
        # If no location part, the first part is the info_type
        info_type = parts[0]

    return data_location, info_type, file_type
