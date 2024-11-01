"""
Utility functions for handling and filtering sample file paths in the BrieFlow pipeline.
"""


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

    # Well has no leading zeros
    well_str = f"W{data_location.get('well')}"

    # Tile with 4 digits leading zero padding if numeric
    tile = data_location.get("tile")
    tile_str = f"_T{tile}" if tile else ""

    # Cycle with 2 digits leading zero padding if numeric
    cycle = data_location.get("cycle")
    cycle_str = f"_C{cycle}" if cycle else ""

    # Construct filename by combining components with info_type and file_type
    filename = f"{well_str}{tile_str}{cycle_str}__{info_type}.{file_type}"
    return filename
