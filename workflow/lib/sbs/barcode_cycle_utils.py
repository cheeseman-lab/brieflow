"""Barcode Cycle Utilities.

This module provides a utility function for determining the start and end sequencing
cycle positions for mapping and recombination barcodes.
"""


def compute_barcode_cycle_positions(
    map_prefix_length,
    recomb_prefix_length,
    sequencing_order,
    skip_cycles_map,
    skip_cycles_recomb,
):
    """Calculate sequencing boundaries for MAP and RECOMB regions.

    Args:
        map_prefix_length (int): Length of mapping prefix prefix
        recomb_prefix_length (int): Length of recombination prefix
        sequencing_order (str): Either "map_recomb" or "recomb_map"
        skip_cycles_map (list): List of cycles to skip for mapping prefix
        skip_cycles_recomb (list): List of cycles to skip for recombination prefix

    Returns:
        tuple: (map_start, map_end, recomb_start, recomb_end)
    """
    # Handle None values and calculate effective lengths after skipping cycles
    skip_map_count = len(skip_cycles_map) if skip_cycles_map is not None else 0
    skip_recomb_count = len(skip_cycles_recomb) if skip_cycles_recomb is not None else 0

    effective_map_length = map_prefix_length - skip_map_count
    effective_recomb_length = recomb_prefix_length - skip_recomb_count

    if sequencing_order == "map_recomb":
        # MAP comes first
        map_start = 1
        map_end = effective_map_length
        recomb_start = map_end + 1
        recomb_end = map_end + effective_recomb_length

    elif sequencing_order == "recomb_map":
        # RECOMB comes first
        recomb_start = 1
        recomb_end = effective_recomb_length
        map_start = recomb_end + 1
        map_end = recomb_end + effective_map_length

    else:
        raise ValueError("sequencing_order must be either 'map_recomb' or 'recomb_map'")

    return map_start, map_end, recomb_start, recomb_end
