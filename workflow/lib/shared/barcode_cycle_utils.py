"""Barcode Cycle Utilities.

This module provides a utility function for determining the start and end sequencing
cycle positions for mapping and recombination barcodes.
"""

def compute_barcode_cycle_positions(MAP_PREFIX_LENGTH, RECOMB_PREFIX_LENGTH, 
                                   SEQUENCING_ORDER, SKIP_CYCLES_MAP, SKIP_CYCLES_RECOMB):
    """Calculate sequencing boundaries for MAP and RECOMB regions.
    
    Args:
        MAP_PREFIX_LENGTH (int): Length of MAP prefix
        RECOMB_PREFIX_LENGTH (int): Length of RECOMB prefix
        SEQUENCING_ORDER (str): Either "map_recomb" or "recomb_map"
        SKIP_CYCLES_MAP (list): List of cycles to skip for MAP
        SKIP_CYCLES_RECOMB (list): List of cycles to skip for RECOMB
    
    Returns:
        tuple: (MAP_START, MAP_END, RECOMB_START, RECOMB_END)
    """
    # Handle None values and calculate effective lengths after skipping cycles
    skip_map_count = len(SKIP_CYCLES_MAP) if SKIP_CYCLES_MAP is not None else 0
    skip_recomb_count = len(SKIP_CYCLES_RECOMB) if SKIP_CYCLES_RECOMB is not None else 0
    
    effective_map_length = MAP_PREFIX_LENGTH - skip_map_count
    effective_recomb_length = RECOMB_PREFIX_LENGTH - skip_recomb_count
    
    if SEQUENCING_ORDER == "map_recomb":
        # MAP comes first
        MAP_START = 1
        MAP_END = effective_map_length
        RECOMB_START = MAP_END + 1
        RECOMB_END = MAP_END + effective_recomb_length
        
    elif SEQUENCING_ORDER == "recomb_map":
        # RECOMB comes first
        RECOMB_START = 1
        RECOMB_END = effective_recomb_length
        MAP_START = RECOMB_END + 1
        MAP_END = RECOMB_END + effective_map_length
        
    else:
        raise ValueError("SEQUENCING_ORDER must be either 'map_recomb' or 'recomb_map'")
    
    return MAP_START, MAP_END, RECOMB_START, RECOMB_END