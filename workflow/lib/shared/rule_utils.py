"""Helper functions for using Snakemake rules for use with Brieflow."""


def filter_outputs_by_cycle_index(outputs, index):
    """Filter and select outputs by cycle index from sorted available cycles.
    
    Args:
        outputs (list): List of output paths
        index (int): Index to select from sorted cycles (-1 for last)
        
    Returns:
        str: Selected path or empty string if not found
    """  
    # Extract and sort cycles
    cycles = set()
    for path in outputs:
        if '_C-' in path and '__' in path:
            cycle_str = path.split('_C-')[1].split('__')[0]
            try:
                cycles.add(int(cycle_str))
            except ValueError:
                continue
    
    available_cycles = sorted(list(cycles))
    
    if not available_cycles:
        return str()
        
    # Handle negative indices
    if index < 0:
        index = len(available_cycles) + index
    
    # Validate index
    if index < 0 or index >= len(available_cycles):
        return str()
        
    # Get cycle number at requested index
    target_cycle = available_cycles[index]
    
    # Filter for target cycle
    cycle_str = f"_C-{str(target_cycle)}__"
    filtered = list(set([path for path in outputs if cycle_str in path]))
    
    return str(filtered[0]) if filtered else str()


def get_alignment_params(wildcards, config):
    """Get alignment parameters for a specific plate.

    Args:
        wildcards (snakemake.Wildcards): Snakemake wildcards object.
        config (dict): Configuration dictionary.
        
    Returns:
        dict: Alignment parameters for the specified plate.
    """
    # Convert plate string to integer - no need for complex string replacements
    plate_id = int(wildcards.plate)
    
    # Get plate-specific alignment config - we know it uses integer keys
    plate_config = config["phenotype"]["alignments"].get(plate_id)
    
    if not plate_config:
        raise ValueError(
            f"No alignment configuration found for plate {plate_id}. "
            f"Available plates: {list(config['phenotype']['alignments'].keys())}"
        )
    
    # Return appropriate config based on whether it's multi-step or not
    if "steps" in plate_config:
        return {
            "align": True,
            "multi_step": True,
            "steps": plate_config["steps"]
        }
    
    return {
        "align": True,
        "multi_step": False,
        "target": plate_config["target"],
        "source": plate_config["source"],
        "riders": plate_config["riders"],
        "remove_channel": plate_config["remove_channel"]
    }
