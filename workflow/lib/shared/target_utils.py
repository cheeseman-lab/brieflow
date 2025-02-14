"""Helper functions for using Snakemake outputs and targets for use with Brieflow."""

import string
from pathlib import Path
from snakemake.io import expand, temp, ancient

def clean_value(val):
    """Convert numpy types to Python native types.
    
    Args:
        val: Value to clean (could be numpy type or native Python type)
        
    Returns:
        Native Python type (int, str, etc.)
    """
    if hasattr(val, 'item'):  # Check if it's a numpy type
        return val.item()
    return val

def map_outputs(outputs, output_type_mappings):
    """Apply Snakemake output type mappings (e.g., temp, protected) to output paths.

    Args:
        outputs (dict): Output templates with `pathlib.Path` paths (e.g., PREPROCESS_OUTPUTS).
        output_type_mappings (dict): Mapping of output rules to Snakemake output types.
                                     Can be a single function (e.g., temp, protected) or
                                     a list of functions for multiple outputs.

    Returns:
        dict: Final mapped outputs with output type mappings applied.
    """
    mapped_outputs = {}

    for rule_name, output_templates in outputs.items():
        # Get the output type mapping for the current rule
        output_func = output_type_mappings.get(rule_name)

        # If no mapping function, keep outputs as is
        if output_func is None:
            mapped_outputs[rule_name] = output_templates
        else:
            # Check if output_func is a list (for multiple outputs)
            if isinstance(output_func, list):
                # Ensure the length matches the output_templates
                if len(output_func) != len(output_templates):
                    raise ValueError(
                        f"Length of output mappings for '{rule_name}' does not match "
                        f"number of outputs. Expected {len(output_templates)}, got {len(output_func)}."
                    )
                # Apply each mapping function to the corresponding template
                mapped_outputs[rule_name] = [
                    func(output) if func else output
                    for output, func in zip(output_templates, output_func)
                ]
            else:
                # Apply the single mapping function to all templates
                mapped_outputs[rule_name] = [
                    output_func(output) for output in output_templates
                ]

    return mapped_outputs


def outputs_to_targets(outputs, wildcards, output_mappings, expansion_method="product"):
    """Expand output templates into full paths by applying the specified wildcards.

    Args:
        outputs (dict): Dictionary of output path templates with placeholders (e.g., PREPROCESS_OUTPUTS).
        wildcards (dict): Dictionary of wildcard values to apply (e.g., {"well": ["A1", "A2"], "cycle": [1, 2]}).
        output_mappings (dict): Mapping of output rules to Snakemake output types (e.g., temp, protected).
        expansion_method (str): Method of expansion, either 'product' (default) or 'zip'.

    Returns:
        dict: Dictionary of expanded output paths, where each rule maps to a list of fully resolved paths.
    """
    expanded_targets = {}
    for rule_name, path_templates in outputs.items():
        # skip temporary outputs
        if output_mappings[rule_name] == temp:
            continue

        if expansion_method == "zip":
            expanded_targets[rule_name] = [
                expand(str(path_template), zip, **wildcards)
                for path_template in path_templates
            ]
        else:  # Default to product expansion
            expanded_targets[rule_name] = [
                expand(str(path_template), **wildcards)
                for path_template in path_templates
            ]
    return expanded_targets


def output_to_input(output_path, wildcard_values, wildcards, ancient_output=False):
    """Resolves an output template into input paths by applying wildcards and additional mappings.

    Args:
        output_path (str or pathlib.Path): A single output path template containing placeholders (e.g., "{well}", "{tile}").
        wildcard_values (dict): Additional wildcard mappings to apply (e.g., {"tile": [1, 2]}).
        wildcards (dict): Wildcard values provided by Snakemake (e.g., {"well": "A1", "cycle": 1}).
        ancient_output (bool, optional): Whether to wrap output paths with snakemake's ancient() function. Defaults to False.

    Returns:
        list: A list of resolved input paths with placeholders replaced by wildcard values.
    """
    # Merge wildcards with wildcard_values
    all_wildcards = {**wildcards, **wildcard_values}
    # Expand the output template with the merged wildcards
    inputs = expand(output_path, **all_wildcards)

    # Prevent rerunning if ancient
    if ancient_output:
        inputs = [ancient(path) for path in inputs]

    return inputs


def outputs_to_targets_with_combinations(output_templates, valid_combinations, extra_keys=None):
    """Convert output templates to targets using valid combinations.
    
    Args:
        output_templates (list): List of output template strings or Path objects
        valid_combinations (list): List of dictionaries containing valid combinations
        extra_keys (list): Optional list of additional keys to iterate over (e.g., tiles)
        
    Returns:
        list: List of target paths with all valid combinations
    """
    targets = []
    
    # If we have any extra keys (like tiles), create cartesian product
    if extra_keys:
        for combo in valid_combinations:
            for extra_val in extra_keys:
                # Create a complete mapping including the extra value
                mapping = {
                    'plate': clean_value(combo['plate']),
                    'well': combo['well'],
                    'cycle': clean_value(combo['cycle']) if 'cycle' in combo else None,
                    'channel': combo['channel'],
                    'tile': extra_val
                }
                
                # Apply mapping to each output template
                for template in output_templates:
                    template_str = str(template)
                    template_keys = [k[1] for k in string.Formatter().parse(template_str) if k[1]]
                    filtered_mapping = {k: v for k, v in mapping.items() 
                                     if k in template_keys and v is not None}
                    formatted_path = template_str.format(**filtered_mapping)
                    targets.append(formatted_path)
    else:
        # No extra keys, just use the combinations directly
        for combo in valid_combinations:
            for template in output_templates:
                template_str = str(template)
                template_keys = [k[1] for k in string.Formatter().parse(template_str) if k[1]]
                filtered_mapping = {k: clean_value(v) if k in ['plate', 'cycle'] else v 
                                 for k, v in combo.items() 
                                 if k in template_keys and v is not None}
                formatted_path = template_str.format(**filtered_mapping)
                targets.append(formatted_path)
    
    return targets


def output_to_input_from_combinations(output_path, valid_combinations, wildcards, expand_values=None, ancient_output=False):
    """Resolves an output template into input paths using valid combinations and optional expansion values.
    
    Args:
        output_path (str or pathlib.Path): Output path template with placeholders
        valid_combinations (list): List of valid combination dictionaries
        wildcards (dict): Wildcard values provided by Snakemake
        expand_values (dict, optional): Additional values to expand each combination with (e.g., {"tile": [1, 2]})
        ancient_output (bool, optional): Whether to wrap output paths with ancient(). Defaults to False.
    
    Returns:
        list: Resolved input paths matching the valid combinations and expansions
    """
    # Filter combinations to match provided wildcards
    matching_combos = []
    for combo in valid_combinations:
        # Check if this combination matches all provided wildcards
        if all(combo.get(key) == value for key, value in wildcards.items() if key in combo):
            matching_combos.append(combo)
    
    # Generate expanded combinations if expand_values provided
    if expand_values:
        expanded_combos = []
        for combo in matching_combos:
            # Similar to output_to_input's expand, but with our combinations
            expanded_values = [dict(zip(expand_values.keys(), v)) 
                             for v in itertools.product(*expand_values.values())]
            for exp_value in expanded_values:
                expanded_combos.append({**combo, **exp_value})
        matching_combos = expanded_combos
    
    # Generate input paths from matching combinations
    inputs = []
    for combo in matching_combos:
        try:
            path = str(output_path).format(**combo)
            inputs.append(path)
        except KeyError as e:
            continue
    
    # Wrap with ancient() if requested
    if ancient_output:
        inputs = [ancient(path) for path in inputs]
    
    return inputs


def get_valid_combinations(df, data_type):
    """Get valid combinations and identify missing data for SBS or phenotype data.

    Args:
        df (pd.DataFrame): DataFrame containing SBS or phenotype data.
        data_type (str): Type of data, either "sbs" or "phenotype".

    Returns:
        tuple: (valid_combinations, warnings)
    """
    if df is None or df.empty:
        return [], ["Warning: No data found!"]
    
    warnings = []
    valid_combinations = []
    
    # Get all unique plates and wells to check for missing combinations
    all_plates = sorted(df['plate'].unique())
    all_wells = sorted(df['well'].unique())
    existing_pairs = set(zip(df['plate'], df['well']))
    
    # Check for missing plate/well combinations first
    for plate in all_plates:
        for well in all_wells:
            if (plate, well) not in existing_pairs:
                warnings.append(f"Warning: No data found for Plate {plate}, Well {well}")
    
    # Process only existing pairs for valid combinations
    for plate, well in existing_pairs:
        well_df = df[(df['plate'] == plate) & (df['well'] == well)]
        if well_df.empty:
            continue
        
        if data_type == "sbs":
            # SBS: Check cycle-channel combinations
            all_cycles = sorted(df['cycle'].unique())  # All possible cycles
            cycles_in_well = sorted(well_df['cycle'].unique())
            
            # Check for missing cycles
            missing_cycles = set(all_cycles) - set(cycles_in_well)
            if missing_cycles:
                warnings.append(f"Warning: Plate {plate}, Well {well} is missing cycles: {sorted(missing_cycles)}")
            
            for cycle in cycles_in_well:
                cycle_df = well_df[well_df['cycle'] == cycle]
                if cycle_df.empty:
                    continue
                
                for _, row in cycle_df.iterrows():
                    valid_combinations.append({
                        'plate': plate,
                        'well': well,
                        'cycle': clean_value(cycle),
                        'channel': row['channel']
                    })
        
        else:  # Phenotype data
            # For phenotype data, check for missing channels
            all_channels = sorted(df['channel'].unique())
            channels_in_well = sorted(well_df['channel'].unique())
            missing_channels = set(all_channels) - set(channels_in_well)
            
            if missing_channels:
                warnings.append(f"Warning: Plate {plate}, Well {well} is missing channels: {sorted(missing_channels)}")
            
            for _, row in well_df.iterrows():
                valid_combinations.append({
                    'plate': plate,
                    'well': well,
                    'channel': row['channel']
                })
    
    return valid_combinations, warnings


def get_sbs_combinations(df):
    """Get valid SBS combinations and identify missing data.

    Args:
        df (pd.DataFrame): DataFrame containing SBS data.

    Returns:
        tuple: (valid_combinations, warnings)
    """
    valid_combinations, warnings = get_valid_combinations(df, "sbs")

    for warning in warnings:
        print(warning)

    return valid_combinations


def get_phenotype_combinations(df):
    """Get valid phenotype combinations and identify missing data.

    Args:
        df (pd.DataFrame): DataFrame containing phenotype data.

    Returns:
        tuple: (valid_combinations, warnings)
    """
    valid_combinations, warnings = get_valid_combinations(df, "phenotype")

    for warning in warnings:
        print(warning)

    return valid_combinations