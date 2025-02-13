"""Helper functions for using Snakemake outputs and targets for use with Brieflow."""

from snakemake.io import expand, temp, ancient


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


def get_valid_combinations(df):
    """Get valid combinations of cycles and channels from a DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing columns 'cycle' and 'channel'.

    Returns:
        list: List of dictionaries with valid combinations of cycles and channels.
    """
    # Group by cycle and get channels that exist for each cycle
    cycle_channels = df.groupby("cycle")["channel"].unique().to_dict()
    valid_combinations = []
    for cycle, channels in cycle_channels.items():
        for channel in channels:
            valid_combinations.append({"cycle": cycle, "channel": channel})
    return valid_combinations


def outputs_to_targets_with_combinations(
    outputs, valid_combinations, plates, wells, tiles=None
):
    """Generate targets for valid combinations.

    Args:
        outputs (list): List of output path templates.
        valid_combinations (list): List of dictionaries with valid combinations of wildcards.
        plates (list): List of plate identifiers.
        wells (list): List of well identifiers.
        tiles (list, optional): List of tile identifiers. Defaults to None.

    Returns:
        list: List of fully resolved target paths.
    """
    targets = []
    for output_template in outputs:
        for plate in plates:
            for well in wells:
                for combo in valid_combinations:
                    kwargs = {
                        "plate": plate,
                        "well": well
                    }
                    if "cycle" in combo:
                        kwargs["cycle"] = combo["cycle"]
                    kwargs["channel"] = combo["channel"]

                    if tiles:
                        for tile in tiles:
                            kwargs["tile"] = tile
                            filepath = str(output_template).format(**kwargs)
                            targets.append(filepath)
                    else:
                        filepath = str(output_template).format(**kwargs)
                        targets.append(filepath)
    return targets