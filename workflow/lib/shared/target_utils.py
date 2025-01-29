"""Helper functions for using Snakemake outputs and targets for use with Brieflow."""

from snakemake.io import expand


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


def outputs_to_targets(outputs, wildcards, expansion_method="product"):
    """Expand output templates into full paths by applying the specified wildcards.

    Args:
        outputs (dict): Dictionary of output path templates with placeholders (e.g., PREPROCESS_OUTPUTS).
        wildcards (dict): Dictionary of wildcard values to apply (e.g., {"well": ["A1", "A2"], "cycle": [1, 2]}).
        expansion_method (str): Method of expansion, either 'product' (default) or 'zip'.

    Returns:
        dict: Dictionary of expanded output paths, where each rule maps to a list of fully resolved paths.
    """
    expanded_targets = {}
    for rule_name, path_templates in outputs.items():
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


def output_to_input(output_path, wildcard_values, wildcards):
    """Resolves an output template into input paths by applying wildcards and additional mappings.

    Args:
        output_path (str or pathlib.Path): A single output path template containing placeholders (e.g., "{well}", "{tile}").
        wildcard_values (dict): Additional wildcard mappings to apply (e.g., {"tile": [1, 2]}).
        wildcards (dict): Wildcard values provided by Snakemake (e.g., {"well": "A1", "cycle": 1}).

    Returns:
        list: A list of resolved input paths with placeholders replaced by wildcard values.
    """
    # Merge wildcards with wildcard_values
    all_wildcards = {**wildcards, **wildcard_values}
    # Expand the output template with the merged wildcards
    return expand(output_path, **all_wildcards)
