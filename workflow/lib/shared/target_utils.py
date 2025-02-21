"""Helper functions for using Snakemake outputs and targets for use with Brieflow."""

from pathlib import Path

from snakemake.io import expand, temp, ancient

from lib.shared.file_utils import parse_filename


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


# TODO: move to rule_utils once this file exists
def get_montage_inputs(montage_data_checkpoint, montage_output_template, channels):
    """Generate montage input file paths based on checkpoint data and output template.

    Args:
        montage_data_checkpoint (object): Checkpoint object containing output directory information.
        montage_output_template (str): Template string for generating output file paths.
        channels (list): List of channels to include in the output file paths.

    Returns:
        list: List of generated output file paths for each channel.
    """
    # Resolve the checkpoint output directory using .get()
    checkpoint_output = Path(montage_data_checkpoint.get().output[0])

    # Get actual existing files
    montage_data_files = list(checkpoint_output.glob("*.tsv"))

    # Extract the gene_sgrna parts and make output paths for each channel
    output_files = []
    for montage_data_file in montage_data_files:
        # parse gene, sgrna from filename
        file_metadata = parse_filename(montage_data_file)[0]
        gene = file_metadata["gene"]
        sgrna = file_metadata["sgrna"]

        for channel in channels:
            output_file = str(montage_output_template).format(
                gene=gene, sgrna=sgrna, channel=channel
            )
            output_files.append(output_file)

    return output_files
