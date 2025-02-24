"""Helper functions for using Snakemake outputs and targets for use with Brieflow."""

from pathlib import Path

import pandas as pd
from snakemake.io import expand, ancient

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


def outputs_to_targets(module_outputs, wildcards_df, module_target_mappings):
    """Convert module output templates to concrete target paths using Snakemake expand."""
    targets = []

    # Extract all wildcards as separate lists for zip expansion
    wildcard_values = {col: wildcards_df[col].tolist() for col in wildcards_df.columns}

    # Process each rule's outputs
    for rule_name, rule_outputs in module_outputs.items():
        if module_target_mappings[rule_name] == "temp":
            continue

        for output in rule_outputs:
            # Convert output to string
            output_str = str(output)

            # Use Snakemake's expand with zip for efficient path generation
            # zip tells expand to use corresponding items from each list rather than all combinations
            expanded_outputs = expand(output_str, zip, **wildcard_values)
            targets.extend(expanded_outputs)

    return targets


def output_to_input(
    output,
    wildcards={},
    expansion_values=[],
    metadata_combos=None,
    subset_values={},
    ancient_output=False,
):
    """Generates input file paths by expanding or subsetting a filepath template.

    This function allows one to retrieve file paths from a given template by:
    - Expanding on values: Generating all possible file paths by substituting wildcards with
      values from `metadata_combos`.
    - Subsetting on values: Filtering paths to include only specific values in `subset_values`.
    - Performing both expansion and subsetting.

    Args:
        output (str): Template file path with wildcards.
        wildcards (dict): Dictionary of fixed wildcard values.
        expansion_values (list): List of wildcard names to expand.
        metadata_combos (pd.DataFrame, optional): DataFrame containing all possible wildcard combinations.
        subset_values (dict): Dictionary of values to subset the final expanded paths.
        ancient_output (bool, optional): If True, marks all returned paths as ancient in Snakemake.

    Returns:
        list: A list of expanded and/or filtered file paths as strings.
    """
    # Get a single string output from a list
    if isinstance(output, list):
        if len(output) == 1:
            output = output[0]
        else:
            raise ValueError(
                "Expected a single string for 'output', but received a list with multiple items."
            )

    if metadata_combos is None:
        # Directly expand paths when metadata_combos is not provided
        expanded_paths = expand(str(output), **wildcards, **subset_values)
    else:
        # Filter metadata_combos based on fixed wildcards
        mask = (
            metadata_combos[list(wildcards.keys())] == pd.Series(dict(wildcards))
        ).all(axis=1)
        filtered_combos = metadata_combos[mask]

        # Extract relevant expansion values
        expansion_dicts = filtered_combos[expansion_values].to_dict(orient="records")

        # Expand paths using Snakemake's expand function
        expanded_paths = [
            expand(str(output), **wildcards, **subset_values, **combo)
            for combo in expansion_dicts
        ]

        # Flatten nested lists of paths
        expanded_paths = [path for sublist in expanded_paths for path in sublist]

    # Mark paths as ancient if requested
    if ancient_output:
        expanded_paths = [ancient(path) for path in expanded_paths]

    return expanded_paths


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
