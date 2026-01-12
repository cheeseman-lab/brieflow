"""Helper functions for using Snakemake outputs and targets for use with Brieflow."""

from pathlib import Path
import string

import pandas as pd
from snakemake.io import expand, ancient, temp

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
        if module_target_mappings[rule_name] == temp:
            continue

        for output in rule_outputs:
            # Convert output to string
            output_str = str(output)

            # Use Snakemake's expand with zip for efficient path generation
            # Check if output_str contains any wildcard placeholders (i.e., "{")
            if "{" in output_str and "}" in output_str:
                expanded_outputs = expand(output_str, zip, **wildcard_values)
                targets.extend(expanded_outputs)
            else:
                targets.append(output_str)

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
    # Normalize "output" to a single template (str/Path). Many call sites pass a single
    # Path template (not a list), so we handle both.
    if isinstance(output, list):
        if len(output) != 1:
            raise ValueError(
                "Expected a single template for 'output', but received a list with multiple items."
            )
        output = output[0]

    if metadata_combos is None:
        # Directly expand paths when metadata_combos is not provided
        expanded_paths = expand(str(output), **wildcards, **subset_values)
    else:
        # Filter metadata_combos based on fixed wildcards
        mask = pd.Series(True, index=metadata_combos.index)
        for key, value in wildcards.items():
            # Convert both sides to string to ensure matching types
            if key in metadata_combos.columns:
                mask &= metadata_combos[key].astype(str) == str(value)

        filtered_combos = metadata_combos[mask]

        # Extract relevant expansion values
        selected_cols_df = filtered_combos[expansion_values]

        if len(expansion_values) == 1:
            expansion_dicts = [
                {expansion_values[0]: value} for value in selected_cols_df.iloc[:, 0]
            ]
        else:
            expansion_dicts = selected_cols_df.to_dict(orient="records")

        expanded_paths = []
        for combo in expansion_dicts:
            all_wildcards_for_expand = {}
            all_wildcards_for_expand.update(wildcards)  # fixed wildcards
            all_wildcards_for_expand.update(subset_values)  # subset wildcards
            all_wildcards_for_expand.update(combo)  # expansion wildcards

            expanded_paths.extend(expand(str(output), **all_wildcards_for_expand))

        # Flatten nested lists of paths (if expand returns list of lists)
        if any(isinstance(i, list) for i in expanded_paths):
            expanded_paths = [path for sublist in expanded_paths for path in sublist]

        # Remove duplicates while preserving order
        expanded_paths = list(dict.fromkeys(expanded_paths))

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


def map_wildcard_outputs(wildcard_combos_df, output_template, wildcards_to_map):
    """Map specified wildcards in a template string using values from a DataFrame.

    Given a template path and a list of wildcards to map (e.g. ["cell_class", "channel_combo"]),
    replaces only those placeholders with values from the DataFrame, leaving others untouched.
    Useful for creating lists of output paths where we need to fill in some wildcards but not others.

    Args:
        wildcard_combos_df (pd.DataFrame): DataFrame with one column per wildcard.
        output_template (str): Template string with placeholders.
        wildcards_to_map (list[str]): List of wildcard names to substitute.

    Returns:
        list[str]: List of template paths with specified wildcards substituted.
    """
    output_template = str(output_template)
    all_wildcards = [
        field_name
        for _, field_name, _, _ in string.Formatter().parse(output_template)
        if field_name
    ]

    mapped_paths = []
    wildcard_combos_df = wildcard_combos_df[wildcards_to_map].drop_duplicates()
    for _, row in wildcard_combos_df.iterrows():
        mapped_path = output_template.format(
            **{
                w: row[w] if w in wildcards_to_map else f"{{{w}}}"
                for w in all_wildcards
            }
        )
        mapped_paths.append(mapped_path)

    return mapped_paths
