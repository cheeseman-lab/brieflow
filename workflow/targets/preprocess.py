from snakemake.io import expand, temp, protected

from lib.shared.file_utils import get_filename


# mappings for rule outputs
# can map outputs of rules to None, temp, protected, etc
PREPROCESS_OUTPUT_MAPPINGS = {
    "extract_metadata_sbs": None,
    "extract_metadata_phenotype": None,
    "convert_sbs": None,
    "convert_phenotype": None,
    "calculate_ic_sbs": None,
    "calculate_ic_phenotype": temp,
}


def get_preprocess_outputs(
    preprocess_fp, sbs_wells, sbs_tiles, sbs_cycles, phenotype_wells, phenotype_tiles
):
    return {
        "extract_metadata_sbs": [
            preprocess_fp
            / "metadata"
            / "sbs"
            / get_filename({"well": "{well}", "cycle": "{cycle}"}, "metadata", "tsv"),
        ],
        "extract_metadata_phenotype": [
            preprocess_fp
            / "metadata"
            / "phenotype"
            / get_filename({"well": "{well}"}, "metadata", "tsv"),
        ],
        "convert_sbs": [
            preprocess_fp
            / "images"
            / "sbs"
            / get_filename(
                {"well": "{well}", "tile": "{tile}", "cycle": "{cycle}"},
                "image",
                "tiff",
            ),
        ],
        "convert_phenotype": [
            preprocess_fp
            / "images"
            / "phenotype"
            / get_filename({"well": "{well}", "tile": "{tile}"}, "image", "tiff"),
        ],
        "calculate_ic_sbs": [
            preprocess_fp
            / "ic_fields"
            / "sbs"
            / get_filename({"well": "{well}", "cycle": "{cycle}"}, "ic_field", "tiff"),
        ],
        "calculate_ic_phenotype": [
            preprocess_fp
            / "ic_fields"
            / "phenotype"
            / get_filename({"well": "{well}"}, "ic_field", "tiff"),
        ],
    }


def get_preprocess_mapped_outputs(
    preprocess_fp, sbs_wells, sbs_tiles, sbs_cycles, phenotype_wells, phenotype_tiles
):
    preprocess_outputs = get_preprocess_outputs(
        preprocess_fp,
        sbs_wells,
        sbs_tiles,
        sbs_cycles,
        phenotype_wells,
        phenotype_tiles,
    )
    preprocess_mapped_outputs = map_outputs(
        preprocess_outputs, PREPROCESS_OUTPUT_MAPPINGS
    )
    return preprocess_mapped_outputs


def get_preprocess_targets(
    preprocess_fp, sbs_wells, sbs_tiles, sbs_cycles, phenotype_wells, phenotype_tiles
):
    # Generate preprocess outputs without wildcards expanded
    preprocess_outputs = get_preprocess_outputs(
        preprocess_fp,
        sbs_wells,
        sbs_tiles,
        sbs_cycles,
        phenotype_wells,
        phenotype_tiles,
    )

    # Define the wildcards for each rule
    wildcards = {
        "well": sbs_wells,
        "tile": sbs_tiles,
        "cycle": sbs_cycles,
    }

    # Expand targets using the defined wildcards
    preprocess_targets = get_expanded_targets(preprocess_outputs, wildcards)
    return preprocess_targets


def map_outputs(preprocess_outputs, mappings):
    mapped_outputs = {}
    for rule_name, outputs in preprocess_outputs.items():
        output_func = mappings.get(rule_name)
        if output_func is None:
            # No mapping function, keep outputs as is
            mapped_outputs[rule_name] = outputs
        else:
            # Apply the function from the mapping (e.g., temp or protected)
            mapped_outputs[rule_name] = [output_func(output) for output in outputs]
    return mapped_outputs


def get_expanded_targets(output_paths, wildcards):
    """Wrap each output path template in expand and apply the specified wildcards."""
    expanded_targets = {}
    for rule_name, path_templates in output_paths.items():
        expanded_targets[rule_name] = [
            expand(path_template, **wildcards) for path_template in path_templates
        ]
    return expanded_targets


def get_rule_input(output_template, wildcard_values, wildcards):
    """Maps an output template to an input path using wildcards and additional mappings.

    Args:
        output_template (list): Template for output paths (e.g., PREPROCESS_OUTPUTS).
        wildcard_values (dict): Additional wildcard mappings (e.g., {"tile": SBS_TILES}).
        wildcards (dict): Current wildcards from Snakemake (e.g., wildcards.well, wildcards.cycle).

    Returns:
        list: Flattened expanded input paths.
    """
    # Merge wildcards with wildcard_values
    all_wildcards = {**wildcards, **wildcard_values}
    # Expand the output template with the merged wildcards and flatten
    return expand(output_template, **all_wildcards)
