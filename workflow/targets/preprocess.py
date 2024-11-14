from snakemake.io import expand, temp, protected

from lib.shared.file_utils import get_filename


# Apply mappings to output types
def map_outputs(preprocess_outputs, mappings):
    mapped_outputs = {}
    for rule_name, outputs in preprocess_outputs.items():
        output_type = mappings.get(rule_name)
        if output_type == "temp":
            mapped_outputs[rule_name] = [temp(output) for output in outputs]
        elif output_type == "protected":
            mapped_outputs[rule_name] = [protected(output) for output in outputs]
        elif output_type is None:
            mapped_outputs[rule_name] = outputs
        else:
            raise ValueError(f"Invalid output type: {output_type}")
    return mapped_outputs


# mappings for rule outputs
# can map outputs of rules to None, temp, protected, etc
PREPROCESS_OUTPUT_MAPPINGS = {
    "extract_metadata_sbs": "temp",
    "extract_metadata_phenotype": None,
    "convert_sbs": "temp",
    "convert_phenotype": "temp",
    "calculate_ic_sbs": "protected",
    "calculate_ic_phenotype": "protected",
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
        ]
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
    return {
        "extract_metadata_sbs": expand(
            preprocess_fp
            / "metadata"
            / "sbs"
            / get_filename({"well": "{well}", "cycle": "{cycle}"}, "metadata", "tsv"),
            well=sbs_wells,
            cycle=sbs_cycles,
        ),
    }


def get_expanded_targets(output_paths, wildcards):
    """Wrap each output path template in expand and apply the specified wildcards."""
    expanded_targets = {}
    for rule_name, path_templates in output_paths.items():
        expanded_targets[rule_name] = [
            expand(path_template, **wildcards) for path_template in path_templates
        ]
    return expanded_targets


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
