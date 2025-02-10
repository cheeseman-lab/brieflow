from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets, outputs_to_targets_with_combinations


PREPROCESS_FP = ROOT_FP / "preprocess"

PREPROCESS_OUTPUTS = {
    "extract_metadata_sbs": [
        PREPROCESS_FP
        / "metadata"
        / "sbs"
        / get_filename(
            {"well": "{well}", "cycle": "{cycle}", "channel": "{channel}"}, "metadata", "tsv"
        ),
    ],
    "combine_metadata_sbs": [
        PREPROCESS_FP
        / "metadata"
        / "sbs"
        / get_filename(
            {"cycle": "{cycle}", "channel": "{channel}"}, "combined_metadata", "hdf5"
        ),
    ],
    "extract_metadata_phenotype": [
        PREPROCESS_FP
        / "metadata"
        / "phenotype"
        / get_filename({"well": "{well}", "channel": "{channel}"}, "metadata", "tsv"),
    ],
    "combine_metadata_phenotype": [
        PREPROCESS_FP
        / "metadata"
        / "phenotype"
        / get_filename({"channel": "{channel}"}, "combined_metadata", "hdf5"),
    ],
    "convert_sbs": [
        PREPROCESS_FP
        / "images"
        / "sbs"
        / get_filename(
            {"well": "{well}", "cycle": "{cycle}", "tile": "{tile}"}, "image", "tiff"
        ),
    ],
    "convert_phenotype": [
        PREPROCESS_FP
        / "images"
        / "phenotype"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "image", "tiff"),
    ],
    "calculate_ic_sbs": [
        PREPROCESS_FP
        / "ic_fields"
        / "sbs"
        / get_filename({"well": "{well}", "cycle": "{cycle}"}, "ic_field", "tiff"),
    ],
    "calculate_ic_phenotype": [
        PREPROCESS_FP
        / "ic_fields"
        / "phenotype"
        / get_filename({"well": "{well}"}, "ic_field", "tiff"),
    ],
}

PREPROCESS_OUTPUT_MAPPINGS = {
    "extract_metadata_sbs": None,
    "combine_metadata_sbs": None,
    "extract_metadata_phenotype": None,
    "combine_metadata_phenotype": None,
    "convert_sbs": None,
    "convert_phenotype": None,
    "calculate_ic_sbs": None,
    "calculate_ic_phenotype": None,
}

PREPROCESS_OUTPUTS_MAPPED = map_outputs(PREPROCESS_OUTPUTS, PREPROCESS_OUTPUT_MAPPINGS)

# Generate SBS preprocessing targets
SBS_WILDCARDS = {
    "well": SBS_WELLS,
    "tile": SBS_TILES, 
    **{k: [d[k] for d in SBS_VALID_COMBINATIONS] for k in ["cycle", "channel"]}
}
PREPROCESS_OUTPUTS_SBS = {
    rule_name: templates
    for rule_name, templates in PREPROCESS_OUTPUTS.items()
    if "sbs" in rule_name
}
PREPROCESS_TARGETS_SBS = (
    outputs_to_targets_with_combinations(
        PREPROCESS_OUTPUTS["extract_metadata_sbs"],
        SBS_VALID_COMBINATIONS,
        SBS_WELLS
    ) + 
    outputs_to_targets_with_combinations(
        PREPROCESS_OUTPUTS["convert_sbs"],
        SBS_VALID_COMBINATIONS,
        SBS_WELLS,
        SBS_TILES
    ) +
    outputs_to_targets_with_combinations(
        PREPROCESS_OUTPUTS["calculate_ic_sbs"],
        SBS_VALID_COMBINATIONS,
        SBS_WELLS
    ) +
    outputs_to_targets_with_combinations(
        PREPROCESS_OUTPUTS["combine_metadata_sbs"],
        SBS_VALID_COMBINATIONS,
        SBS_WELLS
    )
)

# Generate phenotype preprocessing targets
PHENOTYPE_WILDCARDS = {
    "well": PHENOTYPE_WELLS,
    "tile": PHENOTYPE_TILES,
    "channel": PHENOTYPE_CHANNELS,
}
PREPROCESS_OUTPUTS_PHENOTYPE = {
    rule_name: templates
    for rule_name, templates in PREPROCESS_OUTPUTS.items()
    if "phenotype" in rule_name
}
PREPROCESS_TARGETS_PHENOTYPE = (
    outputs_to_targets_with_combinations(
        PREPROCESS_OUTPUTS["extract_metadata_phenotype"],
        [{"channel": ch} for ch in PHENOTYPE_CHANNELS],
        PHENOTYPE_WELLS
    ) +
    outputs_to_targets_with_combinations(
        PREPROCESS_OUTPUTS["convert_phenotype"],
        [{"channel": ch} for ch in PHENOTYPE_CHANNELS],
        PHENOTYPE_WELLS,
        PHENOTYPE_TILES
    ) +
    outputs_to_targets_with_combinations(
        PREPROCESS_OUTPUTS["calculate_ic_phenotype"],
        [{"channel": ch} for ch in PHENOTYPE_CHANNELS],
        PHENOTYPE_WELLS
    ) +
    outputs_to_targets_with_combinations(
        PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
        [{"channel": ch} for ch in PHENOTYPE_CHANNELS],
        PHENOTYPE_WELLS
    )
)


# Combine all preprocessing targets
PREPROCESS_TARGETS_ALL = PREPROCESS_TARGETS_SBS + PREPROCESS_TARGETS_PHENOTYPE