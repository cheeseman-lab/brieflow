from lib.shared.target_utils import (
    map_outputs,
    outputs_to_targets,
)

PREPROCESS_FP = ROOT_FP / "preprocess"

PREPROCESS_OUTPUTS = {
    "extract_metadata_sbs": [
        PREPROCESS_FP
        / "metadata"
        / "sbs"
        / get_filename({"well": "{well}", "cycle": "{cycle}"}, "metadata", "tsv"),
    ],
    "extract_metadata_phenotype": [
        PREPROCESS_FP
        / "metadata"
        / "phenotype"
        / get_filename({"well": "{well}"}, "metadata", "tsv"),
    ],
    "convert_sbs": [
        PREPROCESS_FP
        / "images"
        / "sbs"
        / get_filename(
            {"well": "{well}", "tile": "{tile}", "cycle": "{cycle}"},
            "image",
            "tiff",
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
    "extract_metadata_phenotype": None,
    "convert_sbs": None,
    "convert_phenotype": None,
    "calculate_ic_sbs": None,
    "calculate_ic_phenotype": None,
}

PREPROCESS_WILDCARDS = {
    "well": SBS_WELLS,
    "tile": SBS_TILES,
    "cycle": SBS_CYCLES,
}

PREPROCESS_OUTPUTS_MAPPED = map_outputs(
    PREPROCESS_OUTPUTS,
    PREPROCESS_OUTPUT_MAPPINGS,
)

PREPROCESS_TARGETS = outputs_to_targets(
    PREPROCESS_OUTPUTS,
    PREPROCESS_WILDCARDS,
)

PREPROCESS_TARGETS_ALL = sum(PREPROCESS_TARGETS.values(), [])
