from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


PREPROCESS_FP = ROOT_FP / "preprocess"

PREPROCESS_OUTPUTS = {
    "extract_metadata_sbs": [
        PREPROCESS_FP
        / "metadata"
        / "sbs"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "tile": "{tile}",
                "cycle": "{cycle}",
            },
            "metadata",
            "tsv",
        ),
    ],
    "combine_metadata_sbs": [
        PREPROCESS_FP
        / "metadata"
        / "sbs"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
            },
            "combined_metadata",
            "parquet",
        ),
    ],
    "extract_metadata_phenotype": [
        PREPROCESS_FP
        / "metadata"
        / "phenotype"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "tile": "{tile}",
            },
            "metadata",
            "tsv",
        ),
    ],
    "combine_metadata_phenotype": [
        PREPROCESS_FP
        / "metadata"
        / "phenotype"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
            },
            "combined_metadata",
            "parquet",
        ),
    ],
    "convert_sbs": [
        PREPROCESS_FP
        / "images"
        / "sbs"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "tile": "{tile}",
                "cycle": "{cycle}",
            },
            "image",
            "tiff",
        ),
    ],
    "convert_phenotype": [
        PREPROCESS_FP
        / "images"
        / "phenotype"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "tile": "{tile}",
            },
            "image",
            "tiff",
        ),
    ],
    "calculate_ic_sbs": [
        PREPROCESS_FP
        / "ic_fields"
        / "sbs"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "cycle": "{cycle}",
            },
            "ic_field",
            "tiff",
        ),
    ],
    "calculate_ic_phenotype": [
        PREPROCESS_FP
        / "ic_fields"
        / "phenotype"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
            },
            "ic_field",
            "tiff",
        ),
    ],
}

PREPROCESS_OUTPUT_MAPPINGS = {
    "extract_metadata_sbs": temp,
    "combine_metadata_sbs": protected,
    "extract_metadata_phenotype": temp,
    "combine_metadata_phenotype": protected,
    "convert_sbs": None,
    "convert_phenotype": None,
    "calculate_ic_sbs": None,
    "calculate_ic_phenotype": None,
}
PREPROCESS_OUTPUTS_MAPPED = map_outputs(PREPROCESS_OUTPUTS, PREPROCESS_OUTPUT_MAPPINGS)

# Generate SBS preprocessing targets
SBS_WILDCARDS = {
    "plate": SBS_PLATES,
    "well": SBS_WELLS,
    "tile": SBS_TILES,
    "cycle": SBS_CYCLES,
}
PREPROCESS_OUTPUTS_SBS = {
    rule_name: templates
    for rule_name, templates in PREPROCESS_OUTPUTS.items()
    if "sbs" in rule_name
}
PREPROCESS_TARGETS_SBS = outputs_to_targets(
    PREPROCESS_OUTPUTS_SBS, SBS_WILDCARDS, PREPROCESS_OUTPUT_MAPPINGS
)

# Generate phenotype preprocessing targets
PHENOTYPE_WILDCARDS = {
    "plate": PHENOTYPE_PLATES,
    "well": PHENOTYPE_WELLS,
    "tile": PHENOTYPE_TILES,
}
PREPROCESS_OUTPUTS_PHENOTYPE = {
    rule_name: templates
    for rule_name, templates in PREPROCESS_OUTPUTS.items()
    if "phenotype" in rule_name
}
PREPROCESS_TARGETS_PHENOTYPE = outputs_to_targets(
    PREPROCESS_OUTPUTS_PHENOTYPE, PHENOTYPE_WILDCARDS, PREPROCESS_OUTPUT_MAPPINGS
)
# Combine all preprocessing targets
PREPROCESS_TARGETS_ALL = sum(PREPROCESS_TARGETS_SBS.values(), []) + sum(
    PREPROCESS_TARGETS_PHENOTYPE.values(), []
)
