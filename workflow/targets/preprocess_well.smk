from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets_with_combinations

PREPROCESS_FP = ROOT_FP / "preprocess"

PREPROCESS_OUTPUTS = {
    "extract_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}", "cycle": "{cycle}", "channel": "{channel}"}, 
            "metadata", "tsv"
        ),
    ],
    "combine_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}", "channel": "{channel}"}, 
            "combined_metadata", "parquet"
        ),
    ],
    "extract_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}", "channel": "{channel}"}, 
            "metadata", "tsv"
        ),
    ],
    "combine_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}", "channel": "{channel}"}, 
            "combined_metadata", "parquet"
        ),
    ],
    "convert_sbs": [
        PREPROCESS_FP / "images" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}", "cycle": "{cycle}", "tile": "{tile}"}, 
            "image", "tiff"
        ),
    ],
    "convert_phenotype": [
        PREPROCESS_FP / "images" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, 
            "image", "tiff"
        ),
    ],
    "calculate_ic_sbs": [
        PREPROCESS_FP / "ic_fields" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}", "cycle": "{cycle}"}, 
            "ic_field", "tiff"
        ),
    ],
    "calculate_ic_phenotype": [
        PREPROCESS_FP / "ic_fields" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, 
            "ic_field", "tiff"
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
    "calculate_ic_sbs": protected,
    "calculate_ic_phenotype": protected,
}

PREPROCESS_OUTPUTS_MAPPED = map_outputs(PREPROCESS_OUTPUTS, PREPROCESS_OUTPUT_MAPPINGS)

# Generate SBS preprocessing targets
PREPROCESS_TARGETS_SBS = (
    outputs_to_targets_with_combinations(
        output_templates=PREPROCESS_OUTPUTS["extract_metadata_sbs"],
        valid_combinations=SBS_VALID_COMBINATIONS,
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PREPROCESS_OUTPUTS["combine_metadata_sbs"],
        valid_combinations=SBS_VALID_COMBINATIONS
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PREPROCESS_OUTPUTS["convert_sbs"],
        valid_combinations=SBS_VALID_COMBINATIONS,
        extra_keys=SBS_TILES
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PREPROCESS_OUTPUTS["calculate_ic_sbs"],
        valid_combinations=SBS_VALID_COMBINATIONS
    )
)

# Generate phenotype preprocessing targets
PREPROCESS_TARGETS_PHENOTYPE = (
    outputs_to_targets_with_combinations(
        output_templates=PREPROCESS_OUTPUTS["extract_metadata_phenotype"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS,
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PREPROCESS_OUTPUTS["combine_metadata_phenotype"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PREPROCESS_OUTPUTS["convert_phenotype"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS,
        extra_keys=PHENOTYPE_TILES
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PREPROCESS_OUTPUTS["calculate_ic_phenotype"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS
    ) 
)

# Combine all preprocessing targets
PREPROCESS_TARGETS_ALL = PREPROCESS_TARGETS_SBS + PREPROCESS_TARGETS_PHENOTYPE