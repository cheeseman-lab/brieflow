from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets
from lib.preprocess.preprocess import (
    get_metadata_extraction_wildcards, 
    get_metadata_output_wildcards
)

PREPROCESS_FP = ROOT_FP / "preprocess"

# Get metadata-specific wildcard combinations using your functions
sbs_metadata_wildcard_combos = get_metadata_extraction_wildcards(
    "sbs", config, sbs_samples_df, sbs_metadata_samples_df
)
phenotype_metadata_wildcard_combos = get_metadata_extraction_wildcards(
    "phenotype", config, phenotype_samples_df, phenotype_metadata_samples_df
)

# Define output patterns dynamically using your functions
PREPROCESS_OUTPUTS = {
    "extract_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_filename(
            get_metadata_output_wildcards("sbs", config, sbs_samples_df, sbs_metadata_samples_df),
            "metadata", "tsv"
        ),
    ],
    "combine_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "combined_metadata", "parquet"
        ),
    ],
    "extract_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_filename(
            get_metadata_output_wildcards("phenotype", config, phenotype_samples_df, phenotype_metadata_samples_df),
            "metadata", "tsv"
        ),
    ],
    "combine_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "combined_metadata", "parquet"
        ),
    ],
    "convert_sbs": [
        PREPROCESS_FP / "images" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}", "cycle": "{cycle}"},
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
    "combine_metadata_sbs": None,
    "extract_metadata_phenotype": temp,
    "combine_metadata_phenotype": None,
    "convert_sbs": None,
    "convert_phenotype": None,
    "calculate_ic_sbs": None,
    "calculate_ic_phenotype": None,
}
PREPROCESS_OUTPUTS_MAPPED = map_outputs(PREPROCESS_OUTPUTS, PREPROCESS_OUTPUT_MAPPINGS)

# Generate targets using the appropriate wildcard combinations
# Metadata extraction uses metadata-specific wildcards
PREPROCESS_TARGETS_SBS_METADATA = outputs_to_targets(
    {"extract_metadata_sbs": PREPROCESS_OUTPUTS["extract_metadata_sbs"]},
    sbs_metadata_wildcard_combos,
    PREPROCESS_OUTPUT_MAPPINGS
)

PREPROCESS_TARGETS_PHENOTYPE_METADATA = outputs_to_targets(
    {"extract_metadata_phenotype": PREPROCESS_OUTPUTS["extract_metadata_phenotype"]},
    phenotype_metadata_wildcard_combos,
    PREPROCESS_OUTPUT_MAPPINGS
)

# Other targets use regular wildcard combinations
PREPROCESS_OUTPUTS_SBS_OTHER = {
    k: v for k, v in PREPROCESS_OUTPUTS.items() 
    if "sbs" in k and not k.startswith("extract_metadata")
}
PREPROCESS_TARGETS_SBS_OTHER = outputs_to_targets(
    PREPROCESS_OUTPUTS_SBS_OTHER, sbs_wildcard_combos, PREPROCESS_OUTPUT_MAPPINGS
)

PREPROCESS_OUTPUTS_PHENOTYPE_OTHER = {
    k: v for k, v in PREPROCESS_OUTPUTS.items() 
    if "phenotype" in k and not k.startswith("extract_metadata")
}
PREPROCESS_TARGETS_PHENOTYPE_OTHER = outputs_to_targets(
    PREPROCESS_OUTPUTS_PHENOTYPE_OTHER, phenotype_wildcard_combos, PREPROCESS_OUTPUT_MAPPINGS
)

# Combine all preprocessing targets
PREPROCESS_TARGETS_ALL = (
    PREPROCESS_TARGETS_SBS_METADATA + PREPROCESS_TARGETS_SBS_OTHER +
    PREPROCESS_TARGETS_PHENOTYPE_METADATA + PREPROCESS_TARGETS_PHENOTYPE_OTHER
)