from lib.shared.file_utils import get_nested_path
from snakemake.io import temp, directory
from lib.shared.target_utils import map_outputs, outputs_to_targets
from lib.preprocess.file_utils import get_output_pattern

PREPROCESS_FP = ROOT_FP / "preprocess"

IC_EXT = IMG_FMT

PREPROCESS_OUTPUTS = {
    "extract_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_nested_path(
            get_output_pattern(sbs_metadata_wildcard_combos), "metadata", "tsv"
        ),
    ],
    "combine_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_nested_path(
            {"plate": "{plate}", "well": "{well}"}, "combined_metadata", "parquet"
        ),
    ],
    "extract_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_nested_path(
            get_output_pattern(phenotype_metadata_wildcard_combos), "metadata", "tsv"
        ),
    ],
    "combine_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_nested_path(
            {"plate": "{plate}", "well": "{well}"}, "combined_metadata", "parquet"
        ),
    ],
    "convert_sbs": [
        PREPROCESS_FP / "images" / "sbs" / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}", "cycle": "{cycle}"},
            "image", IMG_FMT
        ),
    ],
    "convert_phenotype": [
        PREPROCESS_FP / "images" / "phenotype" / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "image", IMG_FMT
        ),
    ],
    "calculate_ic_sbs": [
        PREPROCESS_FP / "ic_fields" / "sbs" / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "cycle": "{cycle}"},
            "ic_field", IC_EXT
        ),
    ],
    "calculate_ic_phenotype": [
        PREPROCESS_FP / "ic_fields" / "phenotype" / get_nested_path(
            {"plate": "{plate}", "well": "{well}"},
            "ic_field", IC_EXT
        ),
    ],
}

# Define output mappings FIRST (before filtering)
PREPROCESS_OUTPUT_MAPPINGS = {
    "extract_metadata_sbs": temp,
    "combine_metadata_sbs": None,
    "extract_metadata_phenotype": temp,
    "combine_metadata_phenotype": None,
    "convert_sbs": directory if IMG_FMT == "zarr" else None,
    "convert_phenotype": directory if IMG_FMT == "zarr" else None,
    "calculate_ic_sbs": directory if IC_EXT == "zarr" else None,
    "calculate_ic_phenotype": directory if IC_EXT == "zarr" else None,
}

# Convert all Paths to strings
for key in PREPROCESS_OUTPUTS:
    PREPROCESS_OUTPUTS[key] = [str(p) for p in PREPROCESS_OUTPUTS[key]]

# Map outputs after filtering
PREPROCESS_OUTPUTS_MAPPED = map_outputs(PREPROCESS_OUTPUTS, PREPROCESS_OUTPUT_MAPPINGS)

# Generate sbs preprocessing targets
PREPROCESS_OUTPUTS_SBS = {
    rule_name: templates
    for rule_name, templates in PREPROCESS_OUTPUTS.items()
    if "sbs" in rule_name
}
PREPROCESS_TARGETS_SBS = outputs_to_targets(
    PREPROCESS_OUTPUTS_SBS, sbs_wildcard_combos, PREPROCESS_OUTPUT_MAPPINGS
)

# Generate phenotype preprocessing targets
PREPROCESS_OUTPUTS_PHENOTYPE = {
    rule_name: templates
    for rule_name, templates in PREPROCESS_OUTPUTS.items()
    if "phenotype" in rule_name
}
PREPROCESS_TARGETS_PHENOTYPE = outputs_to_targets(
    PREPROCESS_OUTPUTS_PHENOTYPE, phenotype_wildcard_combos, PREPROCESS_OUTPUT_MAPPINGS
)

# Combine all preprocessing targets
PREPROCESS_TARGETS_ALL = PREPROCESS_TARGETS_SBS + PREPROCESS_TARGETS_PHENOTYPE
