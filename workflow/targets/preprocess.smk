from snakemake.io import temp, directory
from lib.shared.file_utils import get_image_output_path, get_data_output_path
from lib.shared.target_utils import map_outputs, outputs_to_targets
from lib.preprocess.file_utils import get_output_pattern

PREPROCESS_FP = ROOT_FP / "preprocess"

IC_EXT = IMG_FMT

# Location dicts (canonical form with {well}; dispatch functions handle zarr nesting)
_pp_sbs_tile = {"plate": "{plate}", "well": "{well}", "tile": "{tile}", "cycle": "{cycle}"}
_pp_phen_tile = {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}
_pp_sbs_ic = {"plate": "{plate}", "well": "{well}", "cycle": "{cycle}"}
_pp_phen_ic = {"plate": "{plate}", "well": "{well}"}

# Expansion helpers
_pp_well_expand = ["row", "col"] if IMG_FMT == "zarr" else ["well"]

# Metadata output patterns (filter out row/col added by split_well_to_cols in zarr mode)
_pp_metadata_sbs_loc = {
    k: v for k, v in get_output_pattern(sbs_metadata_wildcard_combos).items()
    if k not in ("row", "col")
}
_pp_metadata_phen_loc = {
    k: v for k, v in get_output_pattern(phenotype_metadata_wildcard_combos).items()
    if k not in ("row", "col")
}
_pp_combine_metadata_loc = {"plate": "{plate}", "well": "{well}"}

PREPROCESS_OUTPUTS = {
    "extract_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_data_output_path(
            _pp_metadata_sbs_loc, "metadata", "tsv", IMG_FMT
        ),
    ],
    "combine_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_data_output_path(
            _pp_combine_metadata_loc, "combined_metadata", "parquet", IMG_FMT
        ),
    ],
    "extract_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_data_output_path(
            _pp_metadata_phen_loc, "metadata", "tsv", IMG_FMT
        ),
    ],
    "combine_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_data_output_path(
            _pp_combine_metadata_loc, "combined_metadata", "parquet", IMG_FMT
        ),
    ],
    "convert_sbs": [
        PREPROCESS_FP / get_image_output_path(_pp_sbs_tile, "image", IMG_FMT, image_subdir="sbs"),
    ],
    "convert_phenotype": [
        PREPROCESS_FP / get_image_output_path(_pp_phen_tile, "image", IMG_FMT, image_subdir="phenotype"),
    ],
    "calculate_ic_sbs": [
        PREPROCESS_FP / "ic_fields" / "sbs" / get_data_output_path(_pp_sbs_ic, "ic_field", IC_EXT, IMG_FMT),
    ],
    "calculate_ic_phenotype": [
        PREPROCESS_FP / "ic_fields" / "phenotype" / get_data_output_path(_pp_phen_ic, "ic_field", IC_EXT, IMG_FMT),
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
