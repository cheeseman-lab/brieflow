from lib.shared.file_utils import get_nested_path
from snakemake.io import temp, directory
from lib.shared.target_utils import map_outputs, outputs_to_targets
from lib.preprocess.file_utils import get_output_pattern

PREPROCESS_FP = ROOT_FP / "preprocess"

IC_EXT = IMG_FMT

# --- Conditional path helpers based on image format ---
if IMG_FMT == "zarr":
    from lib.shared.file_utils import get_hcs_nested_path

    # Tile-level location for image outputs (HCS layout)
    _pp_sbs_tile_loc = {"plate": "{plate}", "row": "{row}", "col": "{col}", "tile": "{tile}", "cycle": "{cycle}"}
    _pp_phen_tile_loc = {"plate": "{plate}", "row": "{row}", "col": "{col}", "tile": "{tile}"}
    # Well-level location for IC fields (no HCS, just row/col instead of well)
    _pp_sbs_ic_loc = {"plate": "{plate}", "row": "{row}", "col": "{col}", "cycle": "{cycle}"}
    _pp_phen_ic_loc = {"plate": "{plate}", "row": "{row}", "col": "{col}"}

    def _pp_sbs_img(info):
        return PREPROCESS_FP / "sbs" / get_hcs_nested_path(_pp_sbs_tile_loc, info)

    def _pp_phen_img(info):
        return PREPROCESS_FP / "phenotype" / get_hcs_nested_path(_pp_phen_tile_loc, info)

    def _pp_sbs_ic(info):
        return PREPROCESS_FP / "ic_fields" / "sbs" / get_nested_path(_pp_sbs_ic_loc, info, IC_EXT)

    def _pp_phen_ic(info):
        return PREPROCESS_FP / "ic_fields" / "phenotype" / get_nested_path(_pp_phen_ic_loc, info, IC_EXT)

    # Expansion helpers for rules
    _pp_well_expand = ["row", "col"]
else:
    def _pp_sbs_img(info):
        return PREPROCESS_FP / "images" / "sbs" / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}", "cycle": "{cycle}"}, info, IMG_FMT
        )

    def _pp_phen_img(info):
        return PREPROCESS_FP / "images" / "phenotype" / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, info, IMG_FMT
        )

    def _pp_sbs_ic(info):
        return PREPROCESS_FP / "ic_fields" / "sbs" / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "cycle": "{cycle}"}, info, IC_EXT
        )

    def _pp_phen_ic(info):
        return PREPROCESS_FP / "ic_fields" / "phenotype" / get_nested_path(
            {"plate": "{plate}", "well": "{well}"}, info, IC_EXT
        )

    _pp_well_expand = ["well"]


PREPROCESS_OUTPUTS = {
    # Metadata rules always use {well} (not {row}/{col}) — they produce TSVs/parquets
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
    # Image and IC rules use conditional paths
    "convert_sbs": [_pp_sbs_img("image")],
    "convert_phenotype": [_pp_phen_img("image")],
    "calculate_ic_sbs": [_pp_sbs_ic("ic_field")],
    "calculate_ic_phenotype": [_pp_phen_ic("ic_field")],
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
