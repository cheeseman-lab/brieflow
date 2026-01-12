from lib.shared.file_utils import get_filename
from snakemake.io import temp, directory
from lib.shared.target_utils import map_outputs, outputs_to_targets
from lib.preprocess.file_utils import get_output_pattern

PREPROCESS_FP = ROOT_FP / "preprocess"

# Determine output format from config
OME_ZARR_ENABLED = config.get("preprocess", {}).get("ome_zarr", {}).get("enabled", True)
IC_EXT = "zarr" if OME_ZARR_ENABLED else "tiff"

# Define conversion keys for use across all rule files
USE_OME_ZARR = OME_ZARR_ENABLED
CONVERT_SBS_KEY = "convert_sbs_omezarr" if USE_OME_ZARR else "convert_sbs"
CONVERT_PHENOTYPE_KEY = "convert_phenotype_omezarr" if USE_OME_ZARR else "convert_phenotype"

PREPROCESS_OUTPUTS = {
    "extract_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_filename(
            get_output_pattern(sbs_metadata_wildcard_combos), "metadata", "tsv"
        ),
    ],
    "combine_metadata_sbs": [
        PREPROCESS_FP / "metadata" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "combined_metadata", "parquet"
        ),
    ],
    "extract_metadata_phenotype": [
        PREPROCESS_FP / "metadata" / "phenotype" / get_filename(
            get_output_pattern(phenotype_metadata_wildcard_combos), "metadata", "tsv"
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
    "convert_sbs_omezarr": [
        PREPROCESS_FP / "images" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}", "cycle": "{cycle}"},
            "image", "zarr"
        ),
    ],
    "convert_phenotype_omezarr": [
        PREPROCESS_FP / "images" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "image", "zarr"
        ),
    ],
    "calculate_ic_sbs": [
        PREPROCESS_FP / "ic_fields" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}", "cycle": "{cycle}"},
            "ic_field", IC_EXT
        ),
    ],
    "calculate_ic_phenotype": [
        PREPROCESS_FP / "ic_fields" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}"},
            "ic_field", IC_EXT
        ),
    ],
    "export_sbs_preprocess_omezarr": [
        PREPROCESS_FP / "omezarr" / "sbs" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}", "cycle": "{cycle}"},
            "image",
            "zarr",
        ),
    ],
    "export_phenotype_preprocess_omezarr": [
        PREPROCESS_FP / "omezarr" / "phenotype" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "image",
            "zarr",
        ),
    ],
}

# Define output mappings FIRST (before filtering)
PREPROCESS_OUTPUT_MAPPINGS = {
    "extract_metadata_sbs": temp,
    "combine_metadata_sbs": None,
    "extract_metadata_phenotype": temp,
    "combine_metadata_phenotype": None,
    "convert_sbs": None,
    "convert_phenotype": None,
    "convert_sbs_omezarr": directory,
    "convert_phenotype_omezarr": directory,
    "calculate_ic_sbs": directory if OME_ZARR_ENABLED else None,
    "calculate_ic_phenotype": directory if OME_ZARR_ENABLED else None,
    "export_sbs_preprocess_omezarr": directory,
    "export_phenotype_preprocess_omezarr": directory,
}

# Filter outputs based on whether OME-Zarr is the primary format
if OME_ZARR_ENABLED:
    # Remove TIFF conversion targets when Zarr is primary
    PREPROCESS_OUTPUTS.pop("convert_sbs", None)
    PREPROCESS_OUTPUTS.pop("convert_phenotype", None)
else:
    # Remove Zarr conversion targets when TIFF is primary
    PREPROCESS_OUTPUTS.pop("convert_sbs_omezarr", None)
    PREPROCESS_OUTPUTS.pop("convert_phenotype_omezarr", None)

# Filter optional exports if not enabled
omezarr_enabled = config.get("output", {}).get("omezarr", {}).get("enabled", False)
after_steps = config.get("output", {}).get("omezarr", {}).get("after_steps", [])
if not (omezarr_enabled and "preprocess" in after_steps):
    PREPROCESS_OUTPUTS.pop("export_sbs_preprocess_omezarr", None)
    PREPROCESS_OUTPUTS.pop("export_phenotype_preprocess_omezarr", None)

# Map outputs after filtering
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
