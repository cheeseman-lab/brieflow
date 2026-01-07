from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets
from snakemake.io import directory


PREPROCESS_FP = ROOT_FP / "preprocess"

# Determine output format from config
output_formats = config.get("preprocess", {}).get("output_formats", ["zarr"])
if isinstance(output_formats, str):
    output_formats = [output_formats]

ENABLE_ZARR = "zarr" in output_formats
ENABLE_TIFF = "tiff" in output_formats

# Determine downstream input format
# Default to TIFF if enabled, otherwise Zarr
default_downstream = "tiff" if ENABLE_TIFF else "zarr"
downstream_format = config.get("preprocess", {}).get("downstream_input_format", default_downstream)

# IC field format follows downstream preference
IC_EXT = downstream_format

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
    "convert_sbs_omezarr": [
        PREPROCESS_FP
        / "omezarr"
        / "sbs"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "tile": "{tile}",
                "cycle": "{cycle}",
            },
            "image",
            "zarr",
        ),
    ],
    "convert_phenotype_omezarr": [
        PREPROCESS_FP
        / "omezarr"
        / "phenotype"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "tile": "{tile}",
            },
            "image",
            "zarr",
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
            IC_EXT,
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
            IC_EXT,
        ),
    ],
}

# Filter outputs based on config
if not ENABLE_TIFF:
    if "convert_sbs" in PREPROCESS_OUTPUTS:
        del PREPROCESS_OUTPUTS["convert_sbs"]
    if "convert_phenotype" in PREPROCESS_OUTPUTS:
        del PREPROCESS_OUTPUTS["convert_phenotype"]

if not ENABLE_ZARR:
    if "convert_sbs_omezarr" in PREPROCESS_OUTPUTS:
        del PREPROCESS_OUTPUTS["convert_sbs_omezarr"]
    if "convert_phenotype_omezarr" in PREPROCESS_OUTPUTS:
        del PREPROCESS_OUTPUTS["convert_phenotype_omezarr"]

# Convert all Paths to strings
for key in PREPROCESS_OUTPUTS:
    PREPROCESS_OUTPUTS[key] = [str(p) for p in PREPROCESS_OUTPUTS[key]]

PREPROCESS_OUTPUT_MAPPINGS = {
    "extract_metadata_sbs": temp,
    "combine_metadata_sbs": None,
    "extract_metadata_phenotype": temp,
    "combine_metadata_phenotype": None,
    "convert_sbs": None,
    "convert_phenotype": None,
    "convert_sbs_omezarr": directory,
    "convert_phenotype_omezarr": directory,
    "calculate_ic_sbs": directory if IC_EXT == "zarr" else None,
    "calculate_ic_phenotype": directory if IC_EXT == "zarr" else None,
}
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
