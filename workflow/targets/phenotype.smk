from lib.shared.file_utils import get_image_output_path, get_data_output_path
from lib.shared.target_utils import map_outputs, outputs_to_targets
from snakemake.io import directory


PHENOTYPE_FP = ROOT_FP / "phenotype"

PHENOTYPE_IMG_FMT = IMG_FMT

# determine feature eval outputs based on channel names and segment_cells setting
channel_names = config["phenotype"]["channel_names"]
segment_cells = config["phenotype"].get("segment_cells", True)
prefix = "cell" if segment_cells else "nucleus"
eval_features = [f"{prefix}_{channel}_min" for channel in channel_names]

# Location dicts (canonical form with {well}; dispatch functions handle zarr nesting)
_tile = {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}
_well = {"plate": "{plate}", "well": "{well}"}
_plate = {"plate": "{plate}"}

# Expansion helpers
_phen_well_expand = ["row", "col"] if PHENOTYPE_IMG_FMT == "zarr" else ["well"]
_phen_tile_expand = ["row", "col", "tile"] if PHENOTYPE_IMG_FMT == "zarr" else ["well", "tile"]


PHENOTYPE_OUTPUTS = {
    "apply_ic_field_phenotype": [
        PHENOTYPE_FP / get_image_output_path(_tile, "illumination_corrected", PHENOTYPE_IMG_FMT),
    ],
    "align_phenotype": [
        PHENOTYPE_FP / get_image_output_path(_tile, "aligned", PHENOTYPE_IMG_FMT),
    ],
    "segment_phenotype": [
        PHENOTYPE_FP / get_image_output_path(_tile, "nuclei", PHENOTYPE_IMG_FMT, subdirectory="labels"),
        PHENOTYPE_FP / get_image_output_path(_tile, "cells", PHENOTYPE_IMG_FMT, subdirectory="labels"),
        PHENOTYPE_FP / "tsvs" / get_data_output_path(_tile, "segmentation_stats", "tsv", PHENOTYPE_IMG_FMT),
    ],
    "identify_cytoplasm": [
        PHENOTYPE_FP / get_image_output_path(_tile, "identified_cytoplasms", PHENOTYPE_IMG_FMT, subdirectory="labels"),
    ],
    "extract_phenotype_info": [
        PHENOTYPE_FP / "tsvs" / get_data_output_path(_tile, "phenotype_info", "tsv", PHENOTYPE_IMG_FMT),
    ],
    "combine_phenotype_info": [
        PHENOTYPE_FP / "parquets" / get_data_output_path(_well, "phenotype_info", "parquet", PHENOTYPE_IMG_FMT),
    ],
    "extract_phenotype": [
        PHENOTYPE_FP / "tsvs" / get_data_output_path(_tile, "phenotype_cp", "tsv", PHENOTYPE_IMG_FMT),
    ],
    "merge_phenotype": [
        PHENOTYPE_FP / "parquets" / get_data_output_path(_well, "phenotype_cp", "parquet", PHENOTYPE_IMG_FMT),
        PHENOTYPE_FP / "parquets" / get_data_output_path(_well, "phenotype_cp_min", "parquet", PHENOTYPE_IMG_FMT),
    ],
    "eval_segmentation_phenotype": [
        PHENOTYPE_FP / "eval" / "segmentation" / get_data_output_path(_plate, "segmentation_overview", "tsv", PHENOTYPE_IMG_FMT),
        PHENOTYPE_FP / "eval" / "segmentation" / get_data_output_path(_plate, "cell_density_heatmap", "tsv", PHENOTYPE_IMG_FMT),
        PHENOTYPE_FP / "eval" / "segmentation" / get_data_output_path(_plate, "cell_density_heatmap", "png", PHENOTYPE_IMG_FMT),
    ],
    # create heatmap tsv and png for each evaluated feature
    "eval_features": [
        PHENOTYPE_FP / "eval" / "features" / get_data_output_path(_plate, f"{feature}_heatmap", "tsv", PHENOTYPE_IMG_FMT)
        for feature in eval_features
    ] + [
        PHENOTYPE_FP / "eval" / "features" / get_data_output_path(_plate, f"{feature}_heatmap", "png", PHENOTYPE_IMG_FMT)
        for feature in eval_features
    ],
}

# When outputting zarr, image outputs need directory() mapping and should not be temp
# (Snakemake can't reliably temp() a directory output)
# When outputting tiff, intermediate images can be temp() for cleanup
_phenotype_img_temp = directory if PHENOTYPE_IMG_FMT == "zarr" else temp
_phenotype_img_keep = directory if PHENOTYPE_IMG_FMT == "zarr" else None

PHENOTYPE_OUTPUT_MAPPINGS = {
    "apply_ic_field_phenotype": _phenotype_img_temp,
    "align_phenotype": _phenotype_img_keep,
    "segment_phenotype": [_phenotype_img_keep, _phenotype_img_keep, None],
    "identify_cytoplasm": _phenotype_img_temp,
    "extract_phenotype_info": temp,
    "combine_phenotype_info": None,
    "extract_phenotype": temp,
    "merge_phenotype": None,
    "eval_segmentation_phenotype": None,
    "eval_features": None,
}

PHENOTYPE_OUTPUTS_MAPPED = map_outputs(PHENOTYPE_OUTPUTS, PHENOTYPE_OUTPUT_MAPPINGS)

PHENOTYPE_TARGETS_ALL = outputs_to_targets(
    PHENOTYPE_OUTPUTS, phenotype_wildcard_combos, PHENOTYPE_OUTPUT_MAPPINGS
)
