from lib.shared.file_utils import get_image_output_path, get_data_output_path
from lib.shared.target_utils import map_outputs, outputs_to_targets
from snakemake.io import directory


SBS_FP = ROOT_FP / "sbs"

SBS_IMG_FMT = IMG_FMT

# Location dicts (canonical form with {well}; dispatch functions handle zarr nesting)
_tile = {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}
_well = {"plate": "{plate}", "well": "{well}"}
_plate = {"plate": "{plate}"}

# Expansion helpers
_sbs_well_expand = ["row", "col"] if SBS_IMG_FMT == "zarr" else ["well"]
_sbs_tile_expand = ["row", "col", "tile"] if SBS_IMG_FMT == "zarr" else ["well", "tile"]


SBS_OUTPUTS = {
    "align_sbs": [
        SBS_FP / get_image_output_path(_tile, "aligned", SBS_IMG_FMT),
    ],
    "log_filter": [
        SBS_FP / get_image_output_path(_tile, "log_filtered", SBS_IMG_FMT),
    ],
    "compute_standard_deviation": [
        SBS_FP / get_image_output_path(_tile, "standard_deviation", SBS_IMG_FMT),
    ],
    "find_peaks": [
        SBS_FP / get_image_output_path(_tile, "peaks", SBS_IMG_FMT),
    ],
    "max_filter": [
        SBS_FP / get_image_output_path(_tile, "max_filtered", SBS_IMG_FMT),
    ],
    "apply_ic_field_sbs": [
        SBS_FP / get_image_output_path(_tile, "illumination_corrected", SBS_IMG_FMT),
    ],
    "segment_sbs": [
        SBS_FP / get_image_output_path(_tile, "nuclei", SBS_IMG_FMT, subdirectory="labels"),
        SBS_FP / get_image_output_path(_tile, "cells", SBS_IMG_FMT, subdirectory="labels"),
        SBS_FP / "tsvs" / get_data_output_path(_tile, "segmentation_stats", "tsv", SBS_IMG_FMT),
    ],
    "extract_bases": [
        SBS_FP / "tsvs" / get_data_output_path(_tile, "bases", "tsv", SBS_IMG_FMT),
    ],
    "call_reads": [
        SBS_FP / "tsvs" / get_data_output_path(_tile, "reads", "tsv", SBS_IMG_FMT),
    ],
    "call_cells": [
        SBS_FP / "tsvs" / get_data_output_path(_tile, "cells", "tsv", SBS_IMG_FMT),
    ],
    "extract_sbs_info": [
        SBS_FP / "tsvs" / get_data_output_path(_tile, "sbs_info", "tsv", SBS_IMG_FMT),
    ],
    "combine_reads": [
        SBS_FP / "parquets" / get_data_output_path(_well, "reads", "parquet", SBS_IMG_FMT),
    ],
    "combine_cells": [
        SBS_FP / "parquets" / get_data_output_path(_well, "cells", "parquet", SBS_IMG_FMT),
    ],
    "combine_sbs_info": [
        SBS_FP / "parquets" / get_data_output_path(_well, "sbs_info", "parquet", SBS_IMG_FMT),
    ],
    "eval_segmentation_sbs": [
        SBS_FP / "eval" / "segmentation" / get_data_output_path(_plate, "segmentation_overview", "tsv", SBS_IMG_FMT),
        SBS_FP / "eval" / "segmentation" / get_data_output_path(_plate, "cell_density_heatmap", "tsv", SBS_IMG_FMT),
        SBS_FP / "eval" / "segmentation" / get_data_output_path(_plate, "cell_density_heatmap", "png", SBS_IMG_FMT),
    ],
    "eval_mapping": [
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "mapping_vs_threshold_peak", "png", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "mapping_vs_threshold_qmin", "png", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "read_mapping_heatmap", "png", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "cell_mapping_heatmap_one", "tsv", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "cell_mapping_heatmap_one", "png", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "cell_mapping_heatmap_any", "tsv", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "cell_mapping_heatmap_any", "png", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "cell_metric_histogram", "png", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "gene_symbol_histogram", "png", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "mapping_overview", "tsv", SBS_IMG_FMT),
        SBS_FP / "eval" / "mapping" / get_data_output_path(_plate, "barcode_prefix_matching", "png", SBS_IMG_FMT),
    ],
}

# When outputting zarr, image outputs need directory() mapping and should not be temp
# (Snakemake can't reliably temp() a directory output)
# When outputting tiff, intermediate images can be temp() for cleanup
_sbs_img_temp = directory if SBS_IMG_FMT == "zarr" else temp
_sbs_img_keep = directory if SBS_IMG_FMT == "zarr" else None

SBS_OUTPUT_MAPPINGS = {
    "align_sbs": _sbs_img_temp,
    "log_filter": _sbs_img_temp,
    "compute_standard_deviation": _sbs_img_temp,
    "find_peaks": _sbs_img_temp,
    "max_filter": _sbs_img_temp,
    "apply_ic_field_sbs": _sbs_img_temp,
    "segment_sbs": [_sbs_img_keep, _sbs_img_keep, None],
    "extract_bases": temp,
    "call_reads": temp,
    "call_cells": temp,
    "extract_sbs_info": temp,
    "combine_reads": None,
    "combine_cells": None,
    "combine_sbs_info": None,
    "eval_segmentation_sbs": None,
    "eval_mapping": None,
}

SBS_OUTPUTS_MAPPED = map_outputs(SBS_OUTPUTS, SBS_OUTPUT_MAPPINGS)

SBS_TARGETS_ALL = outputs_to_targets(
    SBS_OUTPUTS, sbs_wildcard_combos, SBS_OUTPUT_MAPPINGS
)
