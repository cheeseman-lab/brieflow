from lib.shared.file_utils import get_nested_path
from lib.shared.target_utils import map_outputs, outputs_to_targets
from snakemake.io import directory


SBS_FP = ROOT_FP / "sbs"

SBS_IMG_FMT = IMG_FMT

# --- Conditional path helpers based on image format ---
if SBS_IMG_FMT == "zarr":
    from lib.shared.file_utils import get_hcs_nested_path

    _sbs_tile_loc = {"plate": "{plate}", "row": "{row}", "col": "{col}", "tile": "{tile}"}
    _sbs_well_loc = {"plate": "{plate}", "row": "{row}", "col": "{col}"}
    _sbs_plate_loc = {"plate": "{plate}"}

    def _sbs_img(info):
        return SBS_FP / get_hcs_nested_path(_sbs_tile_loc, info)

    def _sbs_label(info):
        return SBS_FP / get_hcs_nested_path(_sbs_tile_loc, info, subdirectory="labels")

    def _sbs_tsv(info):
        return SBS_FP / "tsvs" / get_nested_path(_sbs_tile_loc, info, "tsv")

    def _sbs_well_pq(info):
        return SBS_FP / "parquets" / get_nested_path(_sbs_well_loc, info, "parquet")

    def _sbs_plate_eval(subdir, info, ext):
        return SBS_FP / "eval" / subdir / get_nested_path(_sbs_plate_loc, info, ext)

    # Expansion helpers for rules
    _sbs_well_expand = ["row", "col"]
    _sbs_tile_expand = ["row", "col", "tile"]
else:
    _sbs_tile_loc = {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}
    _sbs_well_loc = {"plate": "{plate}", "well": "{well}"}
    _sbs_plate_loc = {"plate": "{plate}"}

    def _sbs_img(info):
        return SBS_FP / "images" / get_nested_path(_sbs_tile_loc, info, SBS_IMG_FMT)

    _sbs_label = _sbs_img  # No labels/ nesting for TIFF

    def _sbs_tsv(info):
        return SBS_FP / "tsvs" / get_nested_path(_sbs_tile_loc, info, "tsv")

    def _sbs_well_pq(info):
        return SBS_FP / "parquets" / get_nested_path(_sbs_well_loc, info, "parquet")

    def _sbs_plate_eval(subdir, info, ext):
        return SBS_FP / "eval" / subdir / get_nested_path(_sbs_plate_loc, info, ext)

    _sbs_well_expand = ["well"]
    _sbs_tile_expand = ["well", "tile"]


SBS_OUTPUTS = {
    "align_sbs": [_sbs_img("aligned")],
    "log_filter": [_sbs_img("log_filtered")],
    "compute_standard_deviation": [_sbs_img("standard_deviation")],
    "find_peaks": [_sbs_img("peaks")],
    "max_filter": [_sbs_img("max_filtered")],
    "apply_ic_field_sbs": [_sbs_img("illumination_corrected")],
    "segment_sbs": [
        _sbs_label("nuclei"),
        _sbs_label("cells"),
        _sbs_tsv("segmentation_stats"),
    ],
    "extract_bases": [_sbs_tsv("bases")],
    "call_reads": [_sbs_tsv("reads")],
    "call_cells": [_sbs_tsv("cells")],
    "extract_sbs_info": [_sbs_tsv("sbs_info")],
    "combine_reads": [_sbs_well_pq("reads")],
    "combine_cells": [_sbs_well_pq("cells")],
    "combine_sbs_info": [_sbs_well_pq("sbs_info")],
    "eval_segmentation_sbs": [
        _sbs_plate_eval("segmentation", "segmentation_overview", "tsv"),
        _sbs_plate_eval("segmentation", "cell_density_heatmap", "tsv"),
        _sbs_plate_eval("segmentation", "cell_density_heatmap", "png"),
    ],
    "eval_mapping": [
        _sbs_plate_eval("mapping", "mapping_vs_threshold_peak", "png"),
        _sbs_plate_eval("mapping", "mapping_vs_threshold_qmin", "png"),
        _sbs_plate_eval("mapping", "read_mapping_heatmap", "png"),
        _sbs_plate_eval("mapping", "cell_mapping_heatmap_one", "tsv"),
        _sbs_plate_eval("mapping", "cell_mapping_heatmap_one", "png"),
        _sbs_plate_eval("mapping", "cell_mapping_heatmap_any", "tsv"),
        _sbs_plate_eval("mapping", "cell_mapping_heatmap_any", "png"),
        _sbs_plate_eval("mapping", "cell_metric_histogram", "png"),
        _sbs_plate_eval("mapping", "gene_symbol_histogram", "png"),
        _sbs_plate_eval("mapping", "mapping_overview", "tsv"),
        _sbs_plate_eval("mapping", "barcode_prefix_matching", "png"),
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
