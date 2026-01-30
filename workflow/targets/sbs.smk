from lib.shared.file_utils import get_nested_path
from lib.shared.target_utils import map_outputs, outputs_to_targets
from snakemake.io import directory


SBS_FP = ROOT_FP / "sbs"

# Determine image output format from config (default: tiff for backward compatibility)
SBS_IMG_FMT = config.get("sbs", {}).get("image_output_format", "tiff")

SBS_OUTPUTS = {
    "align_sbs": [
        SBS_FP
        / "images"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "aligned",
            SBS_IMG_FMT,
        ),
    ],
    "log_filter": [
        SBS_FP
        / "images"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "log_filtered",
            SBS_IMG_FMT,
        ),
    ],
    "compute_standard_deviation": [
        SBS_FP
        / "images"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "standard_deviation",
            SBS_IMG_FMT,
        ),
    ],
    "find_peaks": [
        SBS_FP
        / "images"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "peaks",
            SBS_IMG_FMT,
        ),
    ],
    "max_filter": [
        SBS_FP
        / "images"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "max_filtered",
            SBS_IMG_FMT,
        ),
    ],
    "apply_ic_field_sbs": [
        SBS_FP
        / "images"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "illumination_corrected",
            SBS_IMG_FMT,
        ),
    ],
    "segment_sbs": [
        SBS_FP
        / "images"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "nuclei",
            SBS_IMG_FMT,
        ),
        SBS_FP
        / "images"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "cells",
            SBS_IMG_FMT,
        ),
        SBS_FP
        / "tsvs"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "segmentation_stats",
            "tsv",
        ),
    ],
    "extract_bases": [
        SBS_FP
        / "tsvs"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "bases", "tsv"
        ),
    ],
    "call_reads": [
        SBS_FP
        / "tsvs"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "reads", "tsv"
        ),
    ],
    "call_cells": [
        SBS_FP
        / "tsvs"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "cells", "tsv"
        ),
    ],
    "extract_sbs_info": [
        SBS_FP
        / "tsvs"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "sbs_info", "tsv"
        ),
    ],
    "combine_reads": [
        SBS_FP
        / "parquets"
        / get_nested_path({"plate": "{plate}", "well": "{well}"}, "reads", "parquet"),
    ],
    "combine_cells": [
        SBS_FP
        / "parquets"
        / get_nested_path({"plate": "{plate}", "well": "{well}"}, "cells", "parquet"),
    ],
    "combine_sbs_info": [
        SBS_FP
        / "parquets"
        / get_nested_path(
            {"plate": "{plate}", "well": "{well}"}, "sbs_info", "parquet"
        ),
    ],
    "eval_segmentation_sbs": [
        SBS_FP
        / "eval"
        / "segmentation"
        / get_nested_path({"plate": "{plate}"}, "segmentation_overview", "tsv"),
        SBS_FP
        / "eval"
        / "segmentation"
        / get_nested_path({"plate": "{plate}"}, "cell_density_heatmap", "tsv"),
        SBS_FP
        / "eval"
        / "segmentation"
        / get_nested_path({"plate": "{plate}"}, "cell_density_heatmap", "png"),
    ],
    "eval_mapping": [
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "mapping_vs_threshold_peak", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "mapping_vs_threshold_qmin", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "read_mapping_heatmap", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "cell_mapping_heatmap_one", "tsv"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "cell_mapping_heatmap_one", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "cell_mapping_heatmap_any", "tsv"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "cell_mapping_heatmap_any", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "cell_metric_histogram", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "gene_symbol_histogram", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "mapping_overview", "tsv"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_nested_path({"plate": "{plate}"}, "barcode_prefix_matching", "png"),
    ],
}

# When outputting zarr, image outputs need directory() mapping and should not be temp
# (Snakemake can't reliably temp() a directory output)
_sbs_img_dir = directory if SBS_IMG_FMT == "zarr" else None
_sbs_img_temp = temp if SBS_IMG_FMT != "zarr" else None

SBS_OUTPUT_MAPPINGS = {
    "align_sbs": _sbs_img_temp,
    "log_filter": _sbs_img_temp,
    "compute_standard_deviation": _sbs_img_temp,
    "find_peaks": _sbs_img_temp,
    "max_filter": _sbs_img_temp,
    "apply_ic_field_sbs": _sbs_img_temp,
    "segment_sbs": [_sbs_img_dir, _sbs_img_dir, None] if SBS_IMG_FMT == "zarr" else None,
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
