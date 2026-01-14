from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


SBS_FP = ROOT_FP / "sbs"

SBS_OUTPUTS = {
    "align_sbs": [
        SBS_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "aligned", "tiff"
        ),
    ],
    "log_filter": [
        SBS_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "log_filtered",
            "tiff",
        ),
    ],
    "compute_standard_deviation": [
        SBS_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "standard_deviation",
            "tiff",
        ),
    ],
    "find_peaks": [
        SBS_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "peaks", "tiff"
        ),
    ],
    "max_filter": [
        SBS_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "max_filtered",
            "tiff",
        ),
    ],
    "apply_ic_field_sbs": [
        SBS_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "illumination_corrected",
            "tiff",
        ),
    ],
    "segment_sbs": [
        SBS_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "nuclei", "tiff"
        ),
        SBS_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "cells", "tiff"
        ),
        SBS_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "segmentation_stats",
            "tsv",
        ),
    ],
    "extract_bases": [
        SBS_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "bases", "tsv"
        ),
    ],
    "call_reads": [
        SBS_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "reads", "tsv"
        ),
    ],
    "call_cells": [
        SBS_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "cells", "tsv"
        ),
    ],
    "extract_sbs_info": [
        SBS_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "sbs_info", "tsv"
        ),
    ],
    "combine_reads": [
        SBS_FP
        / "parquets"
        / get_filename({"plate": "{plate}", "well": "{well}"}, "reads", "parquet"),
    ],
    "combine_cells": [
        SBS_FP
        / "parquets"
        / get_filename({"plate": "{plate}", "well": "{well}"}, "cells", "parquet"),
    ],
    "combine_sbs_info": [
        SBS_FP
        / "parquets"
        / get_filename({"plate": "{plate}", "well": "{well}"}, "sbs_info", "parquet"),
    ],
    "eval_segmentation_sbs": [
        SBS_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "segmentation_overview", "tsv"),
        SBS_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "cell_density_heatmap", "tsv"),
        SBS_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "cell_density_heatmap", "png"),
    ],
    "eval_mapping": [
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "mapping_vs_threshold_peak", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "mapping_vs_threshold_qmin", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "read_mapping_heatmap", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "cell_mapping_heatmap_one", "tsv"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "cell_mapping_heatmap_one", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "cell_mapping_heatmap_any", "tsv"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "cell_mapping_heatmap_any", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "cell_metric_histogram", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "gene_symbol_histogram", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "mapping_overview", "tsv"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "barcode_prefix_matching", "png"),
    ],
    "export_sbs_omezarr": [
        SBS_FP / "omezarr" / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "sbs_aligned",
            "zarr",
        ),
    ],
}

# Filter exports if not enabled
omezarr_enabled = config.get("output", {}).get("omezarr", {}).get("enabled", False)
after_steps = config.get("output", {}).get("omezarr", {}).get("after_steps", [])
if not (omezarr_enabled and "sbs" in after_steps):
    SBS_OUTPUTS.pop("export_sbs_omezarr", None)

SBS_OUTPUT_MAPPINGS = {
    "align_sbs": temp,
    "log_filter": temp,
    "compute_standard_deviation": temp,
    "find_peaks": temp,
    "max_filter": temp,
    "apply_ic_field_sbs": temp,
    "segment_sbs": None,
    "extract_bases": temp,
    "call_reads": temp,
    "call_cells": temp,
    "extract_sbs_info": temp,
    "combine_reads": None,
    "combine_cells": None,
    "combine_sbs_info": None,
    "eval_segmentation_sbs": None,
    "eval_mapping": None,
    "export_sbs_omezarr": directory,
}

SBS_OUTPUTS_MAPPED = map_outputs(SBS_OUTPUTS, SBS_OUTPUT_MAPPINGS)

SBS_TARGETS_ALL = outputs_to_targets(
    SBS_OUTPUTS, sbs_wildcard_combos, SBS_OUTPUT_MAPPINGS
)
