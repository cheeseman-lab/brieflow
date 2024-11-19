from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


SBS_PROCESS_FP = ROOT_FP / "sbs_process"

SBS_PROCESS_OUTPUTS = {
    "align": [
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "aligned", "tiff"),
    ],
    "log_filter": [
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "log_filtered", "tiff"),
    ],
    "compute_standard_deviation": [
        SBS_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "standard_deviation", "tiff"
        ),
    ],
    "find_peaks": [
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "peaks", "tiff"),
    ],
    "max_filter": [
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "max_filtered", "tiff"),
    ],
    "apply_ic_field": [
        SBS_PROCESS_FP
        / "images"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "illumination_corrected", "tiff"
        ),
    ],
    "segment": [
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "nuclei", "tiff"),
        SBS_PROCESS_FP
        / "images"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "cells", "tiff"),
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename(
            {"well": "{well}", "tile": "{tile}"}, "segmentation_stats", "tsv"
        ),
    ],
    "extract_bases": [
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "bases", "tsv"),
    ],
    "call_reads": [
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "reads", "tsv"),
    ],
    "call_cells": [
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "cells", "tsv"),
    ],
    "extract_sbs_info": [
        SBS_PROCESS_FP
        / "tsvs"
        / get_filename({"well": "{well}", "tile": "{tile}"}, "sbs_info", "tsv"),
    ],
    "combine_reads": [
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "reads", "hdf5"),
    ],
    "combine_cells": [
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "cells", "hdf5"),
    ],
    "combine_sbs_info": [
        SBS_PROCESS_FP / "hdfs" / get_filename({}, "sbs_info", "hdf5"),
    ],
    "eval_segmentation": [
        SBS_PROCESS_FP / "eval" / "segmentation" / "segmentation_overview.tsv",
        SBS_PROCESS_FP / "eval" / "segmentation" / "cell_density_heatmap.tsv",
        SBS_PROCESS_FP / "eval" / "segmentation" / "cell_density_heatmap.png",
    ],
    "eval_mapping": [
        SBS_PROCESS_FP / "eval" / "mapping" / "mapping_vs_threshold_peak.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "mapping_vs_threshold_qmin.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "read_mapping_heatmap.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "cell_mapping_heatmap_one.tsv",
        SBS_PROCESS_FP / "eval" / "mapping" / "cell_mapping_heatmap_one.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "cell_mapping_heatmap_any.tsv",
        SBS_PROCESS_FP / "eval" / "mapping" / "cell_mapping_heatmap_any.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "reads_per_cell_histogram.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "gene_symbol_histogram.png",
        SBS_PROCESS_FP / "eval" / "mapping" / "mapping_overview.tsv",
    ],
}

SBS_PROCESS_OUTPUT_MAPPINGS = {
    "align": None,
    "log_filter": None,
    "compute_standard_deviation": None,
    "find_peaks": None,
    "max_filter": None,
    "apply_ic_field": None,
    "segment": None,
    "extract_bases": None,
    "call_reads": None,
    "call_cells": None,
    "extract_sbs_info": None,
    "combine_reads": None,
    "combine_cells": None,
    "combine_sbs_info": None,
    "eval_segmentation": None,
    "eval_mapping": None,
}

SBS_PROCESS_WILDCARDS = {
    "well": SBS_WELLS,
    "tile": SBS_TILES,
    "cycle": SBS_CYCLES,
}

SBS_PROCESS_OUTPUTS_MAPPED = map_outputs(
    SBS_PROCESS_OUTPUTS, SBS_PROCESS_OUTPUT_MAPPINGS
)

SBS_PROCESS_TARGETS = outputs_to_targets(SBS_PROCESS_OUTPUTS, SBS_PROCESS_WILDCARDS)

SBS_PROCESS_TARGETS_ALL = sum(SBS_PROCESS_TARGETS.values(), [])
