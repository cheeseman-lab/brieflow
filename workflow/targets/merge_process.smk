from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


MERGE_PROCESS_FP = ROOT_FP / "merge_process"

MERGE_PROCESS_OUTPUTS = {
    "fast_alignment": [
        MERGE_PROCESS_FP / "hdfs" / "fast_alignment.hdf5",
    ],
    "merge": [
        MERGE_PROCESS_FP / "hdfs" / "merge.hdf5",
    ],
    "format_merge": [
        MERGE_PROCESS_FP / "hdfs" / "merge_formatted.hdf5",
    ],
    "eval_merge": [
        MERGE_PROCESS_FP / "eval" / "cell_mapping_stats.tsv",
        MERGE_PROCESS_FP / "eval" / "sbs_to_ph_matching_rates.tsv",
        MERGE_PROCESS_FP / "eval" / "sbs_to_ph_matching_rates.png",
        MERGE_PROCESS_FP / "eval" / "ph_to_sbs_matching_rates.tsv",
        MERGE_PROCESS_FP / "eval" / "ph_to_sbs_matching_rates.png",
        MERGE_PROCESS_FP / "eval" / "all_cells_by_channel_min.png",
        MERGE_PROCESS_FP / "eval" / "cells_with_channel_min_0.png",
    ],
    "clean_merge": [
        MERGE_PROCESS_FP / "eval" / "channel_min_histogram.png",
        MERGE_PROCESS_FP / "hdfs" / "merge_cleaned.hdf5",
    ],
    "deduplicate_merge": [
        MERGE_PROCESS_FP / "eval" / "deduplication_stats.tsv",
        MERGE_PROCESS_FP / "hdfs" / "merge_deduplicated.hdf5",
        MERGE_PROCESS_FP / "eval" / "final_sbs_matching_rates.tsv",
        MERGE_PROCESS_FP / "eval" / "final_phenotype_matching_rates.tsv",
    ],
    "final_merge": [
        MERGE_PROCESS_FP / "hdfs" / "merge_final.hdf5",
    ],
}

MERGE_PROCESS_OUTPUT_MAPPINGS = {
    "fast_alignment": None,
    "merge": None,
    "format_merge": None,
    "eval_merge": None,
    "clean_merge": None,
    "deduplicate_merge": None,
    "final_merge": None,
}

MERGE_PROCESS_WILDCARDS = {}

MERGE_PROCESS_OUTPUTS_MAPPED = map_outputs(
    MERGE_PROCESS_OUTPUTS, MERGE_PROCESS_OUTPUT_MAPPINGS
)

MERGE_PROCESS_TARGETS = outputs_to_targets(
    MERGE_PROCESS_OUTPUTS, MERGE_PROCESS_WILDCARDS
)

MERGE_PROCESS_TARGETS_ALL = sum(MERGE_PROCESS_TARGETS.values(), [])
