from lib.shared.file_utils import get_filename
from lib.shared.target_utils import get_merge_targets_by_approach, map_outputs, outputs_to_targets


MERGE_FP = ROOT_FP / "merge"

MERGE_OUTPUTS = {
    "fast_alignment": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "fast_alignment", "parquet"
        ),
    ],
    "fast_merge": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "fast_merge", "parquet"
        ),
    ],
    "positions_merge": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "positions_merge", "parquet"
        ),
    ]
    + (
        [
            MERGE_FP / "eval" / get_filename(
                {"plate": "{plate}", "well": "{well}"}, name, ext
            )
            for name, ext in (
                ("positions_merge_qc", "tsv"),
                ("positions_image_qc", "tsv"),
                ("positions_tile_overlaps", "png"),
                ("positions_phenotype_in_sbs", "png"),
                ("positions_mosaic", "png"),
            )
        ]
        if config.get("merge", {}).get("positions_image_qc", False)
        else []
    ),
    "format_merge": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merge_formatted", "parquet"
        ),
    ],
    "deduplicate_merge": [
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "deduplication_stats", "tsv"
        ),  # [0] - deduplication_stats
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merge_deduplicated", "parquet"
        ),  # [1] - deduplicated_data
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "final_sbs_matching_rates", "tsv"
        ),  # [2] - final_sbs_matching_rates
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "final_phenotype_matching_rates", "tsv"
        ),  # [3] - final_phenotype_matching_rates
    ],
    "final_merge": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merge_final", "parquet"
        ),
    ],
    "eval_merge": [
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}"}, "merge_summary", "tsv"
        ),  # [0]
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}"}, "sbs_to_ph_matching_rates", "tsv"
        ),  # [1]
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}"}, "sbs_to_ph_matching_rates", "png"
        ),  # [2]
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}"}, "ph_to_sbs_matching_rates", "tsv"
        ),  # [3]
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}"}, "ph_to_sbs_matching_rates", "png"
        ),  # [4]
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}"}, "all_cells_by_channel_min", "png"
        ),  # [5]
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}"}, "cells_with_channel_min_0", "png"
        ),  # [6]
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}"}, "dedup_summaries", "tsv"
        ),  # [7]
    ],
}


MERGE_OUTPUT_MAPPINGS = {
    "fast_alignment": None,
    "fast_merge": None,
    "positions_merge": None,
    "format_merge": None,
    "deduplicate_merge": [temp, None, temp, temp],
    "final_merge": None,
    "eval_merge": None,
}

MERGE_OUTPUTS_MAPPED = map_outputs(MERGE_OUTPUTS, MERGE_OUTPUT_MAPPINGS)

# Get targets based on approach
MERGE_TARGETS_SELECTED = get_merge_targets_by_approach(config)

MERGE_TARGETS_ALL = []
for target in MERGE_TARGETS_SELECTED:
    if target in MERGE_OUTPUTS:
        MERGE_TARGETS_ALL.extend(
            outputs_to_targets(
                {target: MERGE_OUTPUTS[target]}, 
                merge_wildcard_combos, 
                {target: MERGE_OUTPUT_MAPPINGS[target]}
            )
        )
        