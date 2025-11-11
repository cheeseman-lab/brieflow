from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


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
    "estimate_stitch_phenotype": [
        MERGE_FP
        / "stitch_configs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitch_config", "yml"
        ),
    ],
    "estimate_stitch_sbs": [
        MERGE_FP
        / "stitch_configs" 
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitch_config", "yml"
        ),
    ],
    "stitch_phenotype": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_cell_positions", "parquet"
        ),  # [0] - phenotype_cell_positions (always)
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_tile_qc", "png"
        ),  # [1] - phenotype_qc_plot (always)
        MERGE_FP / "images" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitched_image", "npy"
        ),  # [2] - phenotype_stitched_image (conditional - may be empty file)
        MERGE_FP / "images" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitched_mask", "npy"
        ),  # [3] - phenotype_stitched_mask (conditional - may be empty file)
    ],  
    "stitch_sbs": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_cell_positions", "parquet"
        ),  # [0] - sbs_cell_positions (always)
        MERGE_FP / "eval" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_tile_qc", "png"
        ),  # [1] - sbs_qc_plot (always)
        MERGE_FP / "images" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitched_image", "npy"
        ),  # [2] - sbs_stitched_image (conditional - may be empty file)
        MERGE_FP / "images" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitched_mask", "npy"
        ),  # [3] - sbs_stitched_mask (conditional - may be empty file)
    ],
    "stitch_alignment": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_scaled", "parquet"
        ),  # [0] - scaled_phenotype_positions
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_triangles", "parquet"
        ),  # [1] - phenotype_triangles
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_triangles", "parquet"
        ),  # [2] - sbs_triangles
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "alignment", "parquet"
        ),  # [3] - alignment_params
        MERGE_FP / "tsvs" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "alignment_summary", "tsv"
        ),  # [4] - alignment_summary
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_transformed", "parquet"
        ),  # [5] - transformed_phenotype_positions
    ],
    "stitch_merge": [
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "raw_matches", "parquet"
        ),  # [0] - raw_matches
        MERGE_FP / "parquets" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merged_cells", "parquet"
        ),  # [1] - merged_cells
        MERGE_FP / "tsvs" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merge_summary", "tsv"
        ),  # [2] - merge_summary
    ],
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
            {"plate": "{plate}"}, "cell_mapping_stats", "tsv"
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
    ],
    "summarize_merge": [
        MERGE_FP / "tsvs" / "aggregated" / get_filename(
            {"plate": "{plate}"}, "alignment_summaries", "tsv"
        ),  # [0] - alignment_summaries
        MERGE_FP / "tsvs" / "aggregated" / get_filename(
            {"plate": "{plate}"}, "cell_merge_summaries", "tsv"
        ),  # [1] - cell_merge_summaries
        MERGE_FP / "tsvs" / "aggregated" / get_filename(
            {"plate": "{plate}"}, "dedup_summaries", "tsv"
        ),  # [2] - dedup_summaries
        MERGE_FP / "tsvs" / "aggregated" / get_filename(
            {"plate": "{plate}"}, "sbs_matching_summaries", "tsv"
        ),  # [3] - sbs_matching_summaries
        MERGE_FP / "tsvs" / "aggregated" / get_filename(
            {"plate": "{plate}"}, "phenotype_matching_summaries", "tsv"
        ),  # [4] - phenotype_matching_summaries
    ],
}


def get_merge_targets_by_approach():
    """Get targets based on the configured approach"""
    approach = config.get("merge", {}).get("approach", "fast")
    
    # Core targets that are always needed
    core_targets = ["format_merge", "deduplicate_merge", "final_merge", "eval_merge"]
    
    if approach == "stitch":
        approach_targets = [
            "estimate_stitch_phenotype", "estimate_stitch_sbs",
            "stitch_phenotype", "stitch_sbs",
            "stitch_alignment", "stitch_merge"
        ]
        # Add aggregate summaries target for stitch approach
        core_targets.append("summarize_merge")
    else:
        # Fast approach targets (default)
        approach_targets = [
            "fast_alignment", "fast_merge"
        ]
    
    all_targets = approach_targets + core_targets
    
    return all_targets


MERGE_OUTPUT_MAPPINGS = {
    "fast_alignment": None,
    "fast_merge": None,
    "estimate_stitch_phenotype": temp,
    "estimate_stitch_sbs": temp,
    "stitch_phenotype": [temp, None, temp, temp],
    "stitch_sbs": [None, None, temp, temp],
    "stitch_alignment": [temp, temp, temp, temp, temp, None],
    "stitch_merge": [temp, None, temp],
    "format_merge": None,
    "deduplicate_merge": [None, temp, temp, temp],
    "final_merge": None,
    "eval_merge": None,
    "summarize_merge": None,
}

MERGE_OUTPUTS_MAPPED = map_outputs(MERGE_OUTPUTS, MERGE_OUTPUT_MAPPINGS)

# Get targets based on approach
MERGE_TARGETS_SELECTED = get_merge_targets_by_approach()

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
        