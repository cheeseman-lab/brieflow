from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


MERGE_FP = ROOT_FP / "merge"

MERGE_OUTPUTS = {
    # Stitching configuration outputs (only needed for well approach)
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
    
    # Well-based stitching outputs (only needed for well approach)
    "stitch_phenotype_image": [
        MERGE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitched_image", "npy"
        ),
    ],
    "stitch_phenotype_mask": [
        MERGE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitched_mask", "npy"
        ),
    ],
    "stitch_phenotype_positions": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_cell_positions", "parquet"
        ),
    ],
    "stitch_sbs_image": [
        MERGE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitched_image", "npy"
        ),
    ],
    "stitch_sbs_mask": [
        MERGE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitched_mask", "npy"
        ),
    ],
    "stitch_sbs_positions": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_cell_positions", "parquet"
        ),
    ],
    
    "stitch_phenotype_qc": [
        MERGE_FP / "qc_plots" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_tile_qc", "png"
        ),
    ],
    "stitch_sbs_qc": [
        MERGE_FP / "qc_plots" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_tile_qc", "png"
        ),
    ],
    
    # 3-Step Well Pipeline Outputs
    "well_alignment": [
        MERGE_FP / "well_alignment" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_scaled", "parquet"
        ),  # [0]
        MERGE_FP / "well_alignment" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_triangles", "parquet"
        ),  # [1]
        MERGE_FP / "well_alignment" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_triangles", "parquet"
        ),  # [2]
        MERGE_FP / "well_alignment" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "alignment", "parquet"
        ),  # [3]
        MERGE_FP / "tsvs" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "alignment_summary", "tsv"
        ),  # [4]
        MERGE_FP / "well_alignment" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_transformed", "parquet"
        ),  # [5]
    ],
    "well_cell_merge": [
        MERGE_FP / "well_cell_merge" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "raw_matches", "parquet"
        ),  # [0]
        MERGE_FP / "well_cell_merge" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merged_cells", "parquet"
        ),  # [1]
        MERGE_FP / "tsvs" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merge_summary", "tsv"
        ),  # [2]
    ],
    "well_merge_deduplicate": [
        MERGE_FP / "well_merge_deduplicate" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "deduplicated_cells", "parquet"
        ),  # [0]
        MERGE_FP / "tsvs" / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "dedup_summary", "tsv"
        ),  # [1]
    ],
    
    
    # Legacy approach outputs
    "fast_alignment": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "fast_alignment", "parquet"
        ),
    ],
    "merge_tile": [
        MERGE_FP
        / "parquets"
        / get_filename({"plate": "{plate}", "well": "{well}"}, "merge_tile", "parquet"),
    ],
    
    # Main merge output (always needed)
    "merge": [
        MERGE_FP
        / "parquets"
        / get_filename({"plate": "{plate}", "well": "{well}"}, "merge", "parquet"),
    ],
    
    # Keep existing outputs for backward compatibility
    "well_merge": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "well_merge", "parquet"
        ),
    ],
    
    # Downstream outputs (always needed)
    "format_merge": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merge_formatted", "parquet"
        ),
    ],
    "deduplicate_merge": [
        MERGE_FP
        / "eval"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "deduplication_stats", "tsv"
        ),
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merge_deduplicated", "parquet"
        ),
        MERGE_FP
        / "eval"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "final_sbs_matching_rates", "tsv"
        ),
        MERGE_FP
        / "eval"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"},
            "final_phenotype_matching_rates",
            "tsv",
        ),
    ],
    "final_merge": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "merge_final", "parquet"
        ),
    ],
    "eval_merge": [
        MERGE_FP
        / "eval"
        / get_filename({"plate": "{plate}"}, "cell_mapping_stats", "tsv"),
        MERGE_FP
        / "eval"
        / get_filename({"plate": "{plate}"}, "sbs_to_ph_matching_rates", "tsv"),
        MERGE_FP
        / "eval"
        / get_filename({"plate": "{plate}"}, "sbs_to_ph_matching_rates", "png"),
        MERGE_FP
        / "eval"
        / get_filename({"plate": "{plate}"}, "ph_to_sbs_matching_rates", "tsv"),
        MERGE_FP
        / "eval"
        / get_filename({"plate": "{plate}"}, "ph_to_sbs_matching_rates", "png"),
        MERGE_FP
        / "eval"
        / get_filename({"plate": "{plate}"}, "all_cells_by_channel_min", "png"),
        MERGE_FP
        / "eval"
        / get_filename({"plate": "{plate}"}, "cells_with_channel_min_0", "png"),
    ],
}


def get_merge_targets_by_approach():
    """Get targets based on the configured approach"""
    approach = config.get("merge", {}).get("approach", "tile")
    
    # Core targets that are always needed
    core_targets = ["format_merge", "final_merge", "eval_merge"]
    
    if approach == "well":
        approach_targets = [
            "estimate_stitch_phenotype", "estimate_stitch_sbs",
            "stitch_phenotype_image", "stitch_phenotype_mask", "stitch_phenotype_positions",
            "stitch_sbs_image", "stitch_sbs_mask", "stitch_sbs_positions", 
            "well_alignment", "well_cell_merge", "well_merge_deduplicate",
        ]
        # For well approach, we skip deduplicate_merge since deduplication is done in well_merge_deduplicate
    else:
        # Legacy approach targets
        approach_targets = [
            "fast_alignment", "merge_tile"
        ]
        # Legacy approach needs the additional deduplicate_merge step
        core_targets.insert(-2, "deduplicate_merge")  # Insert before final_merge and eval_merge
    
    # Always include the main merge target
    all_targets = approach_targets + ["merge"] + core_targets
    
    return all_targets


MERGE_OUTPUT_MAPPINGS = {
    "estimate_stitch_phenotype": temp,
    "estimate_stitch_sbs": temp,
    "stitch_phenotype_image": None,
    "stitch_phenotype_mask": None, 
    "stitch_phenotype_positions": temp,
    "stitch_sbs_image": None,
    "stitch_sbs_mask": None,
    "stitch_sbs_positions": None,
    "well_alignment": None,
    "well_cell_merge": None,
    "well_merge_deduplicate": temp,
    "fast_alignment": None,
    "merge_tile": None,
    "merge": None,
    "format_merge": None,
    "eval_merge": None,
    "deduplicate_merge": None,
    "final_merge": None,
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