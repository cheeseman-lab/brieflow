from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


MERGE_FP = ROOT_FP / "merge"

MERGE_OUTPUTS = {
    # Stitching configuration outputs (only needed for enhanced approach)
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
    
    # Enhanced stitching outputs (only needed for enhanced approach)
    "stitch_phenotype_image": [
        MERGE_FP
        / "stitched_images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitched_image", "npy"
        ),
    ],
    "stitch_phenotype_mask": [
        MERGE_FP
        / "stitched_masks"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitched_mask", "npy"
        ),
    ],
    "stitch_phenotype_positions": [
        MERGE_FP
        / "cell_positions"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_cell_positions", "parquet"
        ),
    ],
    "stitch_sbs_image": [
        MERGE_FP
        / "stitched_images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitched_image", "npy"
        ),
    ],
    "stitch_sbs_mask": [
        MERGE_FP
        / "stitched_masks"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitched_mask", "npy"
        ),
    ],
    "stitch_sbs_positions": [
        MERGE_FP
        / "cell_positions"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_cell_positions", "parquet"
        ),
    ],
    
    # Optional overlay outputs (only needed for enhanced approach)
    "stitch_phenotype_overlay": [
        MERGE_FP
        / "overlays"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_overlay", "png"
        ),
    ],
    "stitch_sbs_overlay": [
        MERGE_FP
        / "overlays"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_overlay", "png"
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
    
    # Merge approach outputs
    "enhanced_well_merge": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "enhanced_well_merge", "parquet"
        ),
    ],
    
    # Legacy approach outputs
    "fast_alignment": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "fast_alignment", "parquet"
        ),
    ],
    "merge_legacy": [
        MERGE_FP
        / "parquets"
        / get_filename({"plate": "{plate}", "well": "{well}"}, "merge_legacy", "parquet"),
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
    approach = config.get("merge", {}).get("approach", "legacy")
    
    # Core targets that are always needed
    core_targets = [
        "format_merge", "deduplicate_merge", "final_merge", "eval_merge"
    ]
    
    if approach == "enhanced":
        # Enhanced approach targets
        approach_targets = [
            "estimate_stitch_phenotype", "estimate_stitch_sbs",
            "stitch_phenotype_image", "stitch_phenotype_mask", "stitch_phenotype_positions",
            "stitch_sbs_image", "stitch_sbs_mask", "stitch_sbs_positions", 
            "stitch_phenotype_overlay", "stitch_sbs_overlay",
            "enhanced_well_merge", "well_merge"
        ]
    else:
        # Legacy approach targets
        approach_targets = [
            "fast_alignment", "merge_legacy"
        ]
    
    # Always include the main merge target
    all_targets = approach_targets + ["merge"] + core_targets
    
    return all_targets


MERGE_OUTPUT_MAPPINGS = {
    "estimate_stitch_phenotype": None,
    "estimate_stitch_sbs": None,
    
    # Enhanced stitching mappings
    "stitch_phenotype_image": None,
    "stitch_phenotype_mask": None, 
    "stitch_phenotype_positions": None,
    "stitch_sbs_image": None,
    "stitch_sbs_mask": None,
    "stitch_sbs_positions": None,
    "stitch_phenotype_overlay": None,
    "stitch_sbs_overlay": None,
    
    # Merge approach mappings
    "enhanced_well_merge": None,
    "well_merge": None,
    "fast_alignment": None,
    "merge_legacy": None,
    "merge": None,
    
    # Existing mappings
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