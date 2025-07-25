from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


MERGE_FP = ROOT_FP / "merge"

MERGE_OUTPUTS = {
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
        MERGE_FP
        / "stitched"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitched", "npy"
        ),
    ],
    "stitch_sbs": [
        MERGE_FP
        / "stitched"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitched", "npy"
        ),
    ],
    "well_merge": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "well_merge", "parquet"
        ),
    ],
    "fast_alignment": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "fast_alignment", "parquet"
        ),
    ],
    "merge": [
        MERGE_FP
        / "parquets"
        / get_filename({"plate": "{plate}", "well": "{well}"}, "merge", "parquet"),
    ],
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
        # Add these new stitching outputs:
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
        MERGE_FP
        / "stitched"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_stitched", "npy"
        ),
    ],
    "stitch_sbs": [
        MERGE_FP
        / "stitched"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "sbs_stitched", "npy"
        ),
    ],
    "well_merge": [
        MERGE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "well_merge", "parquet"
        ),
    ],
}


MERGE_OUTPUT_MAPPINGS = {
    "estimate_stitch_phenotype": None,
    "estimate_stitch_sbs": None, 
    "stitch_phenotype": None,
    "stitch_sbs": None,
    "well_merge": None,
    "fast_alignment": None,
    "merge": None,
    "format_merge": None,
    "eval_merge": None,
    "deduplicate_merge": None,
    "final_merge": None,
}

MERGE_OUTPUTS_MAPPED = map_outputs(MERGE_OUTPUTS, MERGE_OUTPUT_MAPPINGS)

MERGE_TARGETS_ALL = outputs_to_targets(
    MERGE_OUTPUTS, merge_wildcard_combos, MERGE_OUTPUT_MAPPINGS
)
