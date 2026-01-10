from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


PHENOTYPE_FP = ROOT_FP / "phenotype"

# determine feature eval outputs based on channel names
channel_names = config["phenotype"]["channel_names"]
eval_features = [f"cell_{channel}_min" for channel in channel_names]

PHENOTYPE_OUTPUTS = {
    "apply_ic_field_phenotype": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "illumination_corrected",
            "tiff",
        ),
    ],
    "align_phenotype": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "aligned", "tiff"
        ),
    ],
    "segment_phenotype": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "nuclei", "tiff"
        ),
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}, "cells", "tiff"
        ),
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "segmentation_stats",
            "tsv",
        ),
    ],
    "identify_cytoplasm": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "identified_cytoplasms",
            "tiff",
        ),
    ],
    "extract_phenotype_info": [
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "phenotype_info",
            "tsv",
        ),
    ],
    "combine_phenotype_info": [
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_info", "parquet"
        ),
    ],
    "identify_second_objs": [
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "identified_second_objs",
            "tiff",
        ),
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "cell_second_obj_table",
            "tsv",
        ),
        PHENOTYPE_FP
        / "images"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "updated_cytoplasms",
            "tiff",
        ),
    ],
    "extract_phenotype_cp": [
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "phenotype_cp",
            "tsv",
        ),
    ],
    "extract_phenotype_second_objs": [
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "phenotype_second_objs",
            "tsv",
        ),
    ],
    "merge_phenotype_second_objs": [
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_second_objs", "parquet"
        ),
    ],
    "merge_second_objs_phenotype_cp": [
        PHENOTYPE_FP
        / "tsvs"
        / get_filename(
            {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
            "phenotype_with_second_objs",
            "tsv",
        ),
    ],
    "merge_phenotype_cp": [
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_cp", "parquet"
        ),
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_cp_min", "parquet"
        ),
    ],
    "eval_segmentation_phenotype": [
        PHENOTYPE_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "segmentation_overview", "tsv"),
        PHENOTYPE_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "cell_density_heatmap", "tsv"),
        PHENOTYPE_FP
        / "eval"
        / "segmentation"
        / get_filename({"plate": "{plate}"}, "cell_density_heatmap", "png"),
    ],
    # create heatmap tsv and png for each evaluated feature
    "eval_features": [
        PHENOTYPE_FP
        / "eval"
        / "features"
        / get_filename({"plate": "{plate}"}, f"{feature}_heatmap", "tsv")
        for feature in eval_features
    ]
    + [
        PHENOTYPE_FP
        / "eval"
        / "features"
        / get_filename({"plate": "{plate}"}, f"{feature}_heatmap", "png")
        for feature in eval_features
    ],
}

PHENOTYPE_OUTPUT_MAPPINGS = {
    "apply_ic_field_phenotype": temp,
    "align_phenotype": None,
    "segment_phenotype": None,
    "identify_cytoplasm": temp,
    "extract_phenotype_info": temp,
    "combine_phenotype_info": None,
    "identify_second_objs": None,
    "extract_phenotype_cp": None,
    "extract_phenotype_second_objs": None,
    "merge_phenotype_second_objs": None,
    "merge_second_objs_phenotype_cp": None,
    "merge_phenotype_cp": None,
    "eval_segmentation_phenotype": None,
    "eval_features": None,
}

# Determine which outputs to include based on config
PHENOTYPE_SECOND_OBJ_DETECTION = config["phenotype"].get("second_obj_detection", True)

if not PHENOTYPE_SECOND_OBJ_DETECTION:
    # Filter out secondary object rules when disabled
    PHENOTYPE_OUTPUTS_FILTERED = {
        k: v for k, v in PHENOTYPE_OUTPUTS.items()
        if k not in [
            "identify_second_objs",
            "extract_phenotype_second_objs",
            "merge_phenotype_second_objs",
            "merge_second_objs_phenotype_cp",
        ]
    }

    PHENOTYPE_OUTPUT_MAPPINGS_FILTERED = {
        k: v for k, v in PHENOTYPE_OUTPUT_MAPPINGS.items()
        if k not in [
            "identify_second_objs",
            "extract_phenotype_second_objs",
            "merge_phenotype_second_objs",
            "merge_second_objs_phenotype_cp",
        ]
    }
else:
    PHENOTYPE_OUTPUTS_FILTERED = PHENOTYPE_OUTPUTS
    PHENOTYPE_OUTPUT_MAPPINGS_FILTERED = PHENOTYPE_OUTPUT_MAPPINGS

PHENOTYPE_OUTPUTS_MAPPED = map_outputs(PHENOTYPE_OUTPUTS_FILTERED, PHENOTYPE_OUTPUT_MAPPINGS_FILTERED)

PHENOTYPE_TARGETS_ALL = outputs_to_targets(
    PHENOTYPE_OUTPUTS_FILTERED, phenotype_wildcard_combos, PHENOTYPE_OUTPUT_MAPPINGS_FILTERED
)
