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
    "merge_phenotype_info": [
        PHENOTYPE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "phenotype_info", "parquet"
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
    "apply_ic_field_phenotype": None,
    "align_phenotype": None,
    "segment_phenotype": None,
    "identify_cytoplasm": None,
    "extract_phenotype_info": None,
    "merge_phenotype_info": None,
    "extract_phenotype_cp": None,
    "merge_phenotype_cp": None,
    "eval_segmentation_phenotype": None,
    "eval_features": None,
}

# TODO: test and implement segmentation paramsearch for updated brieflow setup
# if config["phenotype"]["mode"] == "segment_phenotype_paramsearch":
#     PHENOTYPE_OUTPUTS.update(
#         {
#             "segment_phenotype_paramsearch": [
#                 PHENOTYPE_FP
#                 / "paramsearch"
#                 / "images"
#                 / get_filename(
#                     {"well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_nuclei",
#                     "tiff",
#                 ),
#                 PHENOTYPE_FP
#                 / "paramsearch"
#                 / "images"
#                 / get_filename(
#                     {"well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_cells",
#                     "tiff",
#                 ),
#                 PHENOTYPE_FP
#                 / "paramsearch"
#                 / "tsvs"
#                 / get_filename(
#                     {"well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_segmentation_stats",
#                     "tsv",
#                 ),
#             ],
#             "summarize_segment_phenotype_paramsearch": [
#                 PHENOTYPE_FP / "paramsearch" / "summary" / "segmentation_summary.tsv",
#                 PHENOTYPE_FP / "paramsearch" / "summary" / "segmentation_grouped.tsv",
#                 PHENOTYPE_FP
#                 / "paramsearch"
#                 / "summary"
#                 / "segmentation_evaluation.txt",
#                 PHENOTYPE_FP / "paramsearch" / "summary" / "segmentation_panel.png",
#             ],
#         }
#     )

#     PHENOTYPE_OUTPUT_MAPPINGS.update(
#         {
#             "segment_phenotype_paramsearch": None,
#             "summarize_segment_phenotype_paramsearch": None,
#         }
#     )

#     PHENOTYPE_WILDCARDS.update(
#         {
#             "nuclei_diameter": config["phenotype"]["paramsearch"]["nuclei_diameter"],
#             "cell_diameter": config["phenotype"]["paramsearch"]["cell_diameter"],
#             "flow_threshold": config["phenotype"]["paramsearch"]["flow_threshold"],
#             "cellprob_threshold": config["phenotype"]["paramsearch"][
#                 "cellprob_threshold"
#             ],
#         }
#     )


PHENOTYPE_OUTPUTS_MAPPED = map_outputs(PHENOTYPE_OUTPUTS, PHENOTYPE_OUTPUT_MAPPINGS)

PHENOTYPE_TARGETS_ALL = (
    # Targets that need tiles
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["apply_ic_field_phenotype"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS,
        extra_keys=PHENOTYPE_TILES
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["align_phenotype"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS,
        extra_keys=PHENOTYPE_TILES
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["segment_phenotype"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS,
        extra_keys=PHENOTYPE_TILES
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["identify_cytoplasm"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS,
        extra_keys=PHENOTYPE_TILES
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["extract_phenotype_info"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS,
        extra_keys=PHENOTYPE_TILES
    ) +
    # Targets that don't need tiles
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["merge_phenotype_info"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["extract_phenotype_cp"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS,
        extra_keys=PHENOTYPE_TILES
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["merge_phenotype_cp"],
        valid_combinations=PHENOTYPE_VALID_COMBINATIONS
    ) +
    # Evaluation targets (plate level only)
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["eval_segmentation_phenotype"],
        valid_combinations=[{"plate": combo["plate"]} for combo in PHENOTYPE_VALID_COMBINATIONS]
    ) +
    outputs_to_targets_with_combinations(
        output_templates=PHENOTYPE_OUTPUTS["eval_features"],
        valid_combinations=[{"plate": combo["plate"]} for combo in PHENOTYPE_VALID_COMBINATIONS]
    )
)