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
        / get_filename({"plate": "{plate}"}, "reads_per_cell_histogram", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "gene_symbol_histogram", "png"),
        SBS_FP
        / "eval"
        / "mapping"
        / get_filename({"plate": "{plate}"}, "mapping_overview", "tsv"),
    ],
}

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
}

# TODO: test and implement segmentation paramsearch for updated brieflow setup
# if config["sbs"]["mode"] == "segment_sbs_paramsearch":
#     SBS_OUTPUTS.update(
#         {
#             "segment_sbs_paramsearch": [
#                 SBS_FP
#                 / "paramsearch"
#                 / "images"
#                 / get_filename(
#                     {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_nuclei",
#                     "tiff",
#                 ),
#                 SBS_FP
#                 / "paramsearch"
#                 / "images"
#                 / get_filename(
#                     {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_cells",
#                     "tiff",
#                 ),
#                 SBS_FP
#                 / "paramsearch"
#                 / "tsvs"
#                 / get_filename(
#                     {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
#                     f"paramsearch_nd{'{nuclei_diameter}'}_cd{'{cell_diameter}'}_ft{'{flow_threshold}'}_cp{'{cellprob_threshold}'}_segmentation_stats",
#                     "tsv",
#                 ),
#             ],
#             "summarize_segment_sbs_paramsearch": [
#                 SBS_FP / "paramsearch" / "summary" / "segmentation_summary.tsv",
#                 SBS_FP / "paramsearch" / "summary" / "segmentation_grouped.tsv",
#                 SBS_FP / "paramsearch" / "summary" / "segmentation_evaluation.txt",
#                 SBS_FP / "paramsearch" / "summary" / "segmentation_panel.png",
#             ],
#         }
#     )

#     SBS_OUTPUT_MAPPINGS.update(
#         {"segment_sbs_paramsearch": None, "summarize_segment_sbs_paramsearch": None}
#     )

#     SBS_WILDCARDS.update(
#         {
#             "nuclei_diameter": config["sbs"]["paramsearch"]["nuclei_diameter"],
#             "cell_diameter": config["sbs"]["paramsearch"]["cell_diameter"],
#             "flow_threshold": config["sbs"]["paramsearch"]["flow_threshold"],
#             "cellprob_threshold": config["sbs"]["paramsearch"]["cellprob_threshold"],
#         }
#     )

SBS_OUTPUTS_MAPPED = map_outputs(SBS_OUTPUTS, SBS_OUTPUT_MAPPINGS)

SBS_TARGETS_ALL = outputs_to_targets(
    SBS_OUTPUTS, sbs_wildcard_combos, SBS_OUTPUT_MAPPINGS
)
