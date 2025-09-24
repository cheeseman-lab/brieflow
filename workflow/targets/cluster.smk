import pandas as pd
from itertools import product

from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


CLUSTER_FP = ROOT_FP / "cluster_ps_test"

CLUSTER_OUTPUTS = {
    "clean_aggregate": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / get_filename({}, "aggregate_cleaned", "tsv"),
    ],
    "phate_leiden_clustering": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {},
            "phate_leiden_clustering",
            "tsv",
        ),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({}, "cluster_sizes", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({}, "clusters", "png"),
    ],
    "benchmark_clusters": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "integrated_results", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "integrated_results", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "combined_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "combined_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "global_metrics", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Shuffled"}, "global_metrics", "json"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "pie_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {"cluster_benchmark": "Shuffled"}, "enrichment_pie_chart", "png"
        ),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename({"cluster_benchmark": "Real"}, "enrichment_bar_chart", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "{leiden_resolution}"
        / get_filename(
            {"cluster_benchmark": "Shuffled"}, "enrichment_bar_chart", "png"
        ),
    ],
}

CLUSTER_OUTPUT_MAPPINGS = {
    "clean_aggregate": None,
    "phate_leiden_clustering": None,
    "benchmark_clusters": None,
}


# TODO: Use all combos
# cluster_wildcard_combos = cluster_wildcard_combos[
#     (cluster_wildcard_combos["cell_class"].isin(["Interphase"]))
#     & (cluster_wildcard_combos["channel_combo"].isin(["DAPI_COXIV_CENPA_WGA"]))
#     & (cluster_wildcard_combos["leiden_resolution"].isin([13]))
# ]

CLUSTER_OUTPUTS_MAPPED = map_outputs(CLUSTER_OUTPUTS, CLUSTER_OUTPUT_MAPPINGS)

CLUSTER_TARGETS_ALL = outputs_to_targets(
    CLUSTER_OUTPUTS, cluster_wildcard_combos, CLUSTER_OUTPUT_MAPPINGS
)
