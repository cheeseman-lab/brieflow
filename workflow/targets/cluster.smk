import pandas as pd
from itertools import product

from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


# CLUSTER_FP = ROOT_FP / "cluster"
CLUSTER_FP = Path(
    "/lab/barcheese01/rkern/aggregate_overhaul/brieflow-analysis/analysis/analysis_root/cluster"
)

CLUSTER_OUTPUTS = {
    "clean_aggregate": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "tsvs"
        / get_filename({}, "aggregate_cleaned", "tsv"),
    ],
    "phate_leiden_clustering": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "tsvs"
        / get_filename(
            {"leiden_resolution": "{leiden_resolution}"},
            "phate_leiden_clustering",
            "tsv",
        ),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "plots"
        / get_filename(
            {"leiden_resolution": "{leiden_resolution}"}, "cluster_sizes", "png"
        ),
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "plots"
        / get_filename({"leiden_resolution": "{leiden_resolution}"}, "clusters", "png"),
    ],
    # "benchmark_clusters": [
    #     CLUSTER_FP
    #     / "{channel_combo}"
    #     / "tsvs"
    #     / get_filename({"dataset": "{dataset}"}, "cluster_gene_table", "tsv"),
    #     CLUSTER_FP
    #     / "{channel_combo}"
    #     / "tsvs"
    #     / get_filename({"dataset": "{dataset}"}, "global_metrics", "tsv"),
    # ],
    # "cluster_eval": [
    #     CLUSTER_FP / "tsvs" / "aggregated_cluster_metrics.tsv",
    # ],
}

CLUSTER_OUTPUT_MAPPINGS = {
    "clean_aggregate": None,
    "phate_leiden_clustering": None,
    "benchmark_clusters": None,
    "cluster_eval": None,
}


# TODO: Use all combos
cluster_wildcard_combos = cluster_wildcard_combos[
    (cluster_wildcard_combos["cell_class"].isin(["all"]))
    & (cluster_wildcard_combos["channel_combo"].isin(["DAPI_COXIV_CENPA_WGA"]))
    & (cluster_wildcard_combos["leiden_resolution"].isin([10]))
]

CLUSTER_OUTPUTS_MAPPED = map_outputs(CLUSTER_OUTPUTS, CLUSTER_OUTPUT_MAPPINGS)

CLUSTER_TARGETS_ALL = outputs_to_targets(
    CLUSTER_OUTPUTS, cluster_wildcard_combos, CLUSTER_OUTPUT_MAPPINGS
)
