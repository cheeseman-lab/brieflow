import pandas as pd
from itertools import product

from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


CLUSTER_FP = ROOT_FP / "cluster"

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
    "benchmark_clusters": [
        CLUSTER_FP
        / "{channel_combo}"
        / "{cell_class}"
        / "plots"
        / get_filename(
            {"leiden_resolution": "{leiden_resolution}"}, "cluster_benchmarks", "png"
        ),
    ],
}

CLUSTER_OUTPUT_MAPPINGS = {
    "clean_aggregate": None,
    "phate_leiden_clustering": None,
    "benchmark_clusters": None,
}


# # TODO: Use all combos
cluster_wildcard_combos = cluster_wildcard_combos[
    (cluster_wildcard_combos["cell_class"].isin(["mitotic"]))
    & (cluster_wildcard_combos["channel_combo"].isin(["DAPI_COXIV_CENPA_WGA"]))
    & (cluster_wildcard_combos["leiden_resolution"].isin([8]))
]

CLUSTER_OUTPUTS_MAPPED = map_outputs(CLUSTER_OUTPUTS, CLUSTER_OUTPUT_MAPPINGS)

CLUSTER_TARGETS_ALL = outputs_to_targets(
    CLUSTER_OUTPUTS, cluster_wildcard_combos, CLUSTER_OUTPUT_MAPPINGS
)
