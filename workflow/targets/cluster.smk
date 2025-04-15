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
    # "phate_leiden_clustering": [
    #     CLUSTER_FP
    #     / "{channel_combo}"
    #     / "plots"
    #     / get_filename({"dataset": "{dataset}"}, "pca_variance_plot", "png"),
    #     CLUSTER_FP
    #     / "{channel_combo}"
    #     / "plots"
    #     / get_filename({"dataset": "{dataset}"}, "phate_leiden_clustering", "pdf"),
    #     CLUSTER_FP
    #     / "{channel_combo}"
    #     / "tsvs"
    #     / get_filename({"dataset": "{dataset}"}, "phate_leiden_uniprot", "tsv"),
    # ],
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
    "cutoff_aggregate": None,
    # "phate_leiden_clustering": None,
    # "benchmark_clusters": None,
    # "cluster_eval": None,
}


channel_combos = ["_".join(combo) for combo in config["cluster"]["channel_combos"]]
datasets = config["cluster"]["dataset_types"]
cluster_wildcard_combos = pd.DataFrame(
    product(channel_combos, datasets), columns=["channel_combo", "dataset"]
)

CLUSTER_OUTPUTS_MAPPED = map_outputs(CLUSTER_OUTPUTS, CLUSTER_OUTPUT_MAPPINGS)

CLUSTER_TARGETS_ALL = outputs_to_targets(
    CLUSTER_OUTPUTS, cluster_wildcard_combos, CLUSTER_OUTPUT_MAPPINGS
)
