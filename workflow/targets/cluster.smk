from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


CLUSTER_FP = ROOT_FP / "cluster"

CLUSTER_OUTPUTS = {
    "generate_dataset": [
        CLUSTER_FP
        / "{channel_combo}"
        / "datasets"
        / get_filename({"dataset": "{dataset}"}, "clean_gene_data", "tsv"),
    ],
    "phate_leiden_clustering": [
        CLUSTER_FP
        / "{channel_combo}"
        / "plots"
        / get_filename({"dataset": "{dataset}"}, "pca_variance_plot", "png"),
        CLUSTER_FP
        / "{channel_combo}"
        / "plots"
        / get_filename({"dataset": "{dataset}"}, "phate_leiden_clustering", "pdf"),
        CLUSTER_FP
        / "{channel_combo}"
        / "tsvs"
        / get_filename({"dataset": "{dataset}"}, "phate_leiden_uniprot", "tsv"),
    ],
    "benchmark_clusters": [
        CLUSTER_FP
        / "{channel_combo}"
        / "tsvs"
        / get_filename({"dataset": "{dataset}"}, "cluster_gene_table", "tsv"),
        CLUSTER_FP
        / "{channel_combo}"
        / "tsvs"
        / get_filename({"dataset": "{dataset}"}, "global_metrics", "tsv"),
    ],
    "cluster_eval": [
        CLUSTER_FP / "tsvs" / "aggregated_cluster_metrics.tsv",
    ],
}

CLUSTER_OUTPUT_MAPPINGS = {
    "generate_dataset": None,
    "phate_leiden_clustering": None,
    "benchmark_clusters": None,
    "cluster_eval": None,
}


CHANNEL_COMBOS = ["_".join(combo) for combo in config["cluster"]["channel_combos"]]
DATASETS = config["cluster"]["dataset_types"]
CLUSTER_WILDCARDS = {
    "channel_combo": CHANNEL_COMBOS,
    "dataset": DATASETS,
}

CLUSTER_OUTPUTS_MAPPED = map_outputs(CLUSTER_OUTPUTS, CLUSTER_OUTPUT_MAPPINGS)

CLUSTER_TARGETS = outputs_to_targets(
    CLUSTER_OUTPUTS, CLUSTER_WILDCARDS, CLUSTER_OUTPUT_MAPPINGS
)

CLUSTER_TARGETS_ALL = sum(CLUSTER_TARGETS.values(), [])
