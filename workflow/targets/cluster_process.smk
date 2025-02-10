from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


CLUSTER_PROCESS_FP = ROOT_FP / "cluster_process"

CLUSTER_PROCESS_OUTPUTS = {
    "generate_dataset": [
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "datasets"
        / get_filename({"dataset": "{dataset}"}, "clean_gene_data", "tsv"),
    ],
    "phate_leiden_clustering": [
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "plots"
        / get_filename({"dataset": "{dataset}"}, "pca_variance_plot", "png"),
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "plots"
        / get_filename({"dataset": "{dataset}"}, "phate_leiden_clustering", "pdf"),
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "tsvs"
        / get_filename({"dataset": "{dataset}"}, "phate_leiden_uniprot", "tsv"),
    ],
    "benchmark_clusters": [
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "tsvs"
        / get_filename({"dataset": "{dataset}"}, "cluster_gene_table", "tsv"),
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "tsvs"
        / get_filename({"dataset": "{dataset}"}, "global_metrics", "tsv"),
    ],
    "cluster_eval": [
        CLUSTER_PROCESS_FP / "tsvs" / "aggregated_cluster_metrics.tsv",
    ],
}

CLUSTER_PROCESS_OUTPUT_MAPPINGS = {
    "generate_dataset": None,
    "phate_leiden_clustering": None,
    "benchmark_clusters": None,
    "cluster_eval": None,
}


CHANNEL_COMBOS = [
    "_".join(combo) for combo in config["cluster_process"]["channel_combos"]
]
DATASETS = config["cluster_process"]["dataset_types"]
CLUSTER_PROCESS_WILDCARDS = {
    "channel_combo": CHANNEL_COMBOS,
    "dataset": DATASETS,
}

CLUSTER_PROCESS_OUTPUTS_MAPPED = map_outputs(
    CLUSTER_PROCESS_OUTPUTS, CLUSTER_PROCESS_OUTPUT_MAPPINGS
)

CLUSTER_PROCESS_TARGETS = outputs_to_targets(
    CLUSTER_PROCESS_OUTPUTS, CLUSTER_PROCESS_WILDCARDS
)

CLUSTER_PROCESS_TARGETS_ALL = sum(CLUSTER_PROCESS_TARGETS.values(), [])
