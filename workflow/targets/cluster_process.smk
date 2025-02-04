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
    "analyze_clusters": [
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "tsvs"
        / get_filename({"dataset": "{dataset}"}, "cluster_gene_table", "tsv"),
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "tsvs"
        / get_filename({"dataset": "{dataset}"}, "global_metrics", "tsv"),
    ],
}

CLUSTER_PROCESS_OUTPUT_MAPPINGS = {
    "generate_dataset": None,
    "phate_leiden_clustering": None,
}

CHANNEL_COMBOS = [
    # ["dapi", "coxiv", "cenpa", "wga"],
    # ["dapi", "coxiv"],
    ["dapi", "cenpa"],
    # ["dapi", "wga"],
]
CHANNEL_COMBOS = ["_".join(combo) for combo in CHANNEL_COMBOS]
DATASETS = ["mitotic"]  # , "interphase", "all"]

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
