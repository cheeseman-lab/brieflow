from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


CLUSTER_PROCESS_FP = ROOT_FP / "cluster_process"

CLUSTER_PROCESS_OUTPUTS = {
    "generate_cluster_datasets": [
        CLUSTER_PROCESS_FP
        / "{channel_combo}"
        / "datasets"
        / get_filename({"dataset": "{dataset}"}, "clean_gene_data", "tsv"),
    ],
}

CLUSTER_PROCESS_OUTPUT_MAPPINGS = {
    "generate_cluster_datasets": None,
}

CHANNEL_COMBOS = [
    # ["dapi", "coxiv", "cenpa", "wga"],
    # ["dapi", "coxiv"],
    ["dapi", "cenpa"],
    # ["dapi", "wga"],
]
CHANNEL_COMBO_STRINGS = ["_".join(combo) for combo in CHANNEL_COMBOS]
DATASETS = ["mitotic"]  # , "interphase", "all"]

CLUSTER_PROCESS_WILDCARDS = {
    "channel_combo": CHANNEL_COMBO_STRINGS,
    "dataset": DATASETS,
}

CLUSTER_PROCESS_OUTPUTS_MAPPED = map_outputs(
    CLUSTER_PROCESS_OUTPUTS, CLUSTER_PROCESS_OUTPUT_MAPPINGS
)

CLUSTER_PROCESS_TARGETS = outputs_to_targets(
    CLUSTER_PROCESS_OUTPUTS, CLUSTER_PROCESS_WILDCARDS
)
print(CLUSTER_PROCESS_TARGETS)

CLUSTER_PROCESS_TARGETS_ALL = sum(CLUSTER_PROCESS_TARGETS.values(), [])
