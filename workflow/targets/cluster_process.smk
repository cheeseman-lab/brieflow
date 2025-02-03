from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


CLUSTER_PROCESS_FP = ROOT_FP / "cluster_process"

CLUSTER_PROCESS_OUTPUTS = {
    "calculate_mitotic_percentage": [CLUSTER_PROCESS_FP / "tsvs" / "test.tsv"],
}

CLUSTER_PROCESS_OUTPUT_MAPPINGS = {
    "calculate_mitotic_percentage": None,
}

CLUSTER_PROCESS_WILDCARDS = {}

CLUSTER_PROCESS_OUTPUTS_MAPPED = map_outputs(
    CLUSTER_PROCESS_OUTPUTS, CLUSTER_PROCESS_OUTPUT_MAPPINGS
)

CLUSTER_PROCESS_TARGETS = outputs_to_targets(
    CLUSTER_PROCESS_OUTPUTS, CLUSTER_PROCESS_WILDCARDS
)

CLUSTER_PROCESS_TARGETS_ALL = sum(CLUSTER_PROCESS_TARGETS.values(), [])
