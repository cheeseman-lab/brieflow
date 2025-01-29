from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


AGGREGATE_PROCESS_FP = ROOT_FP / "aggregate_process"

AGGREGATE_PROCESS_OUTPUTS = {
    "clean_and_transform": [
        AGGREGATE_PROCESS_FP / "hdfs" / "transformed_data.hdf5",
    ],
    "standardize_features": [
        AGGREGATE_PROCESS_FP / "hdfs" / "standardized_data.hdf5",
    ],
    "split_phases": [
        AGGREGATE_PROCESS_FP / "hdfs" / "mitotic_data.hdf5",
        AGGREGATE_PROCESS_FP / "hdfs" / "interphase_data.hdf5",
    ],
}

AGGREGATE_PROCESS_OUTPUT_MAPPINGS = {
    "clean_and_transform": None,
}

AGGREGATE_PROCESS_WILDCARDS = {}

AGGREGATE_PROCESS_OUTPUTS_MAPPED = map_outputs(
    AGGREGATE_PROCESS_OUTPUTS, AGGREGATE_PROCESS_OUTPUT_MAPPINGS
)

AGGREGATE_PROCESS_TARGETS = outputs_to_targets(
    AGGREGATE_PROCESS_OUTPUTS, AGGREGATE_PROCESS_WILDCARDS
)

AGGREGATE_PROCESS_TARGETS_ALL = sum(AGGREGATE_PROCESS_TARGETS.values(), [])
