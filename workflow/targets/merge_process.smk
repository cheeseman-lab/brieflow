from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


MERGE_PROCESS_FP = ROOT_FP / "merge_process"

MERGE_PROCESS_OUTPUTS = {
    "fast_alignment": [
        MERGE_PROCESS_FP / "hdfs" / get_filename({}, "fast_alignment", "hdf5"),
    ],
    "merge": [
        MERGE_PROCESS_FP / "hdfs" / get_filename({}, "merge", "hdf5"),
    ],
    "format_merge": [
        MERGE_PROCESS_FP / "hdfs" / get_filename({}, "merge_formatted", "hdf5"),
    ],
}

MERGE_PROCESS_OUTPUT_MAPPINGS = {
    "fast_alignment": None,
    "merge": None,
    "format_merge": None,
}

MERGE_PROCESS_WILDCARDS = {}

MERGE_PROCESS_OUTPUTS_MAPPED = map_outputs(
    MERGE_PROCESS_OUTPUTS, MERGE_PROCESS_OUTPUT_MAPPINGS
)

MERGE_PROCESS_TARGETS = outputs_to_targets(
    MERGE_PROCESS_OUTPUTS, MERGE_PROCESS_WILDCARDS
)

MERGE_PROCESS_TARGETS_ALL = sum(MERGE_PROCESS_TARGETS.values(), [])
