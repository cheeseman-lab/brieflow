from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


AGGREGATE_FP = ROOT_FP / "aggregate"

# Define standard (non-montage) aggreagte outputs
AGGREGATE_OUTPUTS = {
    "create_datasets": [
        AGGREGATE_FP
        / "parquets"
        / get_filename({"dataset": "{dataset}"}, "merge_data", "parquet"),
    ],
}

AGGREGATE_OUTPUT_MAPPINGS = {
    "create_datasets": None,
}

AGGREGATE_OUTPUTS_MAPPED = map_outputs(AGGREGATE_OUTPUTS, AGGREGATE_OUTPUT_MAPPINGS)

aggregate_wildcard_combos = pd.DataFrame({"dataset": ["all", "mitotic", "interphase"]})
AGGREGATE_TARGETS = outputs_to_targets(
    AGGREGATE_OUTPUTS, aggregate_wildcard_combos, AGGREGATE_OUTPUT_MAPPINGS
)
print(AGGREGATE_TARGETS)

print(AGGREGATE_OUTPUTS_MAPPED["create_datasets"])

AGGREGATE_TARGETS_ALL = AGGREGATE_TARGETS
