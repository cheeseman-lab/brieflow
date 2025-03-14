from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


AGGREGATE_FP = ROOT_FP / "aggregate"

# Define standard (non-montage) aggreagte outputs
AGGREGATE_OUTPUTS = {
    "split_classes": [
        AGGREGATE_FP
        / "parquets"
        / get_filename({"cell_class": "{cell_class}"}, "merge_data", "parquet"),
    ],
}

AGGREGATE_OUTPUT_MAPPINGS = {
    "split_classes": None,
}

AGGREGATE_OUTPUTS_MAPPED = map_outputs(AGGREGATE_OUTPUTS, AGGREGATE_OUTPUT_MAPPINGS)

# aggregate_wildcard_combos = pd.DataFrame({"cell_class": config["aggregate"]["cell_classes"]})
aggregate_wildcard_combos = pd.DataFrame({"cell_class": ["all"]})
AGGREGATE_TARGETS = outputs_to_targets(
    AGGREGATE_OUTPUTS, aggregate_wildcard_combos, AGGREGATE_OUTPUT_MAPPINGS
)
print(AGGREGATE_TARGETS)

print(AGGREGATE_OUTPUTS_MAPPED["split_classes"])

AGGREGATE_TARGETS_ALL = AGGREGATE_TARGETS
