from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


# AGGREGATE_FP = ROOT_FP / "aggregate"
AGGREGATE_FP = Path(
    "/lab/barcheese01/rkern/aggregate_overhaul/brieflow-analysis/analysis/analysis_root/aggregate"
)

# Define standard (non-montage) aggreagte outputs
AGGREGATE_OUTPUTS = {
    "split_datasets": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "cell_class": "{cell_class}",
                "channel_combo": "{channel_combo}",
            },
            "merge_data",
            "parquet",
        ),
    ],
    "filter": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {
                "plate": "{plate}",
                "well": "{well}",
                "cell_class": "{cell_class}",
                "channel_combo": "{channel_combo}",
            },
            "filtered",
            "parquet",
        ),
    ],
    "align": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "aligned",
            "parquet",
        ),
    ],
    "aggregate": [
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "aggregated",
            "tsv",
        ),
    ],
    "eval_aggregate": [
        AGGREGATE_FP
        / "eval"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "na_stats",
            "tsv",
        ),
        AGGREGATE_FP
        / "eval"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "na_stats",
            "png",
        ),
        AGGREGATE_FP
        / "eval"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "feature_violins",
            "png",
        ),
    ],
}

AGGREGATE_OUTPUT_MAPPINGS = {
    "split_datasets": None,
    "filter": None,
    "align": None,
    "aggregate": None,
    "eval_aggregate": None,
}

AGGREGATE_OUTPUTS_MAPPED = map_outputs(AGGREGATE_OUTPUTS, AGGREGATE_OUTPUT_MAPPINGS)

# TODO: Use all combos
aggregate_wildcard_combos = aggregate_wildcard_combos[
    (aggregate_wildcard_combos["plate"].isin([1, 2]))
    & (aggregate_wildcard_combos["well"].isin(["A1", "A2"]))
    & (aggregate_wildcard_combos["cell_class"].isin(["interphase", "mitotic", "all"]))
    & (
        aggregate_wildcard_combos["channel_combo"].isin(
            ["DAPI_COXIV_CENPA_WGA", "DAPI_CENPA"]
        )
    )
]

AGGREGATE_TARGETS = outputs_to_targets(
    AGGREGATE_OUTPUTS, aggregate_wildcard_combos, AGGREGATE_OUTPUT_MAPPINGS
)

AGGREGATE_TARGETS_ALL = AGGREGATE_TARGETS
