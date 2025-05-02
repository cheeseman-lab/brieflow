from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


AGGREGATE_FP = ROOT_FP / "aggregate"

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
            "feature_distributions",
            "png",
        ),
    ],
    "get_feature_table": [
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "feature_table",
            "tsv",
        ),
    ],
}

AGGREGATE_OUTPUT_MAPPINGS = {
    "split_datasets": None,
    "filter": None,
    "align": None,
    "aggregate": None,
    "eval_aggregate": None,
    "get_feature_table": None,
}

AGGREGATE_OUTPUTS_MAPPED = map_outputs(AGGREGATE_OUTPUTS, AGGREGATE_OUTPUT_MAPPINGS)

# TODO: Use all combos
# aggregate_wildcard_combos = aggregate_wildcard_combos[
#     (aggregate_wildcard_combos["plate"].isin([1]))
#     & (aggregate_wildcard_combos["well"].isin(["A1"]))
#     & (aggregate_wildcard_combos["cell_class"].isin(["Mitotic"]))
#     & (aggregate_wildcard_combos["channel_combo"].isin(["DAPI_CENPA"]))
# ]

AGGREGATE_TARGETS_ALL = outputs_to_targets(
    AGGREGATE_OUTPUTS, aggregate_wildcard_combos, AGGREGATE_OUTPUT_MAPPINGS
)


# Define montage outputs
# These are special because we dynamically derive outputs
MONTAGE_OUTPUTS = {
    "montage_data_dir": AGGREGATE_FP / "montages" / "{cell_class}__montage_data",
    "montage_data": AGGREGATE_FP
    / "montages"
    / "{cell_class}__montage_data"
    / get_filename(
        {"gene": "{gene}", "sgrna": "{sgrna}"},
        "montage_data",
        "tsv",
    ),
    "montage": AGGREGATE_FP
    / "montages"
    / "{cell_class}__montages"
    / "{gene}"
    / "{sgrna}"
    / get_filename(
        {"channel": "{channel}"},
        "montage",
        "png",
    ),
    "montage_overlay": AGGREGATE_FP
    / "montages"
    / "{cell_class}__montages"
    / "{gene}"
    / "{sgrna}"
    / get_filename(
        {},
        "overlay_montage",
        "tiff",
    ),
    "montage_flag": AGGREGATE_FP / "montages" / "{cell_class}__montages_complete.flag",
}
cell_classes = aggregate_wildcard_combos["cell_class"].unique()
MONTAGE_TARGETS_ALL = [
    str(MONTAGE_OUTPUTS["montage_flag"]).format(cell_class=cell_class)
    for cell_class in cell_classes
]
