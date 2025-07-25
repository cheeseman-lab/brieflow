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
    "generate_feature_table": [
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "feature_table",
            "tsv",
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
}

AGGREGATE_OUTPUT_MAPPINGS = {
    "split_datasets": None,
    "filter": None,
    "align": None,
    "aggregate": None,
    "eval_aggregate": None,
    "generate_feature_table": None,
}

AGGREGATE_OUTPUTS_MAPPED = map_outputs(AGGREGATE_OUTPUTS, AGGREGATE_OUTPUT_MAPPINGS)

# TODO: Use all combos
# aggregate_wildcard_combos = aggregate_wildcard_combos[
#     (aggregate_wildcard_combos["plate"].isin([1]))
#     & (aggregate_wildcard_combos["well"].isin(["A1"]))
#     & (aggregate_wildcard_combos["cell_class"].isin(["Mitotic"]))
#     & (aggregate_wildcard_combos["channel_combo"].isin(["DAPI_WGA"]))
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


# Define bootstrap outputs
# These are special because we dynamically derive outputs
BOOTSTRAP_OUTPUTS = {
    "bootstrap_data_dir": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__bootstrap_data",
    # Update to use simple naming pattern
    "construct_data": AGGREGATE_FP
    / "bootstrap"
    / "{cell_class}__{channel_combo}__bootstrap_data"
    / "{construct}_construct_data.csv",
    
    "controls_arr": AGGREGATE_FP / "bootstrap" / "inputs" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "controls_arr", "npy"
    ),
    "construct_features_arr": AGGREGATE_FP / "bootstrap" / "inputs" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "construct_features_arr", "npy"
    ),
    "sample_sizes": AGGREGATE_FP / "bootstrap" / "inputs" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "sample_sizes", "csv"
    ),
    "bootstrap_construct_nulls": AGGREGATE_FP
    / "bootstrap"
    / "{cell_class}__{channel_combo}__constructs"
    / "{construct}_nulls.npy",
    
    "bootstrap_construct_pvals": AGGREGATE_FP
    / "bootstrap"
    / "{cell_class}__{channel_combo}__constructs"
    / "{construct}_pvals.csv",
    
    "bootstrap_gene_nulls": AGGREGATE_FP
    / "bootstrap"
    / "{cell_class}__{channel_combo}__genes"
    / "{gene}_nulls.npy",
    
    "bootstrap_gene_pvals": AGGREGATE_FP
    / "bootstrap"
    / "{cell_class}__{channel_combo}__genes"
    / "{gene}_pvals.csv",
    
    "bootstrap_flag": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__bootstrap_complete.flag",
}

# Create bootstrap targets following montage pattern
cell_classes = aggregate_wildcard_combos["cell_class"].unique()
channel_combos = aggregate_wildcard_combos["channel_combo"].unique()
BOOTSTRAP_TARGETS_ALL = [
    str(BOOTSTRAP_OUTPUTS["bootstrap_flag"]).format(
        cell_class=cell_class, 
        channel_combo=channel_combo
    )
    for cell_class in cell_classes 
    for channel_combo in channel_combos
]