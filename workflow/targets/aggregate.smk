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
        / "parquets"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "features_singlecell",
            "parquet",
        ),
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "features_constructs",
            "tsv",
        ),
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "features_genes",
            "tsv",
        ),
        AGGREGATE_FP
        / "tsvs"
        / get_filename(
            {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
            "construct_table",
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
    # Data preparation outputs
    "bootstrap_data_dir": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__bootstrap_data",
    "construct_data": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__bootstrap_data" / "{gene}__{construct}__construct_data.tsv",
    
    # Input arrays
    "controls_arr": AGGREGATE_FP / "bootstrap" / "inputs" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "controls_arr", "tsv"
    ),
    "construct_features_arr": AGGREGATE_FP / "bootstrap" / "inputs" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "construct_features_arr", "tsv"
    ),
    "sample_sizes": AGGREGATE_FP / "bootstrap" / "inputs" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "sample_sizes", "tsv"
    ),

    # Construct-level outputs
    "bootstrap_construct_nulls": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__constructs" / "{gene}__{construct}__nulls.npy",
    "bootstrap_construct_pvals": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__constructs" / "{gene}__{construct}__pvals.tsv",
    
    # Gene-level outputs
    "bootstrap_gene_nulls": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__genes" / "{gene}__nulls.npy",
    "bootstrap_gene_pvals": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__genes" / "{gene}__pvals.tsv",
    
    # Completion flags
    "bootstrap_flag": AGGREGATE_FP / "bootstrap" / "{cell_class}__{channel_combo}__bootstrap_complete.flag",

    # Combined results
    "combined_construct_results": AGGREGATE_FP / "bootstrap" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "all_construct_bootstrap_results", "tsv"
    ),
    "combined_gene_results": AGGREGATE_FP / "bootstrap" / get_filename(
        {"cell_class": "{cell_class}", "channel_combo": "{channel_combo}"},
        "all_gene_bootstrap_results", "tsv"
    ),
}

# Bootstrap target combinations
bootstrap_combos = config.get("aggregate", {}).get("bootstrap_combinations", [])
BOOTSTRAP_TARGETS_ALL = [
    str(output_path).format(cell_class=combo["cell_class"], channel_combo=combo["channel_combo"])
    for combo in bootstrap_combos
    for output_path in [
        BOOTSTRAP_OUTPUTS["combined_construct_results"],
        BOOTSTRAP_OUTPUTS["combined_gene_results"]
    ]
]
