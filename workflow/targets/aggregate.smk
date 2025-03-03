from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


AGGREGATE_FP = ROOT_FP / "aggregate"

# Define standard (non-montage) aggreagte outputs
AGGREGATE_OUTPUTS = {
    "clean_transform_standardize": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "cleaned_data", "parquet"
        ),
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "transformed_data", "parquet"
        ),
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "standardized_data", "parquet"
        ),
    ],
    "split_phases": [
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "mitotic_data", "parquet"
        ),
        AGGREGATE_FP
        / "parquets"
        / get_filename(
            {"plate": "{plate}", "well": "{well}"}, "interphase_data", "parquet"
        ),
    ],
    "process_mitotic_gene_data": [
        AGGREGATE_FP / "tsvs" / "mitotic_gene_data.tsv",
    ],
    "process_interphase_gene_data": [
        AGGREGATE_FP / "tsvs" / "interphase_gene_data.tsv",
    ],
    "process_all_gene_data": [
        AGGREGATE_FP / "tsvs" / "all_gene_data.tsv",
    ],
    "eval_aggregate": [
        AGGREGATE_FP / "eval" / "cell_feature_violins.png",
        AGGREGATE_FP / "eval" / "nuclear_feature_violins.png",
        AGGREGATE_FP / "eval" / "mitotic_missing.tsv",
        AGGREGATE_FP / "eval" / "interphase_missing.tsv",
        AGGREGATE_FP / "eval" / "all_missing.tsv",
        AGGREGATE_FP / "eval" / "mitotic_stats.tsv",
    ],
}

AGGREGATE_OUTPUT_MAPPINGS = {
    "clean_transform_standardize": None,
    "split_phases": None,
    "process_mitotic_gene_data": None,
    "process_interphase_gene_data": None,
    "process_all_gene_data": None,
    "eval_aggregate": None,
}

AGGREGATE_OUTPUTS_MAPPED = map_outputs(AGGREGATE_OUTPUTS, AGGREGATE_OUTPUT_MAPPINGS)

AGGREGATE_TARGETS = outputs_to_targets(
    AGGREGATE_OUTPUTS, merge_wildcard_combos, AGGREGATE_OUTPUT_MAPPINGS
)

# Define montage outputs
# These are special because we dynamically derive outputs
MONTAGE_OUTPUTS = {
    "mitotic_montage_data_dir": AGGREGATE_FP / "montages" / "mitotic_montage_data",
    "mitotic_montage_data": AGGREGATE_FP
    / "montages"
    / "mitotic_montage_data"
    / get_filename(
        {"gene": "{gene}", "sgrna": "{sgrna}"},
        "montage_data",
        "tsv",
    ),
    "mitotic_montage": AGGREGATE_FP
    / "montages"
    / "mitotic_montages"
    / "{gene}"
    / get_filename(
        {"sgrna": "{sgrna}", "channel": "{channel}"},
        "montage",
        "tiff",
    ),
    "mitotic_montage_flag": AGGREGATE_FP
    / "montages"
    / "mitotic_montages_complete.flag",
    "interphase_montage_data_dir": AGGREGATE_FP
    / "montages"
    / "interphase_montage_data",
    "interphase_montage_data": AGGREGATE_FP
    / "montages"
    / "interphase_montage_data"
    / get_filename(
        {"gene": "{gene}", "sgrna": "{sgrna}"},
        "montage_data",
        "tsv",
    ),
    "interphase_montage": AGGREGATE_FP
    / "montages"
    / "interphase_montages"
    / "{gene}"
    / get_filename(
        {"sgrna": "{sgrna}", "channel": "{channel}"},
        "montage",
        "tiff",
    ),
    "interphase_montage_flag": AGGREGATE_FP
    / "montages"
    / "interphase_montages_complete.flag",
}

AGGREGATE_TARGETS_ALL = AGGREGATE_TARGETS + [
    MONTAGE_OUTPUTS["mitotic_montage_flag"],
    MONTAGE_OUTPUTS["interphase_montage_flag"],
]
