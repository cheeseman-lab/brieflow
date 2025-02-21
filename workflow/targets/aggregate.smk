from lib.shared.file_utils import get_filename
from lib.shared.target_utils import map_outputs, outputs_to_targets


AGGREGATE_FP = ROOT_FP / "aggregate"

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

NON_MONTAGE_WILDCARDS = {
    "plate": MERGE_PLATES,
    "well": MERGE_WELLS,
}
NON_MONTAGE_OUTPUTS = {
    rule_name: templates
    for rule_name, templates in AGGREGATE_OUTPUTS.items()
    if "generate" not in rule_name
}
NON_MONTAGE_TARGETS = outputs_to_targets(
    NON_MONTAGE_OUTPUTS, NON_MONTAGE_WILDCARDS, AGGREGATE_OUTPUT_MAPPINGS
)


# Combine all preprocessing targets
AGGREGATE_TARGETS_ALL = sum(NON_MONTAGE_TARGETS.values(), [])
