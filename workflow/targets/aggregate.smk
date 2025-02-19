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
    "prepare_mitotic_montage_data": [
        AGGREGATE_FP / "parquets" / "mitotic_montage_data.parquet",
    ],
    "prepare_interphase_montage_data": [
        AGGREGATE_FP / "parquets" / "interphase_montage_data.parquet",
    ],
    "generate_mitotic_montage": [
        AGGREGATE_FP
        / "tiffs"
        / "mitotic_montages"
        / get_filename(
            {"gene": "{gene}", "sgrna": "{sgrna}", "channel": "{channel}"},
            "montage",
            "tiff",
        ),
    ],
    "generate_interphase_montage": [
        AGGREGATE_FP
        / "tiffs"
        / "interphase_montages"
        / "{gene}"
        / get_filename(
            {"sgrna": "{sgrna}", "channel": "{channel}"},
            "montage",
            "tiff",
        ),
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
    "prepare_mitotic_montage_data": temp,
    "prepare_interphase_montage_data": temp,
    "generate_mitotic_montage": None,
    "generate_interphase_montage": None,
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

# determine combinations of genes, sgrna, and channel combinations from pool design file
# get each gene/sgrna combination that has a dialout value of 0 or 1
df_design = pd.read_csv(config["sbs"]["df_design_path"], sep="\t")
montage_combinations = (
    df_design.query("dialout == [0, 1]")
    .drop_duplicates("sgRNA")[["gene_symbol", "sgRNA"]]
    .rename(columns={"gene_symbol": "gene_symbol_0", "sgRNA": "sgRNA_0"})
    .drop_duplicates()
)
channels = config["phenotype"]["channel_names"]
# explode channels across each gene/sgrna combination
montage_combinations = (
    montage_combinations.assign(key=1)
    .merge(pd.DataFrame({"channel": channels, "key": 1}), on="key")
    .drop("key", axis=1)
)
# TODO: remove montage limit
montage_combinations = montage_combinations.head(100)
print(montage_combinations)

MONTAGE_WILDCARDS = {
    "gene": montage_combinations["gene_symbol_0"].to_list(),
    "sgrna": montage_combinations["sgRNA_0"].to_list(),
    "channel": montage_combinations["channel"].to_list(),
}
MONTAGE_OUTPUTS = {
    rule_name: templates
    for rule_name, templates in AGGREGATE_OUTPUTS.items()
    if "generate" in rule_name
}
MONTAGE_TARGETS = outputs_to_targets(
    MONTAGE_OUTPUTS,
    MONTAGE_WILDCARDS,
    AGGREGATE_OUTPUT_MAPPINGS,
    expansion_method="zip",
)

# Combine all preprocessing targets
AGGREGATE_TARGETS_ALL = sum(NON_MONTAGE_TARGETS.values(), []) + sum(
    MONTAGE_TARGETS.values(), []
)
