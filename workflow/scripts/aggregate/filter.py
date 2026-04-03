import pandas as pd

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.filter import (
    query_filter,
    perturbation_filter,
    missing_values_filter,
    intensity_filter,
)


def write_empty_and_exit(metadata, features, output_path, stage):
    """Write empty parquet and exit if no cells remain after a filtering stage."""
    if len(metadata) == 0:
        print(f"WARNING: No cells after {stage}, writing empty output")
        pd.concat([metadata, features], axis=1).to_parquet(output_path, index=False)
        exit(0)


# Validate required params
for _param_name in ["metadata_cols_fp", "perturbation_name_col", "channel_names"]:
    if getattr(snakemake.params, _param_name, None) is None:
        raise ValueError(f"Required config parameter '{_param_name}' is not set")

# Load cell data
cell_data = pd.read_parquet(snakemake.input[0])
use_classifier = snakemake.params.use_classifier
metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp,
    include_classification_cols=use_classifier,
)
metadata, features = split_cell_data(cell_data, metadata_cols)
output_path = snakemake.output[0]

# Filter
metadata, features = query_filter(
    metadata,
    features,
    snakemake.params.filter_queries,
)
write_empty_and_exit(metadata, features, output_path, "query_filter")

metadata, features = perturbation_filter(
    metadata,
    features,
    snakemake.params.perturbation_name_col,
)
write_empty_and_exit(metadata, features, output_path, "perturbation_filter")

metadata, features = missing_values_filter(
    metadata,
    features,
    drop_cols_threshold=snakemake.params.drop_cols_threshold,
    drop_rows_threshold=snakemake.params.drop_rows_threshold,
    impute=snakemake.params.impute,
)
write_empty_and_exit(metadata, features, output_path, "missing_values_filter")

metadata, features = intensity_filter(
    metadata,
    features,
    snakemake.params.channel_names,
    snakemake.params.contamination,
)

# Save filtered data
cell_data = pd.concat([metadata, features], axis=1)
cell_data.to_parquet(output_path, index=False)
