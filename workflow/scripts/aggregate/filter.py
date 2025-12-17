import pandas as pd

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.filter import (
    query_filter,
    perturbation_filter,
    missing_values_filter,
    intensity_filter,
)

# Load cell data
cell_data = pd.read_parquet(snakemake.input[0])
metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp,
    include_classification_cols=False,
)
metadata, features = split_cell_data(cell_data, metadata_cols)

# Filter
metadata, features = query_filter(
    metadata,
    features,
    snakemake.params.filter_queries,
)
metadata, features = perturbation_filter(
    metadata,
    features,
    snakemake.params.perturbation_name_col,
)
metadata, features = missing_values_filter(
    metadata,
    features,
    drop_cols_threshold=snakemake.params.drop_cols_threshold,
    drop_rows_threshold=snakemake.params.drop_rows_threshold,
    impute=snakemake.params.impute,
)
metadata, features = intensity_filter(
    metadata,
    features,
    snakemake.params.channel_names,
    snakemake.params.contamination,
)

# Save filtered data
cell_data = pd.concat([metadata, features], axis=1)
cell_data.to_parquet(snakemake.output[0], index=False)
