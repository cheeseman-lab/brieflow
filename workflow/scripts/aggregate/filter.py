import pandas as pd

from lib.aggregate.filter import (
    query_filter,
    perturbation_filter,
    missing_values_filter,
    intensity_filter,
)

# Load cell data
cell_data = pd.read_parquet(snakemake.input[0])

# Filter
cell_data = query_filter(
    cell_data,
    snakemake.params.filter_queries,
)
cell_data = perturbation_filter(
    cell_data,
    snakemake.params.perturbation_name_col,
)
cell_data = missing_values_filter(
    cell_data,
    snakemake.params.first_feature,
    drop_cols_threshold=snakemake.params.drop_cols_threshold,
    drop_rows_threshold=snakemake.params.drop_rows_threshold,
    impute=snakemake.params.impute,
)
cell_data = intensity_filter(
    cell_data,
    snakemake.params.first_feature,
    snakemake.params.channel_names,
    snakemake.params.contamination,
)
# Save filtered data
cell_data.to_parquet(snakemake.output[0], index=False)
