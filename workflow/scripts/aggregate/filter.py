import pyarrow.dataset as ds
import pandas as pd

from lib.aggregate.filter import (
    perturbation_filter,
    missing_values_filter,
    intensity_filter,
)

# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input[0], format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print(f"Shape of input data: {cell_data.shape}")

# Filter
cell_data = perturbation_filter(
    cell_data,
    snakemake.params.perturbation_name_col,
    snakemake.params.perturbation_multi_col,
    snakemake.params.filter_single_pert,
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
