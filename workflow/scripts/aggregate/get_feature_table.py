import pyarrow.dataset as ds
import pandas as pd

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.aggregate import aggregate

# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input.filtered_paths, format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print(f"Shape of input data: {cell_data.shape}")

# Modify pert col for nontargeting entries by appending pert id value
mask = cell_data[snakemake.params.perturbation_name_col] == snakemake.params.control_key
cell_data.loc[mask, snakemake.params.perturbation_name_col] = cell_data.loc[mask, snakemake.params.perturbation_name_col] + '_' + cell_data.loc[mask, snakemake.params.perturbation_id_col]

# Determine feature cols
metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp, include_classification_cols=True
)
feature_cols = cell_data.columns.difference(metadata_cols, sort=False)

# Group by gene symbol and calculate median for each feature
feature_table = cell_data.groupby(snakemake.params.perturbation_name_col)[
    feature_cols
].median()
feature_table = feature_table.reset_index()

# Save
feature_table.to_csv(
    snakemake.output[0],
    sep="\t",
    index=False,
)
