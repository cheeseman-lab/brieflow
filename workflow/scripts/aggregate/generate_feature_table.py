import gc

import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import (
    load_metadata_cols,
    split_cell_data,
    get_feature_table_cols,
)

# get snakemake parameters
pert_col = snakemake.params.perturbation_name_col
control_key = snakemake.params.control_key

# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input.filtered_paths, format="parquet")

# determine cols
cell_data_cols = cell_data.schema.names
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, True)
feature_cols = [col for col in cell_data.schema.names if col not in metadata_cols]
feature_cols = get_feature_table_cols(feature_cols)

# load cell data and convert numerical columns to float32
cell_data = cell_data.to_table(
    columns=metadata_cols + feature_cols, use_threads=True, memory_pool=None
).to_pandas()
print(f"Shape of input data: {cell_data.shape}")
for col in cell_data.columns:
    if is_numeric_dtype(cell_data[col]):
        cell_data[col] = cell_data[col].astype("float32")

# centerscale features on controls
# split metadata and features
metadata, features = split_cell_data(cell_data, metadata_cols)
del cell_data
gc.collect()
metadata, features = prepare_alignment_data(
    metadata,
    features,
    snakemake.params.batch_cols,
    pert_col,
    control_key,
    snakemake.params.perturbation_id_col,
)
features = features.astype(np.float32)

# centerscale features on controls
features = centerscale_on_controls(
    features,
    metadata,
    pert_col,
    control_key,
    "batch_values",
).astype(np.float32)

# Calculate sample sizes before aggregation
sample_sizes = metadata.groupby(pert_col, observed=True).size().reset_index(name='cell_count')

# Calculate average cells per sgRNA for each gene
sgrna_sizes = metadata.groupby([pert_col, snakemake.params.perturbation_id_col], observed=True).size().reset_index(name='sgrna_cell_count')
avg_cells_per_sgrna = sgrna_sizes.groupby(pert_col, observed=True)['sgrna_cell_count'].mean().reset_index()
avg_cells_per_sgrna.columns = [pert_col, 'avg_cells_per_sgrna']

# get the median of each perturbation
features = pd.DataFrame(features, columns=feature_cols)
features[pert_col] = metadata[pert_col].values
features_aggregated = features.groupby(pert_col, sort=False, observed=True).median()
features_aggregated = features_aggregated.reset_index()

# Merge sample sizes and average cells per sgRNA with aggregated features
features_with_counts = pd.merge(features_aggregated, sample_sizes, on=pert_col, how='left')
features_final = pd.merge(features_with_counts, avg_cells_per_sgrna, on=pert_col, how='left')

# Reorder columns to put count columns at the beginning
column_order = [pert_col, 'cell_count', 'avg_cells_per_sgrna'] + [col for col in features_final.columns if col not in [pert_col, 'cell_count', 'avg_cells_per_sgrna']]
features_final = features_final[column_order]

features_final.to_csv(snakemake.output[0], sep="\t", index=False)