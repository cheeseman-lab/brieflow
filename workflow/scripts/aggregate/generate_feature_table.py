import gc

import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data

# get snakemake parameters
pert_col = snakemake.params.perturbation_name_col
control_key = snakemake.params.control_key

# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input.filtered_paths, format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
for col in cell_data.columns:
    if is_numeric_dtype(cell_data[col]):
        cell_data[col] = cell_data[col].astype("float32")
print(f"Shape of input data: {cell_data.shape}")

# load sample df as pandas dataframe
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, True)
feature_cols = cell_data.columns.difference(metadata_cols, sort=False)

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

# Modify pert col for nontargeting entries by appending pert id value
control_ind = metadata[pert_col].str.startswith(control_key).to_list()
metadata.loc[control_ind, pert_col] = (
    metadata.loc[control_ind, pert_col]
    + "_"
    + metadata.loc[control_ind, snakemake.params.perturbation_id_col]
)

features_table = metadata[[pert_col]].drop_duplicates().set_index(pert_col)

# perform centerscaling in batches
col_idxs = np.arange(features.shape[1])
idx_chunks = np.array_split(col_idxs, snakemake.params.batches)
for idxs in idx_chunks:
    cols = feature_cols[idxs]
    chunk = features[:, idxs]

    chunk_scaled = centerscale_on_controls(
        chunk,
        metadata,
        pert_col,
        control_key,
        "batch_values",
    ).astype(np.float32)

    chunk_scaled = pd.DataFrame(chunk_scaled, columns=cols)
    chunk_scaled[pert_col] = metadata[pert_col].values
    chunk_median = chunk_scaled.groupby(pert_col, sort=False, observed=True).median()

    features_table = features_table.join(chunk_median, how="left")

features_table.reset_index().to_csv(snakemake.output[0], sep="\t", index=False)
