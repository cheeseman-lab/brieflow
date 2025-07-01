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

# Define vacuole-specific columns to include
vacuole_cols = [
    "num_vacuoles",
    "total_vacuole_area",
    "vacuole_area_ratio",
    "mean_vacuole_diameter",
    "mean_distance_to_nucleus",
]

# Add vacuole columns to feature_cols if they exist in the data
available_vacuole_cols = [col for col in vacuole_cols if col in cell_data_cols]
if available_vacuole_cols:
    feature_cols.extend(available_vacuole_cols)
    print(f"Added vacuole columns: {available_vacuole_cols}")
else:
    print("Warning: No vacuole columns found in the dataset")

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

# Modify pert col for nontargeting entries by appending pert id value
control_ind = metadata[pert_col].str.startswith(control_key).to_list()
metadata.loc[control_ind, pert_col] = (
    metadata.loc[control_ind, pert_col]
    + "_"
    + metadata.loc[control_ind, snakemake.params.perturbation_id_col]
)

# centerscale features on controls
features = centerscale_on_controls(
    features,
    metadata,
    pert_col,
    control_key,
    "batch_values",
).astype(np.float32)

# get the median of each perturbation
features = pd.DataFrame(features, columns=feature_cols)
features[pert_col] = metadata[pert_col].values
features = features.groupby(pert_col, sort=False, observed=True).median()
features = features.reset_index()

features.to_csv(snakemake.output[0], sep="\t", index=False)
