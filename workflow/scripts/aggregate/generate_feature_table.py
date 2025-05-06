import gc

import pyarrow.dataset as ds
import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls, centerscale_by_batch
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data

# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input.filtered_paths, format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print(f"Shape of input data: {cell_data.shape}")

# load sample df as pandas dataframe
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, True)
feature_cols = cell_data.columns.difference(metadata_cols, sort=False)

# centerscale features on controls
# split metadata and features
metadata, features = split_cell_data(cell_data, metadata_cols)
del cell_data
gc.collect()
print(features.info())
metadata, features = prepare_alignment_data(
    metadata,
    features,
    snakemake.params.batch_cols,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    snakemake.params.perturbation_id_col,
)

# Modify pert col for nontargeting entries by appending pert id value
control_ind = metadata[snakemake.params.perturbation_name_col].str.startswith(snakemake.params.control_key).to_list()
metadata.loc[control_ind, snakemake.params.perturbation_name_col] = (
    metadata.loc[control_ind, snakemake.params.perturbation_name_col]
    + "_"
    + metadata.loc[control_ind, snakemake.params.perturbation_id_col]
)

# center scaling by column
for i, col in enumerate(feature_cols):
    control_vals = features[control_ind, i].reshape(-1, 1)

    # center scale by controls
    features[:, i] = StandardScaler(copy=False).fit(control_vals).transform(features[:, i].reshape(-1, 1)).ravel()

    print(i)
    if i > 20:
        break

# Group by gene symbol and calculate median for each feature
features = pd.DataFrame(features.astype(np.float16), columns=feature_cols)
print(features)
features[snakemake.params.perturbation_name_col] = metadata[snakemake.params.perturbation_name_col]
features = features.groupby(snakemake.params.perturbation_name_col).median().reset_index()
print(features)

# Save
features.to_csv(
    snakemake.output[0],
    sep="\t",
    index=False,
)
