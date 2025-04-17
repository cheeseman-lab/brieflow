import gc

import pyarrow.dataset as ds
import pandas as pd
from pandas.api.types import is_numeric_dtype

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.align import (
    prepare_alignment_data,
    embed_by_pca,
    tvn_on_controls,
)

# Load cell data using PyArrow dataset
cell_data = ds.dataset(snakemake.input.filtered_paths, format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()

# convert number columns to float32 to save memory
for col in cell_data.columns:
    if is_numeric_dtype(cell_data[col]):
        cell_data[col] = cell_data[col].astype("float32")

# Prepare alignment data
metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp,
    include_classification_cols=True,
)
metadata, features = split_cell_data(cell_data, metadata_cols)
del cell_data
gc.collect()
features, metadata = prepare_alignment_data(
    metadata,
    features,
    snakemake.params.batch_cols,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    snakemake.params.perturbation_id_col,
)

# Embed data using PCA
features = embed_by_pca(
    features,
    metadata,
    variance_or_ncomp=snakemake.params.variance_or_ncomp,
    batch_col="batch_values",
)

# Perform tvn alignment
features = tvn_on_controls(
    features,
    metadata,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    "batch_values",
)

# Save aligned data
feature_columns = [f"PC_{i}" for i in range(features.shape[1])]
features = pd.DataFrame(features, index=metadata.index, columns=feature_columns)
embeddings_df = pd.concat([metadata, features], axis=1)
del features
gc.collect()

embeddings_df.to_parquet(snakemake.output[0], index=False)
