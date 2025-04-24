import math
import gc

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
import numpy as np

np.random.seed(42)

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.align import (
    prepare_alignment_data,
    centerscale_by_batch,
    tvn_on_controls,
)

## Step 1: Create PCA transformation
PCA_SUBSET = 100000

# Load full dataset as logical PyArrow dataset
cell_dataset = ds.dataset(snakemake.input.filtered_paths, format="parquet")
print(f"Number of rows across all parquet files: {cell_dataset.count_rows()}")

# Count total rows across all Parquet files
total_rows = cell_dataset.count_rows()

# Choose random row indices
n_sample = min(PCA_SUBSET, total_rows)
random_indices = np.random.choice(total_rows, size=n_sample, replace=False)
random_indices.sort()

# load sample df
sample_df = cell_dataset.scanner().take(random_indices)
sample_df = sample_df.to_pandas(use_threads=True, memory_pool=None)

# load sample df as pandas dataframe
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, True)
metadata, features = split_cell_data(sample_df, metadata_cols)
metadata, features = prepare_alignment_data(
    metadata,
    features,
    snakemake.params.batch_cols,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    snakemake.params.perturbation_id_col,
)
pca = PCA(n_components=snakemake.params.variance_or_ncomp).fit(
    centerscale_by_batch(features, metadata, "batch_values")
)

## Step 2: Batched alignment

# Determine subset indices
subset_factor = snakemake.params.subset_factor
all_indices = np.random.permutation(total_rows)
chunk_size = math.ceil(total_rows / subset_factor)
subset_indices = [
    all_indices[i * chunk_size : (i + 1) * chunk_size] for i in range(subset_factor)
]

# Process each batch
writer = None
for i, indices in enumerate(subset_indices):
    print(f"Processing subset {i + 1}/{subset_factor} with {len(indices)} cells")

    subset_df = cell_dataset.scanner().take(pa.array(indices))
    subset_df = subset_df.to_pandas(use_threads=True, memory_pool=None)
    subset_df = subset_df.dropna(axis=1)

    for col in subset_df.columns:
        if is_numeric_dtype(subset_df[col]):
            subset_df[col] = subset_df[col].astype("float32")

    metadata, features = split_cell_data(subset_df, metadata_cols)
    del subset_df
    gc.collect()

    metadata, features = prepare_alignment_data(
        metadata,
        features,
        snakemake.params.batch_cols,
        snakemake.params.perturbation_name_col,
        snakemake.params.control_key,
        snakemake.params.perturbation_id_col,
    )

    features = centerscale_by_batch(features, metadata, "batch_values")

    features = pca.transform(features)

    features = tvn_on_controls(
        features,
        metadata,
        snakemake.params.perturbation_name_col,
        snakemake.params.control_key,
        "batch_values",
    )

    feature_columns = [f"PC_{j}" for j in range(features.shape[1])]
    features = pd.DataFrame(features, index=metadata.index, columns=feature_columns)
    aligned_cell_data = pd.concat([metadata, features], axis=1)
    del features
    gc.collect()

    # Convert to Arrow table and write chunk
    aligned_cell_data = pa.Table.from_pandas(aligned_cell_data, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(snakemake.output[0], aligned_cell_data.schema)
    writer.write_table(aligned_cell_data)

# Close writer
if writer is not None:
    writer.close()
