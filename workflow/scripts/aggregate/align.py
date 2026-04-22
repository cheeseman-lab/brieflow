import math
import gc
import warnings

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.decomposition import PCA
import numpy as np

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.align import (
    prepare_alignment_data,
    centerscale_by_batch,
    tvn_on_controls,
)
from lib.aggregate.filter import harmonize_pool_schema
from lib.aggregate.perturbation_score import perturbation_score

warnings.filterwarnings(
    "ignore", category=UserWarning, module="sklearn.feature_selection"
)
warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.feature_selection"
)
np.random.seed(0)

## Step 1: Create PCA transformation
PCA_SUBSET = 100000

# Filter out empty parquet files to avoid schema conflicts
non_empty_paths = [
    p for p in snakemake.input.filtered_paths if pq.read_metadata(p).num_rows > 0
]

if len(non_empty_paths) == 0:
    print("WARNING: No cells in input, writing empty output")
    pq.write_table(pa.table({}), snakemake.output[0])
    exit(0)

# Load full dataset as logical PyArrow dataset (only non-empty files)
cell_dataset = ds.dataset(non_empty_paths, format="parquet")
total_rows = cell_dataset.count_rows()
print(
    f"Number of rows across {len(non_empty_paths)} non-empty parquet files: {total_rows}"
)

# Choose random row indices
n_sample = min(PCA_SUBSET, total_rows)
random_indices = np.random.choice(total_rows, size=n_sample, replace=False)
random_indices.sort()

# load sample df as pandas dataframe
use_classifier = snakemake.params.get("use_classifier", False)
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, use_classifier)

# Harmonize the pool's column set once: drop cols present in only some per-well
# files (typically driven by per-well filter decisions diverging when class
# composition differs across wells) and apply drop_cols_threshold at the pool
# level — matching the convention used by missing_values_filter.
kept_metadata_cols, kept_feature_cols, _pool_report = harmonize_pool_schema(
    non_empty_paths,
    metadata_cols,
    drop_cols_threshold=snakemake.params.get("drop_cols_threshold"),
)
scan_cols = kept_metadata_cols + kept_feature_cols

# load sample df
sample_df = (
    cell_dataset.scanner(columns=scan_cols)
    .take(random_indices)
    .to_pandas(use_threads=True, memory_pool=None)
)

# Residual NaN handling: after pool-level col drops, any remaining NaN is
# sparse and row-localized. Drop those rows from the PCA training sample so
# PCA/StandardScaler see a clean matrix. Matches the drop_rows_threshold intent
# at a per-row granularity appropriate for a training sample.
_sample_pre = len(sample_df)
sample_df = sample_df.dropna(subset=kept_feature_cols).reset_index(drop=True)
if len(sample_df) < _sample_pre:
    print(
        f"[pool] dropped {_sample_pre - len(sample_df)} PCA-sample rows with residual NaN"
    )
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
num_align_batches = snakemake.params.num_align_batches
all_indices = np.random.permutation(total_rows)
chunk_size = math.ceil(total_rows / num_align_batches)
subset_indices = [
    all_indices[i * chunk_size : (i + 1) * chunk_size] for i in range(num_align_batches)
]

# Process each batch
writer = None
for i, indices in enumerate(subset_indices):
    print(f"Processing subset {i + 1}/{num_align_batches} with {len(indices)} cells")

    subset_df = (
        cell_dataset.scanner(columns=scan_cols)
        .take(pa.array(indices))
        .to_pandas(use_threads=True, memory_pool=None)
    )
    # Drop any residual per-row NaN (pool-level col drops above have already
    # eliminated systematically-missing columns).
    _pre = len(subset_df)
    subset_df = subset_df.dropna(subset=kept_feature_cols).reset_index(drop=True)
    if len(subset_df) < _pre:
        print(
            f"[pool] dropped {_pre - len(subset_df)} rows with residual NaN "
            f"from batch {i + 1}"
        )

    # CALCULATE PERTURBATION SCORE
    subset_df["perturbation_score"] = np.nan
    subset_df["perturbation_auc"] = np.nan
    metadata_cols += ["perturbation_score", "perturbation_auc"]
    if not snakemake.params.skip_perturbation_score:
        perturbation_score(
            subset_df,
            metadata_cols,
            snakemake.params.perturbation_name_col,
            snakemake.params.control_key,
        )

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
        control_col=snakemake.params.get("control_name_col"),
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
