import gc

import pyarrow.dataset as ds
import pandas as pd

from lib.aggregate.align import (
    prepare_alignment_data,
    embed_by_pca,
    tvn_on_controls,
)

import threading
import time
import psutil
from datetime import datetime
import sys

LOG_PATH = "/lab/barcheese01/rkern/aggregate_overhaul/mem_usage.txt"
sys.stdout = open(LOG_PATH, "a")


def log_memory_usage():
    while True:
        mem = psutil.Process().memory_info().rss / 1e6  # MB
        print(f"{datetime.now().isoformat()} - RSS Memory: {mem:.2f} MB")
        sys.stdout.flush()
        time.sleep(30)


threading.Thread(target=log_memory_usage, daemon=True).start()


# Load cell data using PyArrow dataset
print("Loading cell data")
cell_data = ds.dataset(snakemake.input.filtered_paths, format="parquet")
cell_data = cell_data.to_table(use_threads=True, memory_pool=None).to_pandas()
print(f"Shape of input data: {cell_data.shape}")
print(cell_data.info())
gc.collect()

for col in cell_data.select_dtypes("Int64").columns:
    cell_data[col] = cell_data[col].astype("float32")

for col in cell_data.select_dtypes("int64").columns:
    cell_data[col] = cell_data[col].astype("float32")
print("Converted int64 and Int64 columns to float32")
print(cell_data.info())

# Align
features, metadata = prepare_alignment_data(
    cell_data, snakemake.params.batch_cols, snakemake.params.first_feature
)
del cell_data
gc.collect()

features = features.values
metadata = metadata[[snakemake.params.perturbation_name_col, "batch_values"]]
print("Done preparing alignment data")
print(metadata.info())
print(f"Total features size: {features.nbytes / 1e6:.2f} MB")

pca_embeddings = embed_by_pca(
    features,
    metadata,
    variance_or_ncomp=snakemake.params.pc_count,
    batch_col="batch_values",
)
del features
gc.collect()
print("Done performing PCA")

tvn_normalized = tvn_on_controls(
    pca_embeddings,
    metadata,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    "batch_values",
)
del pca_embeddings
gc.collect()

# Save aligned data
feature_columns = [f"PC_{i}" for i in range(tvn_normalized.shape[1])]
embeddings_df = pd.DataFrame(
    tvn_normalized, index=metadata.index, columns=feature_columns
)
cell_data = pd.concat([metadata, embeddings_df], axis=1)
cell_data.to_parquet(snakemake.output[0], index=False)
