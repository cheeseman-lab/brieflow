import gc

import pyarrow.dataset as ds
import pandas as pd
from pandas.api.types import is_numeric_dtype

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

for col in cell_data.columns:
    if is_numeric_dtype(cell_data[col]):
        cell_data[col] = cell_data[col].astype("float32")

print("Converted number columns to float32")
print(cell_data.info())
mem = psutil.Process().memory_info().rss / 1e6  # MB
print(f"{datetime.now().isoformat()} - RSS Memory: {mem:.2f} MB")

# Align
features, metadata = prepare_alignment_data(
    cell_data,
    snakemake.params.batch_cols,
    snakemake.params.first_feature,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    snakemake.params.perturbation_id_col,
)
del cell_data
gc.collect()

gc.collect()
print("Done preparing alignment data")
print(metadata.info())
print(f"Total features size: {features.nbytes / 1e6:.2f} MB")
mem = psutil.Process().memory_info().rss / 1e6  # MB
print(f"{datetime.now().isoformat()} - RSS Memory: {mem:.2f} MB")

features = embed_by_pca(
    features,
    metadata,
    variance_or_ncomp=snakemake.params.pc_count,
    batch_col="batch_values",
)
gc.collect()
print("Done performing PCA")
print(f"Total features size: {features.nbytes / 1e6:.2f} MB")
mem = psutil.Process().memory_info().rss / 1e6  # MB
print(f"{datetime.now().isoformat()} - RSS Memory: {mem:.2f} MB")

features = tvn_on_controls(
    features,
    metadata,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
    "batch_values",
)
gc.collect()
print("Done performing TVN")
print(f"Total features size: {features.nbytes / 1e6:.2f} MB")
mem = psutil.Process().memory_info().rss / 1e6  # MB
print(f"{datetime.now().isoformat()} - RSS Memory: {mem:.2f} MB")

# Save aligned data
feature_columns = [f"PC_{i}" for i in range(features.shape[1])]
features = pd.DataFrame(features, index=metadata.index, columns=feature_columns)
gc.collect()
embeddings_df = pd.concat([metadata, features], axis=1)
del features
gc.collect()
embeddings_df.to_parquet(snakemake.output[0], index=False)
mem = psutil.Process().memory_info().rss / 1e6  # MB
print(f"{datetime.now().isoformat()} - RSS Memory: {mem:.2f} MB")
