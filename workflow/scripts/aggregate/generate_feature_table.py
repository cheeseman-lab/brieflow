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
pert_id_col = snakemake.params.perturbation_id_col
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
    pert_id_col,
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

# TABLE 1: Construct-level table (one row per sgRNA)
print("Creating construct-level table...")

# Calculate sample sizes at sgRNA level
construct_sample_sizes = (
    metadata.groupby(pert_id_col, observed=True).size().reset_index(name="cell_count")
)

# Get corresponding gene for each sgRNA
construct_gene_map = (
    metadata.groupby(pert_id_col, observed=True)[pert_col].first().reset_index()
)

# Get median features at sgRNA level
features = pd.DataFrame(features, columns=feature_cols)
features[pert_id_col] = metadata[pert_id_col].values

construct_features = features.groupby(pert_id_col, sort=False, observed=True).median()
construct_features = construct_features.reset_index()

# Merge everything for construct table
construct_table = pd.merge(
    construct_features, construct_sample_sizes, on=pert_id_col, how="left"
)
construct_table = pd.merge(
    construct_table, construct_gene_map, on=pert_id_col, how="left"
)

# Reorder columns: sgRNA, gene, cell_count, features
construct_columns = [pert_id_col, pert_col, "cell_count"] + feature_cols
construct_table = construct_table[construct_columns]

print(f"Construct table shape: {construct_table.shape}")

# TABLE 2: Gene-level table (median of construct medians)
print("Creating gene-level table...")

# Filter out controls for gene-level aggregation
non_control_constructs = construct_table[
    ~construct_table[pert_col].str.contains(control_key, na=False)
]

# Calculate gene-level sample sizes (sum of construct cell counts)
gene_sample_sizes = (
    non_control_constructs.groupby(pert_col, observed=True)["cell_count"]
    .sum()
    .reset_index()
)
gene_sample_sizes.columns = [pert_col, "cell_count"]

# Calculate gene-level medians (median of construct medians)
gene_features = non_control_constructs.groupby(pert_col, sort=False, observed=True)[
    feature_cols
].median()
gene_features = gene_features.reset_index()

# Merge gene features with sample sizes
gene_table = pd.merge(gene_features, gene_sample_sizes, on=pert_col, how="left")

# Add controls to gene table (controls are their own "genes")
control_constructs = construct_table[
    construct_table[pert_col].str.contains(control_key, na=False)
]
control_gene_table = control_constructs[[pert_col, "cell_count"] + feature_cols].copy()

# Combine gene table with controls
final_gene_table = pd.concat([gene_table, control_gene_table], ignore_index=True)

# Reorder columns: gene, cell_count, features
gene_columns = [pert_col, "cell_count"] + feature_cols
final_gene_table = final_gene_table[gene_columns]

print(f"Gene table shape: {final_gene_table.shape}")

# Save both tables
construct_output = snakemake.output[0].replace(
    "feature_table.tsv", "construct_table.tsv"
)
gene_output = snakemake.output[0]  # Keep original name for gene table

construct_table.to_csv(construct_output, sep="\t", index=False)
final_gene_table.to_csv(gene_output, sep="\t", index=False)

print(f"Saved construct table to: {construct_output}")
print(f"Saved gene table to: {gene_output}")
