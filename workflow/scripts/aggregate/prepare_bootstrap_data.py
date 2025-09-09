"""Prepare bootstrap data for statistical testing."""

from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.bootstrap import write_construct_data

# Get parameters
perturbation_col = snakemake.params.perturbation_name_col
perturbation_id_col = snakemake.params.perturbation_id_col
control_key = snakemake.params.control_key
exclusion_string = snakemake.params.exclusion_string
metadata_cols_fp = snakemake.params.metadata_cols_fp

print("Loading single-cell features data...")
all_features_cells = pd.read_parquet(snakemake.input.features_singlecell)
print(f"Features data shape: {all_features_cells.shape}")

print("Loading construct and gene tables...")
construct_table = pd.read_csv(snakemake.input.construct_table, sep="\t")
gene_table = pd.read_csv(snakemake.input.gene_table, sep="\t")
print(f"Construct table shape: {construct_table.shape}")
print(f"Gene table shape: {gene_table.shape}")

# Filter for control cells (already center-scaled)
control_mask = all_features_cells[perturbation_col].str.contains(control_key, na=False)
control_cells = all_features_cells[control_mask]
print(f"Control cells for bootstrap sampling: {len(control_cells)}")

# Load metadata columns and split control cell data
metadata_cols = load_metadata_cols(metadata_cols_fp, include_classification_cols=True)
controls_metadata, controls_features = split_cell_data(control_cells, metadata_cols)

# Get available features from construct table
available_features = [
    col
    for col in construct_table.columns
    if col not in [perturbation_id_col, perturbation_col, "cell_count"]
]

# Filter control features to match available features
controls_features_selected = controls_features[available_features]

# Create controls array (individual cells for sampling)
controls_data = pd.concat(
    [controls_metadata[[perturbation_col]], controls_features_selected], axis=1
)
controls_arr = controls_data.values

# Create construct features array from construct table
construct_mask = pd.Series([True] * len(construct_table))
if exclusion_string is not None:
    construct_mask = construct_mask & ~construct_table[perturbation_col].str.contains(
        exclusion_string, na=False
    )

construct_features_df = construct_table[construct_mask]

# Create construct features array (sgRNA_ID + features)
construct_data_cols = [perturbation_id_col] + available_features
construct_features_arr = construct_features_df[construct_data_cols].values

# Create sample sizes dataframe (sgRNA_ID + cell_count)
sample_sizes_df = construct_features_df[[perturbation_id_col, "cell_count"]].copy()

print(f"Controls array shape: {controls_arr.shape}")
print(f"Construct features array shape: {construct_features_arr.shape}")
print(f"Sample sizes dataframe shape: {sample_sizes_df.shape}")

# Save bootstrap arrays
print("Saving bootstrap arrays...")
controls_df = pd.DataFrame(controls_arr)
controls_df.columns = [perturbation_col] + available_features
controls_df.to_csv(snakemake.output.controls_arr, sep="\t", index=False)

construct_features_df_export = pd.DataFrame(construct_features_arr)
construct_features_df_export.columns = [perturbation_id_col] + available_features
construct_features_df_export.to_csv(
    snakemake.output.construct_features_arr, sep="\t", index=False
)

sample_sizes_df.to_csv(snakemake.output.sample_sizes, sep="\t", index=False)

# Create checkpoint directory and construct data files
output_dir = Path(snakemake.output[0])
output_dir.mkdir(parents=True, exist_ok=True)

# Create construct data files (pseudo-gene assignments already in construct table)
construct_ids = [str(row[0]) for row in construct_features_arr if not pd.isna(row[0])]
print(f"Found {len(construct_ids)} unique constructs")

print(f"Creating {len(construct_ids)} construct data files...")
for construct_id in construct_ids:
    write_construct_data(
        construct_id,
        construct_features_df,
        perturbation_col,
        perturbation_id_col,
        output_dir,
    )

print("Bootstrap data preparation complete!")
