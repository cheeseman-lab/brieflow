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
from lib.aggregate.bootstrap import create_pseudogene_groups

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

print(
    f"Number of metadata columns: {len(metadata_cols)} | Number of feature columns: {len(feature_cols)}"
)

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
    method=snakemake.params.feature_normalization,
).astype(np.float32)

# OUTPUT 1: Save center-scaled single-cell data for bootstrap
print("Saving center-scaled single-cell data for bootstrap...")
aligned_cell_data = pd.concat(
    [metadata, pd.DataFrame(features, columns=feature_cols)], axis=1
)
aligned_output = snakemake.output[0]
aligned_cell_data.to_parquet(aligned_output, index=False)
print(f"Saved aligned cell data to: {aligned_output}")

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
features_df = pd.DataFrame(features, columns=feature_cols)
features_df[pert_id_col] = metadata[pert_id_col].values

construct_features = features_df.groupby(
    pert_id_col, sort=False, observed=True
).median()
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

# Add pseudo-gene entries if specified
pseudogene_patterns = snakemake.params.get("pseudogene_patterns", None)

if pseudogene_patterns:
    print("Adding pseudo-gene entries to gene table...")

    # Import the pseudo-gene grouping function
    from lib.aggregate.bootstrap import create_pseudogene_groups

    # Create pseudo-gene groups from construct table
    pseudogene_groups, remaining_constructs = create_pseudogene_groups(
        construct_table, pseudogene_patterns, pert_col, seed=42
    )

    pseudogene_rows = []

    for pseudogene_group in pseudogene_groups:
        pseudogene_id = pseudogene_group["pseudogene_id"]
        constructs = pseudogene_group["constructs"]

        print(f"  Creating gene table entry for: {pseudogene_id}")

        # Get construct IDs from this pseudo-gene group
        construct_ids_in_group = [c[pert_id_col] for c in constructs]

        # Find matching rows in construct_table
        group_mask = construct_table[pert_id_col].isin(construct_ids_in_group)
        group_constructs = construct_table[group_mask]

        if len(group_constructs) == 0:
            print(f"    Warning: No matching constructs found for {pseudogene_id}")
            continue

        # Aggregate features (median across constructs)
        feature_medians = group_constructs[feature_cols].median()

        # Sum cell counts
        total_cells = group_constructs["cell_count"].sum()

        # Create pseudo-gene row
        pseudogene_row = {pert_col: pseudogene_id, "cell_count": total_cells}
        pseudogene_row.update(feature_medians.to_dict())

        pseudogene_rows.append(pseudogene_row)

        print(
            f"    Aggregated {len(group_constructs)} constructs, {total_cells} total cells"
        )

    if pseudogene_rows:
        # Create DataFrame from pseudo-gene rows
        pseudogene_df = pd.DataFrame(pseudogene_rows)

        # Ensure column order matches final_gene_table
        pseudogene_df = pseudogene_df[gene_columns]

        # Append to final gene table
        final_gene_table = pd.concat(
            [final_gene_table, pseudogene_df], ignore_index=True
        )

        print(f"Added {len(pseudogene_rows)} pseudo-gene entries to gene table")
        print(f"Final gene table shape: {final_gene_table.shape}")

    # Also modify construct table - change gene names for pseudo-gene constructs
    print("Modifying construct table with pseudo-gene assignments...")

    # Create pseudo-gene construct entries
    pseudogene_construct_rows = []
    original_construct_ids_to_remove = []

    for pseudogene_group in pseudogene_groups:
        pseudogene_id = pseudogene_group["pseudogene_id"]
        for construct in pseudogene_group["constructs"]:
            construct_id = construct[pert_id_col]

            # Find the original construct in construct_table
            construct_mask = construct_table[pert_id_col] == construct_id
            original_construct = construct_table[construct_mask]

            if len(original_construct) > 0:
                # Create new row with pseudo-gene as the "gene"
                new_row = original_construct.iloc[0].copy()
                new_row[pert_col] = pseudogene_id  # Change gene to pseudo-gene name
                pseudogene_construct_rows.append(new_row)
                original_construct_ids_to_remove.append(construct_id)

    if pseudogene_construct_rows:
        # Remove original constructs that are now part of pseudo-genes
        construct_table = construct_table[
            ~construct_table[pert_id_col].isin(original_construct_ids_to_remove)
        ]

        # Add pseudo-gene constructs
        pseudogene_construct_df = pd.DataFrame(pseudogene_construct_rows)
        construct_table = pd.concat(
            [construct_table, pseudogene_construct_df], ignore_index=True
        )

        print(
            f"Modified construct table with {len(pseudogene_construct_rows)} pseudo-gene construct entries"
        )
        print(
            f"Removed {len(original_construct_ids_to_remove)} original construct entries"
        )
        print(f"Final construct table shape: {construct_table.shape}")

# OUTPUT 2 & 3: Save both tables
construct_output = snakemake.output[1]
gene_output = snakemake.output[2]

construct_table.to_csv(construct_output, sep="\t", index=False)
final_gene_table.to_csv(gene_output, sep="\t", index=False)

print(f"Saved gene table to: {gene_output}")
print(f"Saved construct table to: {construct_output}")
