import gc
import math

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import pandas as pd
import numpy as np
from pandas.api.types import is_numeric_dtype

from lib.aggregate.align import prepare_alignment_data, centerscale_on_controls
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.bootstrap import create_pseudogene_groups

# get snakemake parameters
pert_col = snakemake.params.perturbation_name_col
pert_id_col = snakemake.params.perturbation_id_col
control_key = snakemake.params.control_key
num_batches = snakemake.params.get("num_align_batches", 1)

# Load cell data using PyArrow dataset (lazy - no data loaded yet)
print("Loading cell data as PyArrow dataset...")
cell_dataset = ds.dataset(snakemake.input.filtered_paths, format="parquet")

# Determine columns
cell_data_cols = cell_dataset.schema.names
use_classifier = snakemake.params.get("use_classifier", False)
metadata_cols = load_metadata_cols(snakemake.params.metadata_cols_fp, use_classifier)
feature_cols = [col for col in cell_dataset.schema.names if col not in metadata_cols]

# Filter metadata_cols to only include columns that exist in the parquet
existing_metadata_cols = [col for col in metadata_cols if col in cell_data_cols]

print(
    f"Number of metadata columns: {len(existing_metadata_cols)} | Number of feature columns: {len(feature_cols)}"
)

# Count total rows
total_rows = cell_dataset.count_rows()
print(f"Total rows across all parquet files: {total_rows}")

# Create random indices for batched processing (like align.py)
np.random.seed(0)
all_indices = np.random.permutation(total_rows)
chunk_size = math.ceil(total_rows / num_batches)
subset_indices = [
    all_indices[i * chunk_size : (i + 1) * chunk_size] for i in range(num_batches)
]

print(f"Processing data in {num_batches} batch(es), ~{chunk_size} rows per batch")

# Initialize outputs
aligned_output = snakemake.output[0]
writer = None

# Accumulators for construct-level aggregation
# We'll collect per-construct data across batches
construct_cell_counts = {}  # {construct_id: count}
construct_gene_map = {}  # {construct_id: gene_name}
construct_feature_sums = {}  # {construct_id: [sum of features]}
construct_feature_counts = {}  # {construct_id: count for averaging}
# For median, we need all values - store them
construct_feature_values = {}  # {construct_id: list of feature arrays}

# Process each batch
for batch_idx, indices in enumerate(subset_indices):
    print(
        f"\n=== Processing batch {batch_idx + 1}/{num_batches} with {len(indices)} cells ==="
    )

    # Load only this batch using .scanner().take() (like align.py)
    indices_sorted = np.sort(indices)  # Sort for efficient reading
    batch_df = (
        cell_dataset.scanner(columns=existing_metadata_cols + feature_cols)
        .take(pa.array(indices_sorted))
        .to_pandas(use_threads=True)
    )
    print(f"Loaded batch shape: {batch_df.shape}")

    # Convert numerical columns to float32
    for col in batch_df.columns:
        if is_numeric_dtype(batch_df[col]):
            batch_df[col] = batch_df[col].astype("float32")

    # Split metadata and features
    metadata, features = split_cell_data(batch_df, metadata_cols)
    del batch_df
    gc.collect()

    # Prepare alignment data (add batch_values column)
    metadata, features = prepare_alignment_data(
        metadata,
        features,
        snakemake.params.batch_cols,
        pert_col,
        control_key,
        pert_id_col,
    )
    features = features.astype(np.float32)

    # Centerscale features on controls
    features = centerscale_on_controls(
        features,
        metadata,
        pert_col,
        control_key,
        "batch_values",
        method=snakemake.params.feature_normalization,
    ).astype(np.float32)

    # OUTPUT 1: Write center-scaled single-cell data incrementally
    print(f"Writing batch {batch_idx + 1} to parquet...")
    aligned_batch = pd.concat(
        [metadata.reset_index(drop=True), pd.DataFrame(features, columns=feature_cols)],
        axis=1,
    )

    # Write using PyArrow for incremental output (like align.py)
    aligned_table = pa.Table.from_pandas(aligned_batch, preserve_index=False)
    if writer is None:
        writer = pq.ParquetWriter(aligned_output, aligned_table.schema)
    writer.write_table(aligned_table)

    del aligned_table, aligned_batch
    gc.collect()

    # Accumulate construct-level data for median computation
    print(f"Accumulating construct statistics for batch {batch_idx + 1}...")
    for construct_id in metadata[pert_id_col].unique():
        mask = metadata[pert_id_col].values == construct_id
        construct_features = features[mask]
        gene_name = metadata.loc[mask, pert_col].iloc[0]

        if construct_id not in construct_cell_counts:
            construct_cell_counts[construct_id] = 0
            construct_gene_map[construct_id] = gene_name
            construct_feature_values[construct_id] = []

        construct_cell_counts[construct_id] += mask.sum()
        construct_feature_values[construct_id].append(construct_features)

    # Clean up batch data
    del metadata, features
    gc.collect()

# Close the parquet writer
if writer is not None:
    writer.close()
print(f"\nSaved aligned cell data to: {aligned_output}")

# TABLE 1: Construct-level table (one row per sgRNA)
print("\n=== Creating construct-level table ===")

construct_rows = []
for construct_id in construct_cell_counts.keys():
    # Concatenate all feature arrays for this construct
    all_features = np.vstack(construct_feature_values[construct_id])
    # Compute median across all cells
    median_features = np.median(all_features, axis=0)

    row = {
        pert_id_col: construct_id,
        pert_col: construct_gene_map[construct_id],
        "cell_count": construct_cell_counts[construct_id],
    }
    for i, col in enumerate(feature_cols):
        row[col] = median_features[i]
    construct_rows.append(row)

# Free memory from accumulated features
del construct_feature_values
gc.collect()

construct_table = pd.DataFrame(construct_rows)

# Reorder columns: sgRNA, gene, cell_count, features
construct_columns = [pert_id_col, pert_col, "cell_count"] + feature_cols
construct_table = construct_table[construct_columns]

print(f"Construct table shape: {construct_table.shape}")

# TABLE 2: Gene-level table (median of construct medians)
print("\n=== Creating gene-level table ===")

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

print(f"\nSaved gene table to: {gene_output}")
print(f"Saved construct table to: {construct_output}")
