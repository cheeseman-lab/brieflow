import numpy as np
import pandas as pd
import glob
from pathlib import Path

from lib.aggregate.bootstrap import (
    load_construct_null_arrays,
    calculate_pvals,
)

# Get gene ID from wildcards
gene_id = snakemake.wildcards.gene
cell_class = snakemake.wildcards.cell_class
channel_combo = snakemake.wildcards.channel_combo

print(f"Running gene-level bootstrap aggregation for: {gene_id}")

# Build the pattern using the template from params
construct_nulls_pattern = snakemake.params.construct_nulls_pattern
# Replace {construct} with * for glob pattern
pattern = construct_nulls_pattern.replace("{construct}", "*")

print(f"Looking for construct null files with pattern: {pattern}")
construct_null_paths = glob.glob(pattern)

if not construct_null_paths:
    print(f"No construct null files found for gene {gene_id} with pattern: {pattern}")
    
    # Check what files do exist in the constructs directory
    constructs_dir = f"brieflow_output/aggregate/bootstrap/{cell_class}__{channel_combo}__constructs"
    if Path(constructs_dir).exists():
        all_files = glob.glob(f"{constructs_dir}/*_nulls.npy")
        print(f"Available null files: {[Path(f).name for f in all_files[:10]]}")
    else:
        print(f"Constructs directory does not exist: {constructs_dir}")
    
    # Create empty output files - this gene has no constructs
    np.save(snakemake.output[0], np.array([]))
    pd.DataFrame({"gene": [gene_id], "error": ["No construct files found"]}).to_csv(
        snakemake.output[1], sep="\t", index=False
    )
    print("Created empty output files for gene with no constructs")
    exit(0)

print(f"Found {len(construct_null_paths)} construct null files for gene {gene_id}")
print(f"Files: {[Path(f).name for f in construct_null_paths]}")

# Load the arrays using the correct function
construct_null_arrays = load_construct_null_arrays(construct_null_paths)

# Load gene table to get observed gene median
gene_table = pd.read_csv(snakemake.input.gene_table, sep="\t")
gene_row = gene_table[gene_table["gene_symbol_0"] == gene_id]

if len(gene_row) == 0:
    raise ValueError(f"Gene {gene_id} not found in gene table")

# Get feature names (excluding metadata columns)
feature_names = [
    col for col in gene_table.columns if col not in ["gene_symbol_0", "cell_count"]
]

# Get observed gene medians (from gene table)
observed_gene_medians = gene_row[feature_names].values[0].astype(float)

print(f"Number of features: {len(feature_names)}")
print(f"Construct null array shapes: {[arr.shape for arr in construct_null_arrays]}")

# Get parameters
num_sims = snakemake.params.num_sims
total_cells = int(gene_row["cell_count"].iloc[0])

# Aggregate construct null distributions to gene level
print("Aggregating construct bootstrap results to gene level...")

# Stack all construct null distributions
stacked_nulls = np.stack(
    construct_null_arrays
)  # Shape: (num_constructs, num_sims, num_features)

# Take median across constructs for each simulation
gene_null_medians = np.median(stacked_nulls, axis=0)  # Shape: (num_sims, num_features)

# Calculate p-values
p_vals = calculate_pvals(gene_null_medians, observed_gene_medians)

print(f"Gene-level aggregation complete!")
print(f"Aggregated {len(construct_null_arrays)} constructs")

# Format results
pval_df = pd.DataFrame(
    {
        "gene": [gene_id],
        "num_constructs": [len(construct_null_arrays)],
        "total_cells": [total_cells],
        "num_sims": [num_sims],
        **{feature: [pval] for feature, pval in zip(feature_names, p_vals)},
    }
)

# Save outputs
print("Saving gene-level bootstrap results...")

# Save aggregated null distribution
np.save(snakemake.output[0], gene_null_medians)

# Save p-values as TSV
pval_df.to_csv(snakemake.output[1], sep="\t", index=False)

print(f"Gene-level bootstrap analysis for {gene_id} complete!")