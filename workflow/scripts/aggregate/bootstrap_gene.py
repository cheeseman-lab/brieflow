"""Aggregate construct-level bootstrap results to gene level."""

import numpy as np
import pandas as pd
import glob
from pathlib import Path

from lib.aggregate.bootstrap import load_construct_null_arrays, calculate_pvals

# Get parameters from wildcards
gene_id = snakemake.wildcards.gene
cell_class = snakemake.wildcards.cell_class
channel_combo = snakemake.wildcards.channel_combo
num_sims = snakemake.params.num_sims

print(f"Running gene-level bootstrap aggregation for: {gene_id}")

# Build pattern for construct null files
construct_nulls_pattern = snakemake.params.construct_nulls_pattern
pattern = construct_nulls_pattern.replace("{construct}", "*")

print(f"Looking for construct null files with pattern: {pattern}")
construct_null_paths = glob.glob(pattern)

if not construct_null_paths:
    print(f"No construct null files found for gene {gene_id}")

    # Create empty output files for genes with no constructs
    np.save(snakemake.output[0], np.array([]))
    pd.DataFrame({"gene": [gene_id], "error": ["No construct files found"]}).to_csv(
        snakemake.output[1], sep="\t", index=False
    )
    print("Created empty output files for gene with no constructs")
    exit(0)

print(f"Found {len(construct_null_paths)} construct null files for gene {gene_id}")

# Load construct null distributions
construct_null_arrays = load_construct_null_arrays(construct_null_paths)

# Load gene table to get observed gene values
gene_table = pd.read_csv(snakemake.input.gene_table, sep="\t")
gene_row = gene_table[gene_table["gene_symbol_0"] == gene_id]

if len(gene_row) == 0:
    print(f"Gene {gene_id} not found in gene table")
    print(f"Available genes: {sorted(gene_table['gene_symbol_0'].unique())}")
    raise ValueError(f"Gene {gene_id} not found in gene table")

# Get feature names (excluding metadata columns)
feature_names = [
    col for col in gene_table.columns if col not in ["gene_symbol_0", "cell_count"]
]

# Get observed gene medians
observed_gene_medians = gene_row[feature_names].values[0].astype(float)
total_cells = int(gene_row["cell_count"].iloc[0])

print(f"Number of features: {len(feature_names)}")
print(f"Construct null array shapes: {[arr.shape for arr in construct_null_arrays]}")

# Aggregate construct null distributions to gene level
print("Aggregating construct bootstrap results to gene level...")

# Stack all construct null distributions and take median across constructs
stacked_nulls = np.stack(
    construct_null_arrays
)  # (num_constructs, num_sims, num_features)
gene_null_medians = np.median(stacked_nulls, axis=0)  # (num_sims, num_features)

# Calculate p-values
p_vals = calculate_pvals(gene_null_medians, observed_gene_medians)

print(
    f"Gene-level aggregation complete! Aggregated {len(construct_null_arrays)} constructs"
)

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
np.save(snakemake.output[0], gene_null_medians)
pval_df.to_csv(snakemake.output[1], sep="\t", index=False)

print(f"Gene-level bootstrap analysis for {gene_id} complete!")
