import numpy as np
import pandas as pd

from lib.aggregate.bootstrap import (
    load_construct_null_arrays,
    calculate_pvals,
)

# Get gene ID from wildcards
gene_id = snakemake.wildcards.gene
print(f"Running gene-level bootstrap aggregation for: {gene_id}")

# Load construct null arrays for this gene
print("Loading construct null distributions...")
construct_null_paths = snakemake.input.construct_nulls
print(f"Type: {type(construct_null_paths)}")
print(f"Value: {construct_null_paths}")
print(f"Length: {len(construct_null_paths)}")
if hasattr(construct_null_paths, '__iter__'):
    print(f"First few items: {list(construct_null_paths)[:3]}")

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
