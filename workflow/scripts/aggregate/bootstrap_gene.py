import numpy as np
import pandas as pd

from lib.aggregate.bootstrap import (
    load_construct_null_arrays,
    aggregate_gene_results,
    format_gene_results
)

# Get gene ID from wildcards
gene_id = snakemake.wildcards.gene
print(f"Running gene-level bootstrap aggregation for: {gene_id}")

# Load construct null arrays
print("Loading construct null distributions...")
construct_null_paths = snakemake.input.construct_nulls
construct_null_arrays = load_construct_null_arrays(construct_null_paths)
print(f"Loaded {len(construct_null_arrays)} construct null distributions")

# Load gene features array
construct_features_arr = np.load(snakemake.input.construct_features_arr, allow_pickle=True)

# Get feature names from construct_features_arr (excluding first column which is construct ID)
feature_names = [f"feature_{i}" for i in range(construct_features_arr.shape[1] - 1)]

print(f"Gene features array shape: {construct_features_arr.shape}")
print(f"Number of features: {len(feature_names)}")

# Get parameters
num_sims = snakemake.params.num_sims

# Aggregate construct results to gene level
print("Aggregating construct bootstrap results to gene level...")
median_null_medians, p_vals, num_constructs = aggregate_gene_results(
    gene_id,
    construct_null_arrays,
    construct_features_arr
)

print(f"Gene-level aggregation complete!")
print(f"Aggregated {num_constructs} constructs")
print(f"Gene null distribution shape: {median_null_medians.shape}")
print(f"Gene p-values shape: {p_vals.shape}")

# Format results
pval_df = format_gene_results(
    gene_id,
    p_vals,
    feature_names,
    num_constructs,
    num_sims
)

# Save outputs
print("Saving gene-level bootstrap results...")

# Save null distribution
np.save(snakemake.output[0], median_null_medians)

# Save p-values
pval_df.to_csv(snakemake.output[1], sep='\t', index=False)

print(f"Gene-level bootstrap analysis for {gene_id} complete!")