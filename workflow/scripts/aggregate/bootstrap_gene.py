import numpy as np
import pandas as pd

from lib.aggregate.bootstrap import (
    load_construct_null_arrays,
    calculate_pvals,
    format_gene_results
)

# Get gene ID from wildcards
gene_id = snakemake.wildcards.gene
print(f"Running gene-level bootstrap aggregation for: {gene_id}")

# Load construct null arrays for this gene
print("Loading construct null distributions...")
construct_null_paths = snakemake.input.construct_nulls
construct_null_arrays = load_construct_null_arrays(construct_null_paths)
print(f"Loaded {len(construct_null_arrays)} construct null distributions")

# Load gene table to get observed gene median
gene_table = pd.read_csv(snakemake.input.gene_table, sep='\t')
gene_row = gene_table[gene_table['gene_symbol_0'] == gene_id]

if len(gene_row) == 0:
    raise ValueError(f"Gene {gene_id} not found in gene table")

# Get feature names (excluding metadata columns)
feature_names = [col for col in gene_table.columns 
                if col not in ['gene_symbol_0', 'cell_count']]

# Get observed gene medians (from gene table)
observed_gene_medians = gene_row[feature_names].values[0].astype(float)

print(f"Construct null array shapes: {[arr.shape for arr in construct_null_arrays]}")
print(f"Observed gene medians shape: {observed_gene_medians.shape}")
print(f"Number of features: {len(feature_names)}")

# Get parameters
num_sims = snakemake.params.num_sims
total_cells = int(gene_row['cell_count'].iloc[0])

# Aggregate construct null distributions to gene level
print("Aggregating construct bootstrap results to gene level...")

# Stack all construct null distributions
stacked_nulls = np.stack(construct_null_arrays)  # Shape: (num_constructs, num_sims, num_features)

# Take median across constructs for each simulation
gene_null_medians = np.median(stacked_nulls, axis=0)  # Shape: (num_sims, num_features)

print(f"Gene null distribution shape: {gene_null_medians.shape}")

# Calculate p-values
p_vals = calculate_pvals(gene_null_medians, observed_gene_medians)

print(f"Gene-level aggregation complete!")
print(f"Aggregated {len(construct_null_arrays)} constructs")
print(f"P-values shape: {p_vals.shape}")

# Format results
pval_df = format_gene_results(
    gene_id,
    p_vals,
    feature_names,
    len(construct_null_arrays),  # num_constructs
    num_sims
)

# Add total cell count
pval_df['total_cells'] = total_cells

# Reorder columns
column_order = ['gene', 'num_constructs', 'total_cells', 'num_sims'] + feature_names
pval_df = pval_df[column_order]

# Save outputs
print("Saving gene-level bootstrap results...")

# Save aggregated null distribution
np.save(snakemake.output[0], gene_null_medians)

# Save p-values as TSV
pval_df.to_csv(snakemake.output[1], sep='\t', index=False)

print(f"Gene-level bootstrap analysis for {gene_id} complete!")