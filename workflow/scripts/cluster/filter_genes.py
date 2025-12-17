"""Filter genes based on pre-computed significant genes list.

This script:
1. Loads a pre-computed significant genes list (from notebook 9)
2. Loads aggregated data (with PCA features for clustering)
3. Filters aggregated data to only include significant genes
4. Outputs filtered aggregated data for downstream clustering
"""

import pandas as pd

# Load significant genes list
sig_genes_df = pd.read_csv(snakemake.input.significant_genes, sep="\t")
significant_genes = sig_genes_df["gene"].tolist()

# Load aggregated data (with PCA features - this is what we actually cluster)
# Use [0] since input is a list with one element
aggregated_df = pd.read_csv(snakemake.input.aggregated[0], sep="\t")

# Get gene column name
gene_col = snakemake.params.perturbation_name_col

print(f"Significant genes from file: {len(significant_genes)}")
print(f"Aggregated data: {len(aggregated_df)} genes")

# Filter aggregated data by gene list
filtered_aggregated = aggregated_df[aggregated_df[gene_col].isin(significant_genes)].copy()

print(f"Filtered aggregated data: {len(filtered_aggregated)} genes")

# Verify we have PCA columns
if "PC_0" in filtered_aggregated.columns:
    print("✓ Filtered data contains PC_0 column (ready for clustering)")
else:
    print("⚠ WARNING: Filtered data does not contain PC_0 column!")

# Save filtered aggregated table
filtered_aggregated.to_csv(snakemake.output[0], sep="\t", index=False)
