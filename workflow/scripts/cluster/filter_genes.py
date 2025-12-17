"""Filter genes based on bootstrap statistics and subset aggregated data.

This script:
1. Loads merged bootstrap+features_genes data to identify which genes pass filter
2. Loads aggregated data (with PCA features for clustering)
3. Filters aggregated data to only include genes that pass bootstrap thresholds + controls
4. Outputs filtered aggregated data for downstream clustering
"""

import pandas as pd
from lib.cluster.cluster_eval import get_filtered_gene_list

# Load merged bootstrap data (for determining which genes pass filter)
merged_df = pd.read_csv(snakemake.input.merged_bootstrap, sep="\t")

# Load aggregated data (with PCA features - this is what we actually cluster)
aggregated_df = pd.read_csv(snakemake.input.aggregated, sep="\t")

# Get parameters
gene_col = snakemake.params.perturbation_name_col
control_patterns = snakemake.params.control_patterns
zscore_threshold = snakemake.params.zscore_threshold
zscore_direction = snakemake.params.zscore_direction
fdr_threshold = snakemake.params.fdr_threshold
filter_mode = snakemake.params.filter_mode

print(f"Filtering genes with mode: {filter_mode}")
print(f"  Z-score threshold: {zscore_threshold} (direction: {zscore_direction})")
print(f"  FDR threshold: {fdr_threshold}")
print(f"  Control patterns: {control_patterns}")
print(f"Merged bootstrap data: {len(merged_df)} genes")
print(f"Aggregated data: {len(aggregated_df)} genes")

# Get list of genes that pass filter using library function
filtered_gene_list = get_filtered_gene_list(
    merged_data=merged_df,
    perturbation_name_col=gene_col,
    control_patterns=control_patterns,
    zscore_threshold=zscore_threshold,
    zscore_direction=zscore_direction,
    fdr_threshold=fdr_threshold,
    filter_mode=filter_mode,
)

print(f"Genes passing filter: {len(filtered_gene_list)}")

# Filter the AGGREGATED data by gene list
filtered_aggregated = aggregated_df[
    aggregated_df[gene_col].isin(filtered_gene_list)
].copy()

print(f"Filtered aggregated data: {len(filtered_aggregated)} genes")

# Verify we have PCA columns
if "PC_0" in filtered_aggregated.columns:
    print("✓ Filtered data contains PC_0 column (ready for clustering)")
else:
    print("⚠ WARNING: Filtered data does not contain PC_0 column!")

# Save filtered aggregated table
filtered_aggregated.to_csv(snakemake.output[0], sep="\t", index=False)
