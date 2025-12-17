"""Merge bootstrap results with gene-level feature table."""

import pandas as pd
from lib.cluster.cluster_eval import merge_bootstrap_with_genes

# Load bootstrap results (contains pval, log10, fdr columns)
bootstrap_df = pd.read_csv(snakemake.input.bootstrap, sep="\t")

# Load gene features table
genes_df = pd.read_csv(snakemake.input.genes, sep="\t")

# Get the gene column name from config
gene_col = snakemake.params.perturbation_name_col

# Merge using library function
merged = merge_bootstrap_with_genes(
    bootstrap_df=bootstrap_df,
    genes_df=genes_df,
    perturbation_name_col=gene_col,
    bootstrap_gene_col="gene",
)

# Save merged table
merged.to_csv(snakemake.output[0], sep="\t", index=False)

print(f"Merged {len(genes_df)} genes with bootstrap results")
print(f"Output shape: {merged.shape}")
