"""Combine bootstrap results from all constructs and genes."""

import pandas as pd
import glob

# Get directory parameters
constructs_dir = snakemake.params.constructs_dir
genes_dir = snakemake.params.genes_dir

print("Combining all bootstrap results...")

# Combine construct results
print("Loading construct results...")
construct_pval_files = glob.glob(f"{constructs_dir}/*__pvals.tsv")
print(f"Found {len(construct_pval_files)} construct result files")

if not construct_pval_files:
    raise ValueError(f"No construct p-value files found in {constructs_dir}")

construct_results = []
for pval_file in construct_pval_files:
    df = pd.read_csv(pval_file, sep="\t")
    construct_results.append(df)

construct_df = pd.concat(construct_results, ignore_index=True)
construct_df = construct_df.sort_values(["gene", "construct"])

# Combine gene results
print("Loading gene results...")
gene_pval_files = glob.glob(f"{genes_dir}/*__pvals.tsv")
print(f"Found {len(gene_pval_files)} gene result files")

if not gene_pval_files:
    raise ValueError(f"No gene p-value files found in {genes_dir}")

gene_results = []
for pval_file in gene_pval_files:
    df = pd.read_csv(pval_file, sep="\t")
    gene_results.append(df)

gene_df = pd.concat(gene_results, ignore_index=True)
gene_df = gene_df.sort_values("gene")

# Save combined results
print("Saving combined results...")
construct_df.to_csv(snakemake.output[0], sep="\t", index=False)
gene_df.to_csv(snakemake.output[1], sep="\t", index=False)

print(f"Saved combined construct results: {snakemake.output[0]}")
print(f"Saved combined gene results: {snakemake.output[1]}")
print("Bootstrap combination complete!")
