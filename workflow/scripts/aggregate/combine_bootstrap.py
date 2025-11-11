"""Combine bootstrap results from all constructs and genes."""

import pandas as pd
import numpy as np
import glob
from lib.aggregate.bootstrap import apply_multiple_hypothesis_correction

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

# Get feature columns (exclude metadata columns)
construct_metadata_cols = ["gene", "construct", "sample_size", "num_sims"]
construct_feature_cols = [
    col for col in construct_df.columns if col not in construct_metadata_cols
]

print(f"Found {len(construct_feature_cols)} feature columns for construct analysis")

# Apply MHT correction to construct results
construct_df_corrected = apply_multiple_hypothesis_correction(
    construct_df, construct_feature_cols
)

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

# Get feature columns for gene analysis
gene_metadata_cols = ["gene", "num_constructs", "total_cells", "num_sims"]
gene_feature_cols = [col for col in gene_df.columns if col not in gene_metadata_cols]

print(f"Found {len(gene_feature_cols)} feature columns for gene analysis")

# Apply MHT correction to gene results
gene_df_corrected = apply_multiple_hypothesis_correction(gene_df, gene_feature_cols)

# Save combined results
construct_df_corrected.to_csv(snakemake.output[0], sep="\t", index=False)
gene_df_corrected.to_csv(snakemake.output[1], sep="\t", index=False)

print("Bootstrap combination complete!")
