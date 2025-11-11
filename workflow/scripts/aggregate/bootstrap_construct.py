"""Bootstrap statistical testing for individual constructs."""

import pandas as pd
import numpy as np

from lib.aggregate.bootstrap import run_construct_bootstrap

# Load construct data to get construct ID and gene
construct_data = pd.read_csv(snakemake.input.construct_data, sep="\t")
construct_id = construct_data["construct_id"].iloc[0]
gene = construct_data["gene"].iloc[0]
print(f"Running bootstrap analysis for construct: {construct_id} (gene: {gene})")

# Load bootstrap input arrays
print("Loading bootstrap input arrays...")
controls_df = pd.read_csv(snakemake.input.controls_arr, sep="\t")
controls_arr = controls_df.values

construct_features_df = pd.read_csv(snakemake.input.construct_features_arr, sep="\t")
construct_features_arr = construct_features_df.values

sample_sizes_df = pd.read_csv(snakemake.input.sample_sizes, sep="\t")

# Get feature names (excluding first column which is construct ID)
feature_names = construct_features_df.columns[1:].tolist()

print(f"Controls array shape: {controls_arr.shape}")
print(f"Construct features array shape: {construct_features_arr.shape}")
print(f"Number of features: {len(feature_names)}")

# Get parameters
num_sims = snakemake.params.num_sims

# Get sample size for this construct
construct_mask = sample_sizes_df.iloc[:, 0] == construct_id
if not construct_mask.any():
    raise ValueError(f"Construct {construct_id} not found in sample sizes")
sample_size = int(sample_sizes_df.loc[construct_mask, "cell_count"].iloc[0])
print(f"Sample size for {construct_id}: {sample_size}")

# Run bootstrap analysis
print(f"Running {num_sims} bootstrap simulations...")
null_medians_arr, p_vals = run_construct_bootstrap(
    construct_id, construct_features_arr, controls_arr, sample_size, num_sims
)

print(f"Bootstrap analysis complete!")
print(f"Null distribution shape: {null_medians_arr.shape}")
print(f"P-values shape: {p_vals.shape}")

# Format results
pval_df = pd.DataFrame(
    {
        "gene": [gene],
        "construct": [construct_id],
        "sample_size": [sample_size],
        "num_sims": [num_sims],
        **{feature: [pval] for feature, pval in zip(feature_names, p_vals)},
    }
)

# Reorder columns to put metadata first
column_order = ["gene", "construct", "sample_size", "num_sims"] + feature_names
pval_df = pval_df[column_order]

# Save outputs
print("Saving bootstrap results...")
np.save(snakemake.output[0], null_medians_arr)
pval_df.to_csv(snakemake.output[1], sep="\t", index=False)

print(f"Bootstrap analysis for {construct_id} complete!")
