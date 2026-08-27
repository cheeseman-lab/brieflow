"""Bootstrap statistical testing for individual constructs."""

import pandas as pd
import numpy as np

from lib.aggregate.bootstrap import run_construct_bootstrap
from lib.aggregate.cell_data_utils import GROUP_KEY_SEP

# Load construct data to get construct ID and gene
construct_data = pd.read_csv(snakemake.input.construct_data, sep="\t")
construct_id = construct_data["construct_id"].iloc[0]
gene = construct_data["gene"].iloc[0]
print(f"Running bootstrap analysis for construct: {construct_id} (gene: {gene})")

# Load bootstrap input arrays
print("Loading bootstrap input arrays...")
controls_df = pd.read_csv(snakemake.input.controls_arr, sep="\t")

# Restrict the null pool to controls sharing this construct's group key
control_scope = snakemake.params.get("bootstrap_control_scope", "pooled")
if control_scope not in ("pooled", "within_group"):
    raise ValueError(f"Unknown bootstrap_control_scope: {control_scope}")
if control_scope == "within_group" and GROUP_KEY_SEP in str(construct_id):
    group_key = str(construct_id).split(GROUP_KEY_SEP, 1)[1]
    group_mask = (
        controls_df.iloc[:, 0].astype(str).str.split(GROUP_KEY_SEP, n=1).str[1]
        == group_key
    )
    print(
        f"Restricting controls to group '{group_key}': {int(group_mask.sum())} of {len(controls_df)} rows"
    )
    controls_df = controls_df[group_mask]
    if len(controls_df) == 0:
        raise ValueError(
            f"No control cells found for group '{group_key}' (construct {construct_id})"
        )

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
