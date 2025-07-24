import pandas as pd
import numpy as np

from lib.aggregate.bootstrap import (
    run_construct_bootstrap,
    format_construct_results,
    get_sample_size
)

# Load construct data to get construct ID
construct_data = pd.read_csv(snakemake.input.construct_data)
construct_id = construct_data['construct_id'].iloc[0]
print(f"Running bootstrap analysis for construct: {construct_id}")

# Load bootstrap arrays
print("Loading bootstrap input arrays...")
controls_arr = np.load(snakemake.input.controls_arr, allow_pickle=True)
construct_features_arr = np.load(snakemake.input.construct_features_arr, allow_pickle=True)
sample_sizes_df = pd.read_csv(snakemake.input.sample_sizes)
feature_names = np.load(snakemake.input.feature_names, allow_pickle=True).tolist()

print(f"Controls array shape: {controls_arr.shape}")
print(f"Construct features array shape: {construct_features_arr.shape}")
print(f"Number of features: {len(feature_names)}")

# Get parameters
sample_size_col = snakemake.params.sample_size_column
num_sims = snakemake.params.num_sims

# Get sample size for this construct
sample_size = get_sample_size(sample_sizes_df, construct_id, sample_size_col)
print(f"Sample size for {construct_id}: {sample_size}")

# Run bootstrap analysis
print(f"Running {num_sims} bootstrap simulations...")
null_medians_arr, p_vals = run_construct_bootstrap(
    construct_id,
    construct_features_arr,
    controls_arr,
    sample_size,
    num_sims
)

print(f"Bootstrap analysis complete!")
print(f"Null distribution shape: {null_medians_arr.shape}")
print(f"P-values shape: {p_vals.shape}")

# Format results
pval_df = format_construct_results(
    construct_id,
    p_vals,
    feature_names,
    sample_size,
    num_sims
)

# Save outputs
print("Saving bootstrap results...")

# Save null distribution
np.save(snakemake.output[0], null_medians_arr)

# Save p-values
pval_df.to_csv(snakemake.output[1], index=False)

print(f"Bootstrap analysis for {construct_id} complete!")