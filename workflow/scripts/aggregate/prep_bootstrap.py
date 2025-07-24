import pandas as pd
import numpy as np

from lib.aggregate.bootstrap import prep_bootstrap_data, select_bootstrap_features

# Load feature table data
print("Loading feature table data...")
feature_data = pd.read_csv(snakemake.input[0], sep='\t')
print(f"Shape of input data: {feature_data.shape}")

# Get column info
perturbation_col = snakemake.params.perturbation_name_col
control_key = snakemake.params.control_key
sample_size_col = snakemake.params.sample_size_column
exclusion_string = snakemake.params.exclusion_string

# Identify metadata vs feature columns
metadata_cols = [perturbation_col]
if sample_size_col != perturbation_col:
    metadata_cols.append(sample_size_col)

# Prepare bootstrap data
print("Preparing bootstrap arrays...")
controls_arr, construct_features_arr, sample_sizes_df, selected_features = prep_bootstrap_data(
    feature_data,
    metadata_cols,
    perturbation_col,
    control_key,
    sample_size_col,
    exclusion_string
)

print(f"Controls array shape: {controls_arr.shape}")
print(f"Construct features array shape: {construct_features_arr.shape}")
print(f"Selected {len(selected_features)} features for bootstrap analysis")
print(f"Sample sizes for {len(sample_sizes_df)} constructs")

# Save outputs
print("Saving bootstrap preparation files...")

# Save controls array
np.save(snakemake.output[0], controls_arr)

# Save construct features array  
np.save(snakemake.output[1], construct_features_arr)

# Save sample sizes
sample_sizes_df.to_csv(snakemake.output[2], index=False)

# Save feature names
np.save(snakemake.output[3], np.array(selected_features))

print("Bootstrap preparation complete!")