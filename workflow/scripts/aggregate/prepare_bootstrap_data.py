from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from lib.aggregate.bootstrap import prep_bootstrap_data, parse_gene_construct_mapping
from lib.shared.file_utils import get_filename

# Load feature table data
print("Loading feature table data...")
feature_data = pd.read_csv(snakemake.input[0], sep='\t')
print(f"Shape of input data: {feature_data.shape}")

# Get parameters
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

# Save bootstrap arrays
print("Saving bootstrap arrays...")
np.save(snakemake.output.controls_arr, controls_arr)
np.save(snakemake.output.construct_features_arr, construct_features_arr)
sample_sizes_df.to_csv(snakemake.output.sample_sizes, index=False)
np.save(snakemake.output.feature_names, np.array(selected_features))

# Create checkpoint directory and construct data files
output_dir = Path(snakemake.output[0])  # Directory output
output_dir.mkdir(parents=True, exist_ok=True)

# Extract construct IDs from the array
construct_ids = np.unique(construct_features_arr[:, 0]).tolist()
print(f"Found {len(construct_ids)} unique constructs")

# Parse gene-construct mapping
gene_construct_mapping = parse_gene_construct_mapping(construct_ids)
print(f"Found {len(gene_construct_mapping)} genes")

# Create construct data files
print(f"Saving {len(construct_ids)} construct data files to {output_dir}")
print(f"Using {multiprocessing.cpu_count()} CPUs")

def write_construct_data(construct_id):
    """Write construct data file for a single construct."""
    print(f"Processing construct: {construct_id}")
    
    # Create metadata file for the construct
    construct_data = pd.DataFrame({
        'construct_id': [construct_id],
        'gene': [construct_id.split('.')[0] if '.' in construct_id else construct_id]
    })
    
    # Save construct data file
    output_file = output_dir / get_filename(
        {"construct": construct_id}, 
        "construct_data", 
        "csv"
    )
    construct_data.to_csv(output_file, index=False)

# Process all constructs in parallel
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    executor.map(write_construct_data, construct_ids)

print(f"Bootstrap preparation complete! Created arrays and {len(construct_ids)} construct files")