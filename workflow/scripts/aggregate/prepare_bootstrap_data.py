from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from lib.aggregate.bootstrap import prep_bootstrap_data, parse_gene_construct_mapping
from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.shared.file_utils import get_filename

# Load and combine all filtered cell data
print("Loading and combining filtered cell data...")
combined_data = []

for input_file in snakemake.input:
    print(f"Loading {input_file}")
    df = pd.read_parquet(input_file)
    combined_data.append(df)

# Combine all dataframes
cell_data = pd.concat(combined_data, ignore_index=True)
print(f"Combined cell data shape: {cell_data.shape}")

# Get parameters
perturbation_col = snakemake.params.perturbation_name_col
control_key = snakemake.params.control_key
exclusion_string = snakemake.params.exclusion_string

# Load metadata columns and split cell data
metadata_cols = load_metadata_cols(
    "/path/to/metadata_cols_fp",  # You'll need to pass this as a parameter
    include_classification_cols=True
)

# Split into metadata and features
metadata, features = split_cell_data(cell_data, metadata_cols)

# For bootstrap, we need to count cells per perturbation to get sample sizes
sample_sizes_df = metadata.groupby(perturbation_col).size().reset_index(name='cell_count')

# Now we need to create construct-level data for bootstrap
# Get unique perturbations and their median features
perturbation_features = []
for perturbation in metadata[perturbation_col].unique():
    if pd.isna(perturbation):
        continue
        
    pert_mask = metadata[perturbation_col] == perturbation
    pert_features = features[pert_mask].median().values  # Use median like generate_feature_table
    
    # Combine perturbation ID with its median features
    pert_row = np.concatenate([[perturbation], pert_features])
    perturbation_features.append(pert_row)

# Convert to array for bootstrap functions
construct_features_arr = np.array(perturbation_features, dtype=object)

# Get controls data (individual cells from control perturbations)
controls_mask = metadata[perturbation_col].str.contains(control_key, na=False)
controls_metadata = metadata[controls_mask]
controls_features = features[controls_mask]

# Combine control metadata and features for bootstrap sampling
controls_data = pd.concat([controls_metadata[[perturbation_col]], controls_features], axis=1)
controls_arr = controls_data.values

# Get selected feature names (same logic as bootstrap feature selection)
from lib.aggregate.bootstrap import select_bootstrap_features
selected_features = select_bootstrap_features(features.columns.tolist())

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
construct_ids = [str(row[0]) for row in construct_features_arr if not pd.isna(row[0])]
# Filter out controls
construct_ids = [cid for cid in construct_ids if control_key not in cid]
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