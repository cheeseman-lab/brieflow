from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.shared.file_utils import get_filename

# Get parameters
perturbation_col = snakemake.params.perturbation_name_col
control_key = snakemake.params.control_key
exclusion_string = snakemake.params.exclusion_string
metadata_cols_fp = snakemake.params.metadata_cols_fp

print("Loading feature table with perturbation-level medians...")
# The first input should be the feature table from generate_feature_table
feature_table = pd.read_csv(snakemake.input[0], sep='\t')
print(f"Feature table shape: {feature_table.shape}")

print("Loading individual control cells for bootstrap sampling...")
# Remaining inputs are filtered parquet files containing individual cells
combined_controls = []
for input_file in snakemake.input[1:]:  # Skip the feature table
    print(f"Loading control cells from {input_file}")
    df = pd.read_parquet(input_file)
    
    # Filter for control cells only
    control_mask = df[perturbation_col].str.contains(control_key, na=False)
    control_cells = df[control_mask]
    if len(control_cells) > 0:
        combined_controls.append(control_cells)

if not combined_controls:
    raise ValueError("No control cells found for bootstrap sampling")

all_control_cells = pd.concat(combined_controls, ignore_index=True)
print(f"Total control cells for bootstrap sampling: {len(all_control_cells)}")

# Load metadata columns and split control cell data
metadata_cols = load_metadata_cols(metadata_cols_fp, include_classification_cols=True)
controls_metadata, controls_features = split_cell_data(all_control_cells, metadata_cols)

# Get available features from feature table (excluding metadata columns)
available_features = [col for col in feature_table.columns 
                     if col not in [perturbation_col, 'cell_count', 'avg_cells_per_sgrna']]

print(f"Using {len(available_features)} features for bootstrap analysis")

# Filter control features to match available features
controls_features_selected = controls_features[available_features]

# Create controls array (individual cells for sampling)
controls_data = pd.concat([controls_metadata[[perturbation_col]], controls_features_selected], axis=1)
controls_arr = controls_data.values

# Create construct features array from feature table
# Filter out controls and exclusion strings from feature table
construct_mask = ~feature_table[perturbation_col].str.contains(control_key, na=False)
if exclusion_string is not None:
    construct_mask = construct_mask & ~feature_table[perturbation_col].str.contains(exclusion_string, na=False)

construct_features_df = feature_table[construct_mask]

# Create construct features array (perturbation ID + median features)
construct_data_cols = [perturbation_col] + available_features
construct_features_arr = construct_features_df[construct_data_cols].values

# Create sample sizes dataframe
sample_sizes_df = construct_features_df[[perturbation_col, 'cell_count']].copy()

print(f"Controls array shape: {controls_arr.shape}")
print(f"Construct features array shape: {construct_features_arr.shape}")
print(f"Sample sizes dataframe shape: {sample_sizes_df.shape}")

# Save bootstrap arrays
print("Saving bootstrap arrays...")
np.save(snakemake.output.controls_arr, controls_arr)
np.save(snakemake.output.construct_features_arr, construct_features_arr)
sample_sizes_df.to_csv(snakemake.output.sample_sizes, sep='\t', index=False)

# Create checkpoint directory and construct data files
output_dir = Path(snakemake.output[0])  # Directory output
output_dir.mkdir(parents=True, exist_ok=True)

# Extract construct IDs from the array
construct_ids = [str(row[0]) for row in construct_features_arr if not pd.isna(row[0])]
print(f"Found {len(construct_ids)} unique constructs")

# Create simple construct data files (just the construct ID for snakemake)
print(f"Saving {len(construct_ids)} construct data files to {output_dir}")

def write_construct_data(construct_id):
    """Write construct data file for a single construct."""
    print(f"Processing construct: {construct_id}")
    
    # Create metadata file for the construct
    construct_data = pd.DataFrame({
        'construct_id': [construct_id],
        'gene': [construct_id.split('.')[0] if '.' in construct_id else construct_id]
    })
    
    # Save construct data file with simple naming
    output_file = output_dir / f"{construct_id}_construct_data.tsv"
    construct_data.to_csv(output_file, sep='\t', index=False)

# Process all constructs in parallel
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    executor.map(write_construct_data, construct_ids)

print(f"Bootstrap preparation complete! Created arrays and {len(construct_ids)} construct files")