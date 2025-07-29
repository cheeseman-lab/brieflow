from pathlib import Path
import multiprocessing
import numpy as np
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data

# Get parameters
perturbation_col = snakemake.params.perturbation_name_col
perturbation_id_col = snakemake.params.perturbation_id_col
control_key = snakemake.params.control_key
exclusion_string = snakemake.params.exclusion_string
metadata_cols_fp = snakemake.params.metadata_cols_fp

print("Loading center-scaled single-cell features data...")
# Load center-scaled single-cell features data (input 0)
all_features_cells = pd.read_parquet(snakemake.input[0])
print(f"Features single-cell data shape: {all_features_cells.shape}")

print("Loading construct table...")
# Load construct table (input 1)
construct_table = pd.read_csv(snakemake.input[1], sep="\t")
print(f"Construct table shape: {construct_table.shape}")

print("Loading gene table...")
# Load gene table (input 2)
gene_table = pd.read_csv(snakemake.input[2], sep="\t")
print(f"Gene table shape: {gene_table.shape}")

# Filter for control cells only (these are already center-scaled)
control_mask = all_features_cells[perturbation_col].str.contains(control_key, na=False)
control_cells = all_features_cells[control_mask]
print(f"Center-scaled control cells for bootstrap sampling: {len(control_cells)}")

# Load metadata columns and split control cell data
metadata_cols = load_metadata_cols(metadata_cols_fp, include_classification_cols=True)
controls_metadata, controls_features = split_cell_data(control_cells, metadata_cols)

# Get available features from construct table (excluding metadata columns)
available_features = [
    col
    for col in construct_table.columns
    if col not in [perturbation_id_col, perturbation_col, "cell_count"]
]

print(f"Using {len(available_features)} features for bootstrap analysis")

# Filter control features to match available features
controls_features_selected = controls_features[available_features]

# Create controls array (individual cells for sampling) - ALREADY CENTER-SCALED
controls_data = pd.concat(
    [controls_metadata[[perturbation_col]], controls_features_selected], axis=1
)
controls_arr = controls_data.values

# Create construct features array from construct table
# Filter out controls
construct_mask = ~construct_table[perturbation_col].str.contains(control_key, na=False)
if exclusion_string is not None:
    construct_mask = construct_mask & ~construct_table[perturbation_col].str.contains(
        exclusion_string, na=False
    )

construct_features_df = construct_table[construct_mask]

# Create construct features array (sgRNA_ID + features)
construct_data_cols = [perturbation_id_col] + available_features
construct_features_arr = construct_features_df[construct_data_cols].values

# Create sample sizes dataframe (sgRNA_ID + cell_count)
sample_sizes_df = construct_features_df[[perturbation_id_col, "cell_count"]].copy()

print(f"Controls array shape: {controls_arr.shape}")
print(f"Construct features array shape: {construct_features_arr.shape}")
print(f"Sample sizes dataframe shape: {sample_sizes_df.shape}")

# Save bootstrap arrays as TSV
print("Saving bootstrap arrays...")
# Controls as TSV
controls_df = pd.DataFrame(controls_arr)
controls_df.columns = [perturbation_col] + available_features
controls_df.to_csv(snakemake.output.controls_arr, sep="\t", index=False)

# Construct features as TSV
construct_features_df_export = pd.DataFrame(construct_features_arr)
construct_features_df_export.columns = [perturbation_id_col] + available_features
construct_features_df_export.to_csv(
    snakemake.output.construct_features_arr, sep="\t", index=False
)

# Sample sizes as TSV
sample_sizes_df.to_csv(snakemake.output.sample_sizes, sep="\t", index=False)

# Create checkpoint directory and construct data files
output_dir = Path(snakemake.output[0])  # Directory output
output_dir.mkdir(parents=True, exist_ok=True)

# Extract construct IDs (sgRNA IDs) from the array
construct_ids = [str(row[0]) for row in construct_features_arr if not pd.isna(row[0])]
print(f"Found {len(construct_ids)} unique constructs")

# Create construct data files for snakemake job spawning
print(f"Saving {len(construct_ids)} construct data files to {output_dir}")


def write_construct_data(construct_id):
    """Write construct data file for a single sgRNA."""
    # Get the gene for this construct
    construct_row = construct_features_df[
        construct_features_df[perturbation_id_col] == construct_id
    ]
    if len(construct_row) > 0:
        gene = construct_row[perturbation_col].iloc[0]
    else:
        gene = "unknown"

    # Create gene_sgRNA combined ID
    combined_id = f"{gene}_{construct_id}"

    # Create metadata file for the construct
    construct_data = pd.DataFrame(
        {"construct_id": [construct_id], "gene": [gene], "combined_id": [combined_id]}
    )

    # Save using combined_id for filename
    output_file = output_dir / f"{combined_id}_construct_data.tsv"
    construct_data.to_csv(output_file, sep="\t", index=False)


# Process all constructs in parallel
with ThreadPoolExecutor(max_workers=multiprocessing.cpu_count()) as executor:
    executor.map(write_construct_data, construct_ids)

print(
    f"Bootstrap preparation complete! Created arrays and {len(construct_ids)} construct files"
)
