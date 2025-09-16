import pandas as pd
import sys

from lib.aggregate.cell_data_utils import load_metadata_cols, split_cell_data
from lib.aggregate.filter import (
    query_filter,
    perturbation_filter,
    missing_values_filter,
    intensity_filter,
)

# Load cell data
cell_data = pd.read_parquet(snakemake.input[0])
print(f"Loaded {len(cell_data)} initial cells")

# Check if we have any data to work with
if len(cell_data) == 0:
    print("Warning: No cells in input data, creating empty output file")
    # Create empty output file with same structure
    cell_data.to_parquet(snakemake.output[0], index=False)
    sys.exit(0)

metadata_cols = load_metadata_cols(
    snakemake.params.metadata_cols_fp,
    include_classification_cols=True,
)
metadata, features = split_cell_data(cell_data, metadata_cols)

print(f"Split data into {len(metadata)} metadata rows and {features.shape[1]} feature columns")

# Filter with robust error handling
try:
    metadata, features = query_filter(
        metadata,
        features,
        snakemake.params.filter_queries,
    )
    print(f"After query filter: {len(metadata)} cells remaining")
    
    # Early exit if no cells remain
    if len(metadata) == 0:
        print("No cells remaining after query filtering, creating empty output")
        cell_data = pd.concat([metadata, features], axis=1)
        cell_data.to_parquet(snakemake.output[0], index=False)
        sys.exit(0)
    
    metadata, features = perturbation_filter(
        metadata,
        features,
        snakemake.params.perturbation_name_col,
    )
    print(f"After perturbation filter: {len(metadata)} cells remaining")
    
    # Early exit if no cells remain
    if len(metadata) == 0:
        print("No cells remaining after perturbation filtering, creating empty output")
        cell_data = pd.concat([metadata, features], axis=1)
        cell_data.to_parquet(snakemake.output[0], index=False)
        sys.exit(0)
    
    metadata, features = missing_values_filter(
        metadata,
        features,
        drop_cols_threshold=snakemake.params.drop_cols_threshold,
        drop_rows_threshold=snakemake.params.drop_rows_threshold,
        impute=snakemake.params.impute,
    )
    print(f"After missing values filter: {len(metadata)} cells remaining")
    
    # Early exit if no cells remain
    if len(metadata) == 0:
        print("No cells remaining after missing values filtering, creating empty output")
        cell_data = pd.concat([metadata, features], axis=1)
        cell_data.to_parquet(snakemake.output[0], index=False)
        sys.exit(0)
    
    metadata, features = intensity_filter(
        metadata,
        features,
        snakemake.params.channel_names,
        snakemake.params.contamination,
    )
    print(f"After intensity filter: {len(metadata)} cells remaining")

except Exception as e:
    print(f"Error during filtering: {e}")
    print("Attempting to save current state...")
    # Try to save whatever we have
    try:
        cell_data = pd.concat([metadata, features], axis=1)
        cell_data.to_parquet(snakemake.output[0], index=False)
        print(f"Saved {len(cell_data)} cells to output")
    except Exception as save_error:
        print(f"Error saving data: {save_error}")
        # Create minimal empty file to prevent pipeline failure
        empty_df = pd.DataFrame()
        empty_df.to_parquet(snakemake.output[0], index=False)
        print("Created empty output file")
    sys.exit(1)

# Save filtered data
cell_data = pd.concat([metadata, features], axis=1)
cell_data.to_parquet(snakemake.output[0], index=False)
print(f"Successfully saved {len(cell_data)} filtered cells to {snakemake.output[0]}")

# Print summary
if len(cell_data) == 0:
    print("\nWARNING: Final output contains 0 cells. This may indicate overly restrictive filtering parameters.")
    print("Consider adjusting your filtering criteria:")
    print(f"  - Filter queries: {snakemake.params.filter_queries}")
    print(f"  - Perturbation column: {snakemake.params.perturbation_name_col}")
    print(f"  - Drop columns threshold: {snakemake.params.drop_cols_threshold}")
    print(f"  - Drop rows threshold: {snakemake.params.drop_rows_threshold}")
    print(f"  - Channel names: {snakemake.params.channel_names}")
    print(f"  - Contamination: {snakemake.params.contamination}")