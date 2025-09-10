"""Aggregate cells and vacuoles data using specified strategy.

This script merges cell-level data from the final merge with vacuole-level data
using different aggregation strategies to handle the one-to-many relationship.
"""

import pandas as pd

# Import from the shared library
from lib.aggregate.vacuole_utils import aggregate_vacuole_data
from lib.shared.file_utils import validate_dtypes

# Get inputs from Snakemake
merge_final_path = snakemake.input[0]
phenotype_vacuoles_path = snakemake.input[1]

# Get output paths
aggregated_data_path = snakemake.output[0]

# Get parameters
agg_strategy = snakemake.params.agg_strategy
plate = snakemake.params.plate
well = snakemake.params.well

print(f"Processing plate {plate}, well {well} with strategy '{agg_strategy}'")

# Load input data
print("Loading merge final data...")
cells_df = validate_dtypes(pd.read_parquet(merge_final_path))
print(f"Loaded {len(cells_df)} cells from merge final")

print("Loading phenotype vacuoles data...")
vacuoles_df = validate_dtypes(pd.read_parquet(phenotype_vacuoles_path))
print(f"Loaded {len(vacuoles_df)} vacuoles from phenotype")

# Validate required columns
required_cells_cols = ["plate", "well", "tile", "cell_0"]
required_vacuoles_cols = ["plate", "well", "tile", "cell_id"]

missing_cells = [col for col in required_cells_cols if col not in cells_df.columns]
missing_vacuoles = [
    col for col in required_vacuoles_cols if col not in vacuoles_df.columns
]

if missing_cells:
    raise ValueError(f"Missing required columns in cells data: {missing_cells}")
if missing_vacuoles:
    raise ValueError(f"Missing required columns in vacuoles data: {missing_vacuoles}")

print(f"Cell data columns: {list(cells_df.columns)}")
print(f"Vacuole data columns: {list(vacuoles_df.columns)}")

# Filter data to current plate/well if needed
if "plate" in cells_df.columns and "well" in cells_df.columns:
    cells_df = cells_df[
        (cells_df["plate"] == plate) & (cells_df["well"] == well)
    ].copy()
    print(f"Filtered to {len(cells_df)} cells for plate {plate}, well {well}")

if "plate" in vacuoles_df.columns and "well" in vacuoles_df.columns:
    vacuoles_df = vacuoles_df[
        (vacuoles_df["plate"] == plate) & (vacuoles_df["well"] == well)
    ].copy()
    print(f"Filtered to {len(vacuoles_df)} vacuoles for plate {plate}, well {well}")

# Perform aggregation
print(f"Aggregating data using strategy: {agg_strategy}")
result_df = aggregate_vacuole_data(cells_df, vacuoles_df, agg_strategy)

print(
    f"Aggregation complete. Result has {len(result_df)} rows and {len(result_df.columns)} columns"
)

# Validate output data types
result_df = validate_dtypes(result_df)

# Save output
print(f"Saving aggregated data to {aggregated_data_path}")
result_df.to_parquet(aggregated_data_path)

print("Aggregation completed successfully!")
