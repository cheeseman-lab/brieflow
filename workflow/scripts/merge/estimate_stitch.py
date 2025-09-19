"""Estimate stitching configuration for microscopy imaging data.

This script generates a YAML configuration file containing tile positions
for stitching microscopy images. It works for both SBS (Sequencing By Synthesis)
and phenotype data types, using coordinate-based estimation to convert stage 
positions to pixel coordinates.
"""

import pandas as pd
import yaml
import numpy as np
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.estimate_stitch import (
    estimate_stitch_coordinate_based,
    convert_numpy_types,
)


# Get parameters
plate = snakemake.params.plate
well = snakemake.params.well
data_type = snakemake.params.data_type

# Load metadata - use the first (and only) input file
metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))

if data_type == "sbs":
    # Apply SBS metadata filters (important for cycle filtering)
    sbs_filters = snakemake.params.get("sbs_metadata_filters", None)
    if sbs_filters is not None:
        for filter_key, filter_value in sbs_filters.items():
            print(f"Filtering SBS metadata: {filter_key} == {filter_value}")
            metadata = metadata[metadata[filter_key] == filter_value]
    
    print(f"After filtering - SBS metadata: {len(metadata)} entries")
    
elif data_type == "phenotype":
    print(f"Loaded phenotype metadata: {len(metadata)} entries")

# Filter to specific plate and well
well_metadata = metadata[
    (metadata["plate"] == int(plate)) & (metadata["well"] == well)
]

print(f"=== Estimating {data_type.upper()} Stitching for Plate {plate}, Well {well} ===")
print(f"{data_type.capitalize()} tiles: {len(well_metadata)}")

if len(well_metadata) == 0:
    print(f"Warning: No {data_type} tiles found for this well")
    # Create empty config
    empty_config = {"total_translation": {}, "confidence": {well: {}}}

    with open(snakemake.output[0], "w") as f:
        yaml.dump(convert_numpy_types(empty_config), f)

    print(f"Created empty {data_type} stitching configuration")
else:
    # Estimate stitching using coordinate-based approach
    print(f"Using coordinate-based estimation for {data_type} data")
    stitch_result = estimate_stitch_coordinate_based(
        metadata_df=well_metadata,
        well=well,
        data_type=data_type,
    )

    # Validate results
    shifts = stitch_result["total_translation"]
    print(f"Generated {len(shifts)} {data_type} tile positions")

    coverage_percent = len(shifts) / len(well_metadata) * 100
    
    print(f"Coverage: {len(shifts)}/{len(well_metadata)} = {coverage_percent:.1f}%")

    # Save results
    with open(snakemake.output[0], "w") as f:
        yaml.dump(convert_numpy_types(stitch_result), f)

    print(f"{data_type.capitalize()} stitching estimation completed successfully")
    print(f"Config saved to: {snakemake.output[0]}")