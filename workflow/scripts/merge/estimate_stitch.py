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

# Get fallback pixel sizes from config
fallback_pixel_size = None
if data_type == "sbs":
    fallback_pixel_size = snakemake.params.get("sbs_pixel_size", None)
elif data_type == "phenotype":
    fallback_pixel_size = snakemake.params.get("phenotype_pixel_size", None)

# Load metadata
metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))

# Apply coordinate alignment for SBS data if transformation parameters are provided
if data_type == "sbs":
    alignment_params = {
        'flip_x': getattr(snakemake.params, 'alignment_flip_x', False),
        'flip_y': getattr(snakemake.params, 'alignment_flip_y', False), 
        'rotate_90': getattr(snakemake.params, 'alignment_rotate_90', False),
    }
    
    # Only apply alignment if at least one transformation is requested
    if any([alignment_params['flip_x'], alignment_params['flip_y'], alignment_params['rotate_90']]):
        from lib.merge.merge_utils import align_metadata
        
        print("Loading phenotype metadata for alignment reference...")
        # Load phenotype metadata from second input
        phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input[1]))
        
        print("Applying coordinate alignment transformations...")
        phenotype_aligned, metadata, transformation_info = align_metadata(
            phenotype_metadata, 
            metadata,
            x_col='x_pos',
            y_col='y_pos',
            reference_df=1,  # Use phenotype as reference
            **alignment_params
        )
        print("Coordinate alignment completed.")
        print(f"Original centers: PH={transformation_info['original_centers'][0]}, SBS={transformation_info['original_centers'][1]}")
        print(f"Final centers: PH={transformation_info['final_centers'][0]}, SBS={transformation_info['final_centers'][1]}")

if data_type == "sbs":
    # Apply SBS metadata filters
    sbs_filters = {}
    if snakemake.params.sbs_metadata_cycle is not None:
        sbs_filters["cycle"] = snakemake.params.sbs_metadata_cycle
    if snakemake.params.sbs_metadata_channel is not None:
        sbs_filters["channel"] = snakemake.params.sbs_metadata_channel

    if sbs_filters:
        for filter_key, filter_value in sbs_filters.items():
            metadata = metadata[metadata[filter_key] == filter_value]

    print(f"After filtering - SBS metadata: {len(metadata)} entries")

elif data_type == "phenotype":
    print(f"Loaded phenotype metadata: {len(metadata)} entries")

print(
    f"=== Estimating {data_type.upper()} Stitching for Plate {plate}, Well {well} ==="
)
print(f"{data_type.capitalize()} tiles: {len(metadata)}")

if fallback_pixel_size is not None:
    print(f"Using fallback pixel size from config: {fallback_pixel_size} μm/pixel")

if len(metadata) == 0:
    print(f"Warning: No {data_type} tiles found for this well")
    # Create empty config
    empty_config = {"total_translation": {}}

    with open(snakemake.output[0], "w") as f:
        yaml.dump(convert_numpy_types(empty_config), f)

    print(f"Created empty {data_type} stitching configuration")
else:
    # Estimate stitching using coordinate-based approach
    print(f"Using coordinate-based estimation for {data_type} data")
    stitch_result = estimate_stitch_coordinate_based(
        metadata_df=metadata,
        well=well,
        data_type=data_type,
        fallback_pixel_size=fallback_pixel_size,  # Pass the fallback pixel size
    )

    # Validate results
    shifts = stitch_result["total_translation"]
    print(f"Generated {len(shifts)} {data_type} tile positions")

    coverage_percent = len(shifts) / len(metadata) * 100

    print(f"Coverage: {len(shifts)}/{len(metadata)} = {coverage_percent:.1f}%")

    # Save results
    with open(snakemake.output[0], "w") as f:
        yaml.dump(convert_numpy_types(stitch_result), f)

    print(f"{data_type.capitalize()} stitching estimation completed successfully")
    print(f"Config saved to: {snakemake.output[0]}")