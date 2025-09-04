"""Estimate stitching configuration for SBS (Sequencing By Synthesis) imaging data.

This script generates a YAML configuration file containing tile positions
for stitching SBS microscopy images. It uses coordinate-based estimation
to convert stage positions to pixel coordinates.
"""

import pandas as pd
import yaml
import numpy as np
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.estimate_stitch import estimate_stitch_sbs_coordinate_based, convert_numpy_types


# Load SBS metadata
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input.sbs_metadata))

# Apply SBS metadata filters (important for cycle filtering)
sbs_filters = snakemake.params.get("sbs_metadata_filters", None)
if sbs_filters is not None:
    for filter_key, filter_value in sbs_filters.items():
        print(f"Filtering SBS metadata: {filter_key} == {filter_value}")
        sbs_metadata = sbs_metadata[sbs_metadata[filter_key] == filter_value]

print(f"After filtering - SBS metadata: {len(sbs_metadata)} entries")

# Filter to specific plate and well
plate = snakemake.params.plate
well = snakemake.params.well

sbs_well_metadata = sbs_metadata[
    (sbs_metadata["plate"] == int(plate)) & (sbs_metadata["well"] == well)
]

print(f"=== Estimating SBS Stitching for Plate {plate}, Well {well} ===")
print(f"SBS tiles: {len(sbs_well_metadata)}")

if len(sbs_well_metadata) == 0:
    print("Warning: No SBS tiles found for this well")
    # Create empty config
    empty_config = {"total_translation": {}, "confidence": {well: {}}}

    with open(snakemake.output[0], "w") as f:
        yaml.dump(convert_numpy_types(empty_config), f)

    print("Created empty SBS stitching configuration")
else:
    # Estimate stitching for SBS data using coordinate-based approach
    print("Using coordinate-based estimation for SBS data")
    sbs_stitch_result = estimate_stitch_sbs_coordinate_based(
        metadata_df=sbs_well_metadata,
        well=well,
    )

    # Validate results
    shifts = sbs_stitch_result["total_translation"]
    print(f"Generated {len(shifts)} SBS tile positions")

    coverage_percent = len(shifts) / len(sbs_well_metadata) * 100
    print(f"Coverage: {len(shifts)}/{len(sbs_well_metadata)} = {coverage_percent:.1f}%")

    if coverage_percent < 95:  # Coordinate-based should give near 100%
        print("⚠️  Warning: Unexpected low coverage for coordinate-based approach")
    else:
        print("✅ Excellent coverage achieved")

    # Save results
    with open(snakemake.output[0], "w") as f:
        yaml.dump(convert_numpy_types(sbs_stitch_result), f)

    print("SBS stitching estimation completed successfully")
    print(f"Config saved to: {snakemake.output[0]}")