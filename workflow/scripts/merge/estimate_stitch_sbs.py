import pandas as pd
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.stitch_well import estimate_stitch_sbs_coordinate_based
import numpy as np


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for YAML serialization."""
    if isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


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

    with open(snakemake.output.sbs_stitch_config, "w") as f:
        yaml.dump(convert_numpy_types(empty_config), f)

    print("Created empty SBS stitching configuration")
    exit(0)

# Estimate stitching for SBS data using coordinate-based approach
print(f"Using optimized SBS estimation (coordinate-based approach)")
sbs_stitch_result = estimate_stitch_sbs_coordinate_based(
    metadata_df=sbs_well_metadata,
    well=well,
    flipud=snakemake.params.flipud,
    fliplr=snakemake.params.fliplr,
    rot90=snakemake.params.rot90,
    channel=snakemake.params.channel,
)

# Validate results
shifts = sbs_stitch_result["total_translation"]
print(f"Generated {len(shifts)} SBS tile positions")

coverage_percent = len(shifts) / len(sbs_well_metadata) * 100
print(f"Coverage: {len(shifts)}/{len(sbs_well_metadata)} = {coverage_percent:.1f}%")

if coverage_percent < 95:  # Coordinate-based should give near 100%
    print(f"⚠️  Warning: Unexpected low coverage for coordinate-based approach")
else:
    print(f"✅ Excellent coverage achieved")

# Save results
with open(snakemake.output[0], "w") as f:
    yaml.dump(convert_numpy_types(sbs_stitch_result), f)

print("SBS stitching estimation completed successfully")
print(f"Config saved to: {snakemake.output[0]}")
