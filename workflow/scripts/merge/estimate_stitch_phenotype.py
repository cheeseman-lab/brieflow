import pandas as pd
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.stitch_well import estimate_stitch_phenotype_coordinate_based
import numpy as np


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for YAML serialization"""
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


# Load phenotype metadata
phenotype_metadata = validate_dtypes(
    pd.read_parquet(snakemake.input.phenotype_metadata)
)

print(f"Loaded phenotype metadata: {len(phenotype_metadata)} entries")

# Filter to specific plate and well
plate = snakemake.params.plate
well = snakemake.params.well

phenotype_well_metadata = phenotype_metadata[
    (phenotype_metadata["plate"] == int(plate)) & (phenotype_metadata["well"] == well)
]

print(f"=== Estimating Phenotype Stitching for Plate {plate}, Well {well} ===")
print(f"Phenotype tiles: {len(phenotype_well_metadata)}")

if len(phenotype_well_metadata) == 0:
    print("Warning: No phenotype tiles found for this well")
    # Create empty config
    empty_config = {"total_translation": {}, "confidence": {well: {}}}

    with open(snakemake.output[0], "w") as f:  # Use index [0]
        yaml.dump(convert_numpy_types(empty_config), f)

    print("Created empty phenotype stitching configuration")
    exit(0)

# Estimate stitching for phenotype data using optimized phenotype-specific function
print(f"Using optimized phenotype estimation (image registration approach)")
phenotype_stitch_result = estimate_stitch_phenotype_coordinate_based(
    metadata_df=phenotype_well_metadata,
    well=well,
    flipud=snakemake.params.flipud,
    fliplr=snakemake.params.fliplr,
    rot90=snakemake.params.rot90,
    channel=snakemake.params.channel,
)

# Validate results
shifts = phenotype_stitch_result["total_translation"]
print(f"Generated {len(shifts)} phenotype tile positions")

if len(shifts) < len(phenotype_well_metadata) * 0.8:  # Less than 80% coverage
    print(
        f"⚠️  Warning: Low coverage ({len(shifts)}/{len(phenotype_well_metadata)} = {len(shifts) / len(phenotype_well_metadata) * 100:.1f}%)"
    )
else:
    print(
        f"✅ Good coverage: {len(shifts)}/{len(phenotype_well_metadata)} = {len(shifts) / len(phenotype_well_metadata) * 100:.1f}%"
    )

# Save results
with open(snakemake.output[0], "w") as f:
    yaml.dump(convert_numpy_types(phenotype_stitch_result), f)

print("Phenotype stitching estimation completed successfully")
print(f"Config saved to: {snakemake.output[0]}")
