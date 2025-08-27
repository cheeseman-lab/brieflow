"""Estimate stitching configuration for phenotype imaging data.

This script generates a YAML configuration file containing tile positions
for stitching phenotype microscopy images. It uses coordinate-based estimation
to convert stage positions to pixel coordinates.
"""

import pandas as pd
import yaml
import numpy as np
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.estimate_stitch import estimate_stitch_phenotype_coordinate_based


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for YAML serialization.

    YAML serialization requires native Python types, but numpy arrays and
    scalars need to be converted first.

    Args:
        obj: Object that may contain numpy types (dict, list, numpy types, etc.)

    Returns:
        Object with numpy types converted to Python native types
    """
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


def main():
    """Main execution function for phenotype stitching estimation."""
    # Load phenotype metadata
    phenotype_metadata = validate_dtypes(
        pd.read_parquet(snakemake.input.phenotype_metadata)
    )

    print(f"Loaded phenotype metadata: {len(phenotype_metadata)} entries")

    # Filter to specific plate and well
    plate = snakemake.params.plate
    well = snakemake.params.well

    phenotype_well_metadata = phenotype_metadata[
        (phenotype_metadata["plate"] == int(plate))
        & (phenotype_metadata["well"] == well)
    ]

    print(f"=== Estimating Phenotype Stitching for Plate {plate}, Well {well} ===")
    print(f"Phenotype tiles: {len(phenotype_well_metadata)}")

    if len(phenotype_well_metadata) == 0:
        print("Warning: No phenotype tiles found for this well")
        # Create empty config
        empty_config = {"total_translation": {}, "confidence": {well: {}}}

        with open(snakemake.output[0], "w") as f:
            yaml.dump(convert_numpy_types(empty_config), f)

        print("Created empty phenotype stitching configuration")
        return

    # Estimate stitching for phenotype data using coordinate-based approach
    print("Using coordinate-based estimation for phenotype data")
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

    coverage_percent = len(shifts) / len(phenotype_well_metadata) * 100
    if coverage_percent < 80:  # Less than 80% coverage
        print(
            f"⚠️  Warning: Low coverage ({len(shifts)}/{len(phenotype_well_metadata)} = {coverage_percent:.1f}%)"
        )
    else:
        print(
            f"✅ Good coverage: {len(shifts)}/{len(phenotype_well_metadata)} = {coverage_percent:.1f}%"
        )

    # Save results
    with open(snakemake.output[0], "w") as f:
        yaml.dump(convert_numpy_types(phenotype_stitch_result), f)

    print("Phenotype stitching estimation completed successfully")
    print(f"Config saved to: {snakemake.output[0]}")


if __name__ == "__main__":
    main()
