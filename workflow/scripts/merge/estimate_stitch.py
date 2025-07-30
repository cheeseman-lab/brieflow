import pandas as pd
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.stitch_well import estimate_stitch_aligned_tiff
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


# Load metadata
phenotype_metadata = validate_dtypes(
    pd.read_parquet(snakemake.input.phenotype_metadata)
)
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input.sbs_metadata))

# Apply metadata filters (especially important for SBS which has multiple cycles)
phenotype_filters = snakemake.params.get("phenotype_metadata_filters", None)
if phenotype_filters is not None:
    for filter_key, filter_value in phenotype_filters.items():
        phenotype_metadata = phenotype_metadata[
            phenotype_metadata[filter_key] == filter_value
        ]

sbs_filters = snakemake.params.get("sbs_metadata_filters", None)
if sbs_filters is not None:
    for filter_key, filter_value in sbs_filters.items():
        print(f"Filtering SBS metadata: {filter_key} == {filter_value}")
        sbs_metadata = sbs_metadata[sbs_metadata[filter_key] == filter_value]

print(f"After filtering:")
print(f"Phenotype metadata: {len(phenotype_metadata)} entries")
print(f"SBS metadata: {len(sbs_metadata)} entries")

# Filter to specific plate and well
plate = snakemake.params.plate
well = snakemake.params.well

phenotype_well_metadata = phenotype_metadata[
    (phenotype_metadata["plate"] == int(plate)) & (phenotype_metadata["well"] == well)
]

sbs_well_metadata = sbs_metadata[
    (sbs_metadata["plate"] == int(plate)) & (sbs_metadata["well"] == well)
]

print(f"=== Estimating Stitching for Plate {plate}, Well {well} ===")
print(f"Phenotype tiles: {len(phenotype_well_metadata)}")
print(f"SBS tiles: {len(sbs_well_metadata)}")

if len(phenotype_well_metadata) == 0 or len(sbs_well_metadata) == 0:
    print("Warning: No tiles found for this well")
    # Create empty configs
    empty_config = {"total_translation": {}, "confidence": {well: {}}}
    
    with open(snakemake.output[0], "w") as f:
        yaml.dump(convert_numpy_types(empty_config), f)
    
    with open(snakemake.output[1], "w") as f:
        yaml.dump(convert_numpy_types(empty_config), f)
    
    print("Created empty stitching configurations")
    exit(0)

# Estimate stitching for phenotype data
print(f"Estimating stitching for phenotype data - Plate {plate}, Well {well}")
phenotype_stitch_result = estimate_stitch_aligned_tiff(
    metadata_df=phenotype_well_metadata,
    well=well,
    data_type="phenotype",
    flipud=snakemake.params.flipud,
    fliplr=snakemake.params.fliplr,
    rot90=snakemake.params.rot90,
    channel=snakemake.params.channel,
    tile_size=(2400, 2400),  # Phenotype tile size
)

# Estimate stitching for SBS data
print(f"Estimating stitching for SBS data - Plate {plate}, Well {well}")
sbs_stitch_result = estimate_stitch_aligned_tiff(
    metadata_df=sbs_well_metadata,
    well=well,
    data_type="sbs",
    flipud=snakemake.params.flipud,
    fliplr=snakemake.params.fliplr,
    rot90=snakemake.params.rot90,
    channel=snakemake.params.channel,
    tile_size=(1200, 1200),  # SBS tile size
)

# Save results
with open(snakemake.output[0], "w") as f:
    yaml.dump(convert_numpy_types(phenotype_stitch_result), f)

with open(snakemake.output[1], "w") as f:
    yaml.dump(convert_numpy_types(sbs_stitch_result), f)

print("Stitching estimation completed successfully")
print(f"Phenotype config saved to: {snakemake.output[0]}")
print(f"SBS config saved to: {snakemake.output[1]}")