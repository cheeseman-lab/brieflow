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
phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input.phenotype_metadata))
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input.sbs_metadata))


# Filter to specific plate and well
plate = snakemake.params.plate
well = snakemake.params.well

phenotype_well_metadata = phenotype_metadata[
    (phenotype_metadata['plate'] == int(plate)) & 
    (phenotype_metadata['well'] == well)
]

sbs_well_metadata = sbs_metadata[
    (sbs_metadata['plate'] == int(plate)) & 
    (sbs_metadata['well'] == well)
]

#DEBUG PRINT
print(f"=== DEBUGGING WELL {well} ===")
print(f"Phenotype tiles: {len(phenotype_well_metadata)}")
if len(phenotype_well_metadata) > 0:
    print(f"Phenotype x_pos range: {phenotype_well_metadata['x_pos'].min()} to {phenotype_well_metadata['x_pos'].max()}")
    print(f"Phenotype y_pos range: {phenotype_well_metadata['y_pos'].min()} to {phenotype_well_metadata['y_pos'].max()}")
    print(f"Unique x positions: {len(phenotype_well_metadata['x_pos'].unique())}")
    print(f"Unique y positions: {len(phenotype_well_metadata['y_pos'].unique())}")
    print(f"Sample positions:\n{phenotype_well_metadata[['tile', 'x_pos', 'y_pos']].head(10)}")

print(f"SBS tiles: {len(sbs_well_metadata)}")
if len(sbs_well_metadata) > 0:
    print(f"SBS x_pos range: {sbs_well_metadata['x_pos'].min()} to {sbs_well_metadata['x_pos'].max()}")
    print(f"SBS y_pos range: {sbs_well_metadata['y_pos'].min()} to {sbs_well_metadata['y_pos'].max()}")
    print(f"Unique x positions: {len(sbs_well_metadata['x_pos'].unique())}")
    print(f"Unique y positions: {len(sbs_well_metadata['y_pos'].unique())}")
    print(f"Sample positions:\n{sbs_well_metadata[['tile', 'x_pos', 'y_pos']].head(10)}")
print("=" * 50)

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
    tile_size=tuple(snakemake.params.tile_size)
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
    tile_size=tuple(snakemake.params.tile_size)
)

# Save results
with open(snakemake.output[0], 'w') as f:
    yaml.dump(convert_numpy_types(phenotype_stitch_result), f)

with open(snakemake.output[1], 'w') as f:
    yaml.dump(convert_numpy_types(sbs_stitch_result), f)

print("Stitching estimation completed successfully")

