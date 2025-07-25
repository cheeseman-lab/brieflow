import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.stitch_well import assemble_aligned_tiff_well

# Load metadata
phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Load stitch configurations
with open(snakemake.input[2], "r") as f:
    phenotype_config = yaml.safe_load(f)

with open(snakemake.input[3], "r") as f:
    sbs_config = yaml.safe_load(f)

# Filter metadata to specific plate and well
plate = snakemake.params.plate
well = snakemake.params.well

phenotype_well_metadata = phenotype_metadata[
    (phenotype_metadata["plate"] == int(plate)) & (phenotype_metadata["well"] == well)
]

sbs_well_metadata = sbs_metadata[
    (sbs_metadata["plate"] == int(plate)) & (sbs_metadata["well"] == well)
]

# Stitch phenotype well
print(f"Stitching phenotype well - Plate {plate}, Well {well}")
phenotype_stitched = assemble_aligned_tiff_well(
    metadata_df=phenotype_well_metadata,
    shifts=phenotype_config["total_translation"],
    well=well,
    flipud=snakemake.params.flipud,
    fliplr=snakemake.params.fliplr,
    rot90=snakemake.params.rot90,
    overlap_percent=snakemake.params.overlap_percent,
)

# Stitch SBS well
print(f"Stitching SBS well - Plate {plate}, Well {well}")
sbs_stitched = assemble_aligned_tiff_well(
    metadata_df=sbs_well_metadata,
    shifts=sbs_config["total_translation"],
    well=well,
    flipud=snakemake.params.flipud,
    fliplr=snakemake.params.fliplr,
    rot90=snakemake.params.rot90,
    overlap_percent=snakemake.params.overlap_percent,
)

# Save stitched images as numpy arrays
np.save(snakemake.output[0], phenotype_stitched)
np.save(snakemake.output[1], sbs_stitched)

print("Well stitching completed successfully")
