import pandas as pd
import yaml

from lib.shared.file_utils import validate_dtypes
from lib.merge.merge_well import well_merge_pipeline

# Load cell information
phenotype_info = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_info = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Load stitch configurations
with open(snakemake.input[2], "r") as f:  # Third input (phenotype_stitch_config)
    phenotype_config = yaml.safe_load(f)

with open(snakemake.input[3], "r") as f:  # Fourth input (sbs_stitch_config)
    sbs_config = yaml.safe_load(f)

# Get parameters
plate = snakemake.params.plate
well = snakemake.params.well

print(f"Starting well-level merge for Plate {plate}, Well {well}")

# Filter to specific plate and well
phenotype_well = phenotype_info[
    (phenotype_info["plate"] == int(plate)) & (phenotype_info["well"] == well)
]

sbs_well = sbs_info[(sbs_info["plate"] == int(plate)) & (sbs_info["well"] == well)]

print(f"Found {len(phenotype_well)} phenotype cells, {len(sbs_well)} SBS cells")

# Perform well-level merge
merged_cells, alignment_df = well_merge_pipeline(
    phenotype_info=phenotype_well,
    sbs_info=sbs_well,
    phenotype_shifts=phenotype_config["total_translation"],
    sbs_shifts=sbs_config["total_translation"],
    well=well,
    det_range=snakemake.params.det_range,
    score_threshold=snakemake.params.score,
    distance_threshold=snakemake.params.threshold,
)

# Save results
merged_cells.to_parquet(snakemake.output[0])

print(f"Well-level merge completed. Merged {len(merged_cells)} cells.")
