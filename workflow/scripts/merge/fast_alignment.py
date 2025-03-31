import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.merge.hash import hash_cell_locations, multistep_alignment, extract_rotation


# Load dfs with metadata on well level
phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Apply metadata filters if they exist
phenotype_filters = snakemake.params.get("phenotype_metadata_filters", None)
if phenotype_filters is not None:
    for filter_key, filter_value in phenotype_filters.items():
        phenotype_metadata = phenotype_metadata[
            phenotype_metadata[filter_key] == filter_value
        ]
sbs_filters = snakemake.params.get("sbs_metadata_filters", None)
if sbs_filters is not None:
    for filter_key, filter_value in sbs_filters.items():
        sbs_metadata = sbs_metadata[sbs_metadata[filter_key] == filter_value]

# Load phentoype/sbs info on well level
phenotype_info = validate_dtypes(pd.read_parquet(snakemake.input[2]))
sbs_info = validate_dtypes(pd.read_parquet(snakemake.input[3]))

# Derive fast alignment per well

# Format XY coordinates for phenotype and SBS
phenotype_xy = phenotype_metadata.rename(
    columns={"x_pos": "x", "y_pos": "y"}
).set_index("tile")[["x", "y"]]
sbs_xy = sbs_metadata.rename(columns={"x_pos": "x", "y_pos": "y"}).set_index("tile")[
    ["x", "y"]
]

# Hash phenotype and sbs info
phenotype_info_hash = validate_dtypes(hash_cell_locations(phenotype_info))
sbs_info_hash = validate_dtypes(
    hash_cell_locations(sbs_info).rename(columns={"tile": "site"})
)

# Perform multistep alignment for well
well_alignment = multistep_alignment(
    phenotype_info_hash,
    sbs_info_hash,
    phenotype_xy,
    sbs_xy,
    det_range=snakemake.params.det_range,
    score=snakemake.params.score,
    initial_sites=snakemake.params.initial_sites,
    n_jobs=snakemake.threads,
)

# Reset index
well_alignment.reset_index(drop=True, inplace=True)

# Parse rotation into 2 columns
well_alignment["rotation_1"] = well_alignment["rotation"].apply(
    lambda r: extract_rotation(r, 1)
)
well_alignment["rotation_2"] = well_alignment["rotation"].apply(
    lambda r: extract_rotation(r, 2)
)
well_alignment.drop(columns=["rotation"], inplace=True)

# Add metadata to alignment data
well_alignment["plate"] = snakemake.params.plate
well_alignment["well"] = snakemake.params.well

# Save alignment data
well_alignment.to_parquet(snakemake.output[0])
