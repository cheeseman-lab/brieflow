import pandas as pd
from lib.shared.file_utils import validate_dtypes
from lib.merge.hash import (
    hash_cell_locations,
    multistep_alignment,
    extract_rotation,
    initial_alignment,
)
from lib.merge.merge_utils import align_metadata, find_closest_tiles

# Load dfs with metadata on well level
phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Apply coordinate alignment if transformation parameters are provided
alignment_params = {
    "flip_x": getattr(snakemake.params, "alignment_flip_x", False),
    "flip_y": getattr(snakemake.params, "alignment_flip_y", False),
    "rotate_90": getattr(snakemake.params, "alignment_rotate_90", False),
}

# Only apply alignment if at least one transformation is requested
if any(
    [
        alignment_params["flip_x"],
        alignment_params["flip_y"],
        alignment_params["rotate_90"],
    ]
):
    print("Applying coordinate alignment transformations...")
    phenotype_metadata, sbs_metadata, transformation_info = align_metadata(
        phenotype_metadata,
        sbs_metadata,
        x_col="x_pos",
        y_col="y_pos",
        **alignment_params,
    )
    print("Coordinate alignment completed.")

# Build and apply SBS filters
sbs_filters = {}
if snakemake.params.sbs_metadata_cycle is not None:
    sbs_filters["cycle"] = snakemake.params.sbs_metadata_cycle
if snakemake.params.sbs_metadata_channel is not None:
    sbs_filters["channel"] = snakemake.params.sbs_metadata_channel

for filter_key, filter_value in sbs_filters.items():
    sbs_metadata = sbs_metadata[sbs_metadata[filter_key] == filter_value]

# Build and apply phenotype filters
ph_filters = {}
if snakemake.params.ph_metadata_channel is not None:
    ph_filters["channel"] = snakemake.params.ph_metadata_channel

for filter_key, filter_value in ph_filters.items():
    phenotype_metadata = phenotype_metadata[
        phenotype_metadata[filter_key] == filter_value
    ]

# If no filters were applied, deduplicate
if not sbs_filters:
    sbs_metadata = sbs_metadata.drop_duplicates(subset=["plate", "well", "tile"])
if not ph_filters:
    phenotype_metadata = phenotype_metadata.drop_duplicates(
        subset=["plate", "well", "tile"]
    )


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

# Get initial site configuration - exactly one must be provided
initial_sbs_tiles = getattr(snakemake.params, "initial_sbs_tiles", None)
initial_sites_param = getattr(snakemake.params, "initial_sites", None)

# Validate exactly one is provided
if (initial_sbs_tiles is None) == (initial_sites_param is None):
    raise ValueError(
        "Exactly one of 'initial_sbs_tiles' or 'initial_sites' must be provided in merge config"
    )

# Get threshold params for validation
d0, d1 = snakemake.params.det_range
score_thresh = snakemake.params.score

if initial_sbs_tiles is not None:
    # Auto-discover initial sites from SBS tiles
    candidate_pairs = []
    for sbs_tile in initial_sbs_tiles:
        closest = find_closest_tiles(
            sbs_metadata, phenotype_metadata, sbs_tile, verbose=False
        )
        best_ph_tile = int(closest.iloc[0]["tile"])
        candidate_pairs.append([best_ph_tile, sbs_tile])
    print(
        f"Discovered {len(candidate_pairs)} candidate pairs from {len(initial_sbs_tiles)} SBS tiles"
    )
else:
    # Use user-provided pairs
    candidate_pairs = initial_sites_param
    print(f"Using {len(candidate_pairs)} user-specified initial sites")

# Run initial alignment on candidates (validates both paths)
initial_alignment_df = initial_alignment(
    phenotype_info_hash, sbs_info_hash, initial_sites=candidate_pairs
)

# Filter by thresholds
valid_pairs_df = initial_alignment_df.query(
    "@d0 <= determinant <= @d1 & score > @score_thresh"
)

# Require minimum 5 valid pairs (only if > 5 candidates were provided)
if len(candidate_pairs) > 5 and len(valid_pairs_df) < 5:
    raise ValueError(
        f"Only {len(valid_pairs_df)} initial sites passed thresholds (need >= 5). "
        f"Candidates tested: {candidate_pairs}. "
        f"Check det_range={snakemake.params.det_range} and score={score_thresh}."
    )

# Convert to list of [tile, site] pairs for multistep_alignment
initial_sites = valid_pairs_df[["tile", "site"]].astype(int).values.tolist()
print(f"{len(initial_sites)} initial sites passed thresholds")

# Perform multistep alignment for well
well_alignment = multistep_alignment(
    phenotype_info_hash,
    sbs_info_hash,
    phenotype_xy,
    sbs_xy,
    det_range=snakemake.params.det_range,
    score=snakemake.params.score,
    initial_sites=initial_sites,
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
