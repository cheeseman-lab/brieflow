import pandas as pd
from lib.shared.file_utils import validate_dtypes
from lib.merge.hash import (
    hash_cell_locations,
    multistep_alignment,
    extract_rotation,
    initial_alignment,
)
from lib.merge.merge_utils import (
    align_metadata,
    find_closest_tiles,
    filter_low_score_seeds,
)

# Load dfs with metadata on well level
phenotype_metadata = validate_dtypes(pd.read_parquet(snakemake.input[0]))
sbs_metadata = validate_dtypes(pd.read_parquet(snakemake.input[1]))

# Apply coordinate alignment if transformation parameters are provided
alignment_params = {
    "flip_x": getattr(snakemake.params, "alignment_flip_x", False),
    "flip_y": getattr(snakemake.params, "alignment_flip_y", False),
    "rotate_90": getattr(snakemake.params, "alignment_rotate_90", False),
}

# metadata_align center-aligns the two scopes' coordinate frames (translation only)
metadata_align = getattr(snakemake.params, "metadata_align", False)

# Apply alignment if center-alignment is requested or any flip/rotate is set
if metadata_align or any(
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


# Optional alignment levers; absent config keys drop out to lib defaults
def _drop_none(d):
    return {k: v for k, v in d.items() if v is not None}


ransac_kwargs = _drop_none(
    {"random_state": getattr(snakemake.params, "ransac_random_state", None)}
)
evaluate_kwargs = (
    _drop_none(
        {
            "threshold_triangle": getattr(snakemake.params, "threshold_triangle", None),
            "ransac_kwargs": ransac_kwargs or None,
        }
    )
    or None
)

# find-optimal-site (gated): try top-K nearest PH tiles per SBS seed, keep best per site
seed_optimize = getattr(snakemake.params, "seed_optimize", False)
seed_topk = getattr(snakemake.params, "seed_topk", None) or 3

if initial_sbs_tiles is not None:
    # Auto-discover initial sites from SBS tiles
    candidate_pairs = []
    for sbs_tile in initial_sbs_tiles:
        closest = find_closest_tiles(
            sbs_metadata, phenotype_metadata, sbs_tile, verbose=False
        )
        if seed_optimize:
            for ph_tile in closest.head(seed_topk)["tile"].astype(int):
                candidate_pairs.append([int(ph_tile), sbs_tile])
        else:
            candidate_pairs.append([int(closest.iloc[0]["tile"]), sbs_tile])
    print(
        f"Discovered {len(candidate_pairs)} candidate pairs from {len(initial_sbs_tiles)} SBS tiles"
        + (f" (seed_optimize: top-{seed_topk} per tile)" if seed_optimize else "")
    )
else:
    # Use user-provided pairs
    candidate_pairs = initial_sites_param
    print(f"Using {len(candidate_pairs)} user-specified initial sites")

# Run initial alignment on candidates (validates both paths)
initial_alignment_df = initial_alignment(
    phenotype_info_hash,
    sbs_info_hash,
    initial_sites=candidate_pairs,
    evaluate_kwargs=evaluate_kwargs,
)

# Filter by thresholds
valid_pairs_df = initial_alignment_df.query(
    "@d0 <= determinant <= @d1 & score > @score_thresh"
)

# find-optimal-site: collapse top-K candidates to the best-scoring tile per site
if seed_optimize:
    n_before = len(valid_pairs_df)
    valid_pairs_df = valid_pairs_df.sort_values(
        "score", ascending=False
    ).drop_duplicates(subset="site", keep="first")
    print(
        f"seed_optimize: kept best-scoring tile per site ({n_before} -> {len(valid_pairs_df)})"
    )

# Drop seeds whose score is a low outlier relative to the cohort (keeps >= 5)
n_before = len(valid_pairs_df)
valid_pairs_df = filter_low_score_seeds(valid_pairs_df)
if len(valid_pairs_df) < n_before:
    print(
        f"filtered {n_before - len(valid_pairs_df)} low-score outlier seed(s) "
        f"({n_before} -> {len(valid_pairs_df)})"
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
    evaluate_kwargs=evaluate_kwargs,
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
