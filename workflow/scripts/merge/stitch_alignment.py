"""Stitch Alignment - Coordinates scaling, triangle hashing, and alignment estimation.

Workflow:
1. Load phenotype and SBS cell positions
2. Performs alignment workflow:
    - Auto-calculate scale factor from coordinate ranges
    - Scale phenotype coordinates to match SBS
    - Generate triangle hash features
    - Perform adaptive regional alignment
    - Apply transformation
3. Save all outputs (scaled coords, triangles, alignment params, summary)
"""

import pandas as pd
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.stitch_alignment import align_well_positions


# Load inputs
phenotype_positions = validate_dtypes(
    pd.read_parquet(snakemake.input.phenotype_positions)
)
sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))

plate = snakemake.params.plate
well = snakemake.params.well
score_threshold = snakemake.params.score

print(f"Processing Plate {plate}, Well {well}")
print(f"Phenotype cells: {len(phenotype_positions):,}")
print(f"SBS cells: {len(sbs_positions):,}")

# Run alignment workflow
try:
    result = align_well_positions(
        phenotype_positions=phenotype_positions,
        sbs_positions=sbs_positions,
        score_threshold=score_threshold,
        adaptive_region=True,
        max_cells_for_hash=75000,
        initial_region_size=7000,
        min_triangles=100,
        threshold_triangle=0.3,
        threshold_point=2.0,
    )

    # Extract results
    status = result["status"]
    phenotype_scaled = result["phenotype_scaled"]
    phenotype_transformed = result["phenotype_transformed"]
    phenotype_triangles = result["phenotype_triangles"]
    sbs_triangles = result["sbs_triangles"]
    alignment_params = result["alignment_params"]
    summary = result["summary"]

    print(f"\n Alignment completed with status: {status}")

except ValueError as e:
    # Critical error - insufficient cells
    print(f" CRITICAL ERROR: {e}")
    raise

except Exception as e:
    # Unexpected error - report but continue with failed result
    print(f"  Unexpected error during alignment: {e}")
    print("Creating failed result with identity transformation...")

    # Use the library's failed result creator
    from lib.merge.stitch_alignment import (
        calculate_scale_factor_from_positions,
        scale_coordinates,
        calculate_coordinate_overlap,
    )

    scale_factor = calculate_scale_factor_from_positions(
        phenotype_positions, sbs_positions
    )
    phenotype_scaled = scale_coordinates(phenotype_positions, scale_factor)
    overlap_fraction = calculate_coordinate_overlap(phenotype_scaled, sbs_positions)

    # Import the helper function
    from lib.merge.stitch_alignment import _create_failed_result

    failed_result = _create_failed_result(
        phenotype_positions=phenotype_positions,
        phenotype_scaled=phenotype_scaled,
        sbs_positions=sbs_positions,
        scale_factor=scale_factor,
        overlap_fraction=overlap_fraction,
        failure_reason=f"unexpected_error: {str(e)[:50]}",
    )

    # Extract results
    phenotype_transformed = failed_result["phenotype_transformed"]
    phenotype_triangles = failed_result["phenotype_triangles"]
    sbs_triangles = failed_result["sbs_triangles"]
    alignment_params = failed_result["alignment_params"]
    summary = failed_result["summary"]

# Save outputs
print("\n--- Saving Outputs ---")

# Output [0]: Scaled phenotype positions
phenotype_scaled.to_parquet(str(snakemake.output.scaled_phenotype_positions))
print(f" Scaled positions: {snakemake.output.scaled_phenotype_positions}")

# Output [1]: Phenotype triangles
phenotype_triangles.to_parquet(str(snakemake.output.phenotype_triangles))
print(f" Phenotype triangles: {snakemake.output.phenotype_triangles}")

# Output [2]: SBS triangles
sbs_triangles.to_parquet(str(snakemake.output.sbs_triangles))
print(f" SBS triangles: {snakemake.output.sbs_triangles}")

# Output [3]: Alignment parameters
alignment_params.to_parquet(str(snakemake.output.alignment_params))
print(f" Alignment params: {snakemake.output.alignment_params}")

# Output [4]: Summary (TSV format with plate/well columns)
summary_with_ids = {"plate": str(plate), "well": str(well), **summary}
summary_df = pd.DataFrame([summary_with_ids])
summary_df.to_csv(
    str(snakemake.output.alignment_summary),
    sep="\t",
    index=False,
    float_format="%.6g",
)
print(f" Alignment summary: {snakemake.output.alignment_summary}")

# Output [5]: Transformed phenotype positions
phenotype_transformed.to_parquet(str(snakemake.output.transformed_phenotype_positions))
print(f" Transformed positions: {snakemake.output.transformed_phenotype_positions}")

print(f"\n Well alignment completed successfully for {plate}/{well}!")
