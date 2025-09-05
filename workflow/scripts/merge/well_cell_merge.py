"""Step 2: Well Cell Merge - Cell-to-cell matching using alignment parameters.

This script performs cell-to-cell matching between phenotype and SBS datasets within
a single well using spatial alignment transformations. It includes tile diversity
filtering, coordinate matching, and metadata extraction for downstream processing.

Key steps:
1. Filter out tiles with insufficient cell diversity
2. Load alignment parameters and apply spatial transformations
3. Match cells based on spatial proximity
4. Extract metadata and validate match quality
5. Save results with comprehensive statistics

Input files:
- scaled_phenotype_positions: From well_alignment rule output [0]
- sbs_positions: From stitch_sbs_well rule output [0]
- alignment_params: From well_alignment rule output [3]
- transformed_phenotype_positions: From well_alignment rule output [5]

Output files:
- raw_matches.parquet: Initial cell match results (output [0])
- merged_cells.parquet: Same as raw_matches for compatibility (output [1])
- merge_summary.tsv: Comprehensive processing statistics (output [2])
"""

import pandas as pd
import numpy as np
import traceback

from lib.shared.file_utils import validate_dtypes
from lib.merge.well_cell_matching import (
    load_alignment_parameters,
    find_cell_matches,
    validate_matches,
    create_empty_outputs,
    filter_tiles_by_diversity,
)

print("=== WELL CELL MERGE ===")

# Load inputs - using named inputs from rule definition
phenotype_scaled = validate_dtypes(
    pd.read_parquet(snakemake.input.scaled_phenotype_positions)
)
sbs_positions = validate_dtypes(pd.read_parquet(snakemake.input.sbs_positions))
alignment_params = validate_dtypes(pd.read_parquet(snakemake.input.alignment_params))
phenotype_transformed = validate_dtypes(
    pd.read_parquet(snakemake.input.transformed_phenotype_positions)
)

plate = snakemake.params.plate
well = snakemake.params.well
threshold = snakemake.params.threshold

print(f"Processing Plate {plate}, Well {well}")
print(
    f"Input: {len(phenotype_scaled):,} phenotype cells, {len(sbs_positions):,} SBS cells"
)
print(f"Distance threshold: {threshold} px")


def create_empty_outputs_with_summary(failure_reason: str):
    """Create empty output files with failure summary."""
    print(f"Creating empty outputs due to: {failure_reason}")

    # Create empty DataFrame with expected columns
    empty_matches = pd.DataFrame(
        columns=[
            "plate",
            "well",
            "site",
            "tile",
            "cell_0",
            "cell_1",
            "i_0",
            "j_0",
            "i_1",
            "j_1",
            "area_0",
            "area_1",
            "distance",
            "stitched_cell_id_0",
            "stitched_cell_id_1",
        ]
    )

    # Save empty outputs
    empty_matches.to_parquet(str(snakemake.output.raw_matches))
    empty_matches.to_parquet(str(snakemake.output.merged_cells))

    # Create failure summary in wide format
    failure_summary = {
        "plate": plate,
        "well": well,
        "status": "failed",
        "failure_reason": failure_reason,
        "distance_threshold_pixels": float(threshold),
        "phenotype_cells_before_filtering": len(phenotype_scaled),
        "sbs_cells_before_filtering": len(sbs_positions),
        "phenotype_cells_after_filtering": 0,
        "sbs_cells_after_filtering": 0,
        "phenotype_tiles_removed": 0,
        "sbs_tiles_removed": 0,
        "alignment_approach": "unknown",
        "alignment_transformation_type": "unknown",
        "alignment_score": 0.0,
        "alignment_determinant": 1.0,
        "raw_matches_found": 0,
        "mean_match_distance": 0.0,
        "max_match_distance": 0.0,
        "matches_under_5px": 0,
        "matches_under_10px": 0,
        "match_rate_phenotype": 0.0,
        "match_rate_sbs": 0.0,
        "validation_status": "failed",
    }

    summary_df = pd.DataFrame([failure_summary])
    summary_df.to_csv(str(snakemake.output.merge_summary), sep="\t", index=False)
    return


# Verify required ID columns exist
if "stitched_cell_id" not in phenotype_scaled.columns:
    print(f"❌ ERROR: Missing 'stitched_cell_id' in phenotype data")
    create_empty_outputs_with_summary("missing_stitched_id_phenotype")
elif "stitched_cell_id" not in sbs_positions.columns:
    print(f"❌ ERROR: Missing 'stitched_cell_id' in SBS data")
    create_empty_outputs_with_summary("missing_stitched_id_sbs")
else:
    # Step 1: Filter tiles by diversity
    print("Applying tile diversity filtering...")

    phenotype_filtered = filter_tiles_by_diversity(phenotype_scaled, "Phenotype")
    sbs_filtered = filter_tiles_by_diversity(sbs_positions, "SBS")

    if len(phenotype_filtered) == 0:
        print("❌ ERROR: No phenotype cells remain after tile filtering")
        create_empty_outputs_with_summary("no_phenotype_after_filtering")
    elif len(sbs_filtered) == 0:
        print("❌ ERROR: No SBS cells remain after tile filtering")
        create_empty_outputs_with_summary("no_sbs_after_filtering")
    else:
        # Filter transformed positions to match filtered scaled positions
        if len(phenotype_transformed) != len(phenotype_scaled):
            print(
                "⚠️  WARNING: Transformed positions length doesn't match scaled positions"
            )

        phenotype_transformed_filtered = phenotype_transformed[
            phenotype_transformed.index.isin(phenotype_filtered.index)
        ]

        print(
            f"After filtering: {len(phenotype_filtered):,} phenotype cells, {len(sbs_filtered):,} SBS cells"
        )

        # Load alignment parameters
        print("Loading alignment parameters...")

        if len(alignment_params) == 0:
            print("❌ ERROR: No alignment parameters found")
            create_empty_outputs_with_summary("no_alignment_params")
        else:
            alignment = load_alignment_parameters(alignment_params.iloc[0])

            print(
                f"Using translation: [{alignment['translation'][0]:.1f}, {alignment['translation'][1]:.1f}], rotation det: {alignment.get('determinant', 1):.3f}"
            )
            print(
                f"Using {alignment.get('approach', 'unknown')} alignment "
                f"(score: {alignment.get('score', 0):.3f}, "
                f"det: {alignment.get('determinant', 1):.3f})"
            )

            # Find cell matches using stitched_cell_id
            print("Finding cell matches...")

            try:
                raw_matches, summary_stats = find_cell_matches(
                    phenotype_positions=phenotype_filtered,
                    sbs_positions=sbs_filtered,
                    alignment=alignment,
                    threshold=threshold,
                    chunk_size=50000,
                    transformed_phenotype_positions=phenotype_transformed_filtered,
                )

                if raw_matches.empty:
                    print("❌ ERROR: No cell matches found")
                    create_empty_outputs_with_summary("no_matches_found")
                else:
                    print(
                        f"Found {len(raw_matches):,} raw matches "
                        f"(mean distance: {raw_matches['distance'].mean():.1f}px)"
                    )

                    # Step 3: Extract metadata directly from matched rows
                    print("Extracting metadata from matched rows...")

                    # Get phenotype and SBS indices that were matched
                    pheno_indices = []
                    sbs_indices = []

                    # The matching function returns stitched_cell_ids in cell_0 and cell_1
                    # We need to find the corresponding row indices
                    pheno_stitched_to_idx = (
                        phenotype_filtered.reset_index()
                        .set_index("stitched_cell_id")["index"]
                        .to_dict()
                    )
                    sbs_stitched_to_idx = (
                        sbs_filtered.reset_index()
                        .set_index("stitched_cell_id")["index"]
                        .to_dict()
                    )

                    # Map stitched IDs back to row indices
                    for _, match in raw_matches.iterrows():
                        pheno_idx = pheno_stitched_to_idx.get(match["cell_0"])
                        sbs_idx = sbs_stitched_to_idx.get(match["cell_1"])

                        if pheno_idx is not None and sbs_idx is not None:
                            pheno_indices.append(pheno_idx)
                            sbs_indices.append(sbs_idx)
                        else:
                            print(
                                f"⚠️  WARNING: Could not find indices for match {match['cell_0']} -> {match['cell_1']}"
                            )

                    if len(pheno_indices) != len(raw_matches):
                        print(
                            f"⚠️  WARNING: Index mapping incomplete: {len(pheno_indices)}/{len(raw_matches)} matches mapped"
                        )

                        # Filter raw_matches to only include successfully mapped ones
                        successfully_mapped = min(len(pheno_indices), len(sbs_indices))
                        raw_matches = raw_matches.iloc[:successfully_mapped].copy()
                        pheno_indices = pheno_indices[:successfully_mapped]
                        sbs_indices = sbs_indices[:successfully_mapped]

                    # Extract metadata directly from the matched rows
                    phenotype_match_rows = phenotype_filtered.loc[pheno_indices]
                    sbs_match_rows = sbs_filtered.loc[sbs_indices]

                    # Build final output with direct extraction
                    final_matches = pd.DataFrame(
                        {
                            "plate": plate,
                            "well": well,
                            # Site from SBS data (tile -> site mapping as per existing logic)
                            "site": sbs_match_rows["tile"].values,
                            # Tile from phenotype data
                            "tile": phenotype_match_rows["tile"].values,
                            # Cell IDs: original_cell_id for downstream merging
                            "cell_0": phenotype_match_rows["original_cell_id"].values,
                            "cell_1": sbs_match_rows["original_cell_id"].values,
                            # Coordinates from matches (already transformed)
                            "i_0": raw_matches["i_0"].values,
                            "j_0": raw_matches["j_0"].values,
                            "i_1": raw_matches["i_1"].values,
                            "j_1": raw_matches["j_1"].values,
                            # Areas if available
                            "area_0": phenotype_match_rows["area"].values
                            if "area" in phenotype_match_rows.columns
                            else np.nan,
                            "area_1": sbs_match_rows["area"].values
                            if "area" in sbs_match_rows.columns
                            else np.nan,
                            # Distance from matching
                            "distance": raw_matches["distance"].values,
                            # Stitched IDs for reference
                            "stitched_cell_id_0": phenotype_match_rows[
                                "stitched_cell_id"
                            ].values,
                            "stitched_cell_id_1": sbs_match_rows[
                                "stitched_cell_id"
                            ].values,
                        }
                    )

                    # Ensure proper data types
                    final_matches["site"] = final_matches["site"].astype(int)
                    final_matches["tile"] = final_matches["tile"].astype(int)

                    print(
                        f"Successfully extracted metadata for {len(final_matches):,} matches"
                    )

                    # Save outputs - using named outputs
                    print("Saving results...")

                    final_matches.to_parquet(str(snakemake.output.raw_matches))
                    final_matches.to_parquet(str(snakemake.output.merged_cells))

                    # Validate matches
                    print("Validating matches...")
                    validation_results = validate_matches(final_matches)

                    if validation_results.get("status") == "valid":
                        if validation_results.get("quality_flags", {}).get(
                            "good_quality", False
                        ):
                            print("✅ Match quality: GOOD")
                        else:
                            print("⚠️  Match quality: ACCEPTABLE")

                        # Report key quality metrics
                        dist_stats = validation_results.get("distance_stats", {})
                        dist_dist = validation_results.get("distance_distribution", {})

                        print(
                            f"Distance stats: mean={dist_stats.get('mean', 0):.1f}px, "
                            f"max={dist_stats.get('max', 0):.1f}px"
                        )
                        print(
                            f"Quality distribution: "
                            f"{dist_dist.get('under_5px', 0)} under 5px, "
                            f"{dist_dist.get('under_10px', 0)} under 10px"
                        )

                        # Warn about potential issues
                        if validation_results.get("duplication", {}).get(
                            "has_duplicates", False
                        ):
                            pheno_dups = validation_results["duplication"][
                                "phenotype_duplicates"
                            ]
                            sbs_dups = validation_results["duplication"][
                                "sbs_duplicates"
                            ]
                            print(
                                f"⚠️  WARNING: Found duplicates (phenotype: {pheno_dups}, SBS: {sbs_dups})"
                            )

                        if dist_dist.get("over_20px", 0) > 0:
                            print(
                                f"⚠️  WARNING: {dist_dist['over_20px']} matches have distance >20px"
                            )
                    else:
                        print(
                            f"❌ WARNING: Match validation failed: {validation_results.get('status', 'unknown')}"
                        )

                    # Create comprehensive summary in wide format
                    summary_dict = {
                        "plate": plate,
                        "well": well,
                        "status": "success",
                        "distance_threshold_pixels": float(threshold),
                        
                        # Input data metrics
                        "phenotype_cells_before_filtering": len(phenotype_scaled),
                        "sbs_cells_before_filtering": len(sbs_positions),
                        "phenotype_cells_after_filtering": len(phenotype_filtered),
                        "sbs_cells_after_filtering": len(sbs_filtered),
                        
                        # Tile filtering metrics
                        "phenotype_tiles_removed": (
                            len(phenotype_scaled["tile"].unique()) - len(phenotype_filtered["tile"].unique())
                            if "tile" in phenotype_scaled.columns else 0
                        ),
                        "sbs_tiles_removed": (
                            len(sbs_positions["tile"].unique()) - len(sbs_filtered["tile"].unique())
                            if "tile" in sbs_positions.columns else 0
                        ),
                        
                        # Alignment information
                        "alignment_approach": str(alignment.get("approach", "unknown")),
                        "alignment_transformation_type": str(alignment.get("transformation_type", "unknown")),
                        "alignment_score": float(alignment.get("score", 0)),
                        "alignment_determinant": float(alignment.get("determinant", 1)),
                        
                        # Matching results
                        "raw_matches_found": len(final_matches),
                        "mean_match_distance": float(final_matches["distance"].mean()) if len(final_matches) > 0 else 0.0,
                        "max_match_distance": float(final_matches["distance"].max()) if len(final_matches) > 0 else 0.0,
                        "matches_under_5px": int((final_matches["distance"] < 5).sum()) if len(final_matches) > 0 else 0,
                        "matches_under_10px": int((final_matches["distance"] < 10).sum()) if len(final_matches) > 0 else 0,
                        "match_rate_phenotype": float(len(final_matches) / len(phenotype_filtered)) if len(phenotype_filtered) > 0 else 0.0,
                        "match_rate_sbs": float(len(final_matches) / len(sbs_filtered)) if len(sbs_filtered) > 0 else 0.0,
                    }

                    # Add validation results if available
                    if validation_results:
                        summary_dict.update({
                            "validation_status": validation_results.get("status", "unknown"),
                            
                            # Distance stats
                            "validation_distance_mean": validation_results.get("distance_stats", {}).get("mean", 0),
                            "validation_distance_median": validation_results.get("distance_stats", {}).get("median", 0),
                            "validation_distance_max": validation_results.get("distance_stats", {}).get("max", 0),
                            "validation_distance_std": validation_results.get("distance_stats", {}).get("std", 0),
                            
                            # Distance distribution
                            "validation_distribution_under_1px": validation_results.get("distance_distribution", {}).get("under_1px", 0),
                            "validation_distribution_under_2px": validation_results.get("distance_distribution", {}).get("under_2px", 0),
                            "validation_distribution_under_5px": validation_results.get("distance_distribution", {}).get("under_5px", 0),
                            "validation_distribution_under_10px": validation_results.get("distance_distribution", {}).get("under_10px", 0),
                            "validation_distribution_over_20px": validation_results.get("distance_distribution", {}).get("over_20px", 0),
                            
                            # Quality flags
                            "validation_quality_has_duplicates": validation_results.get("quality_flags", {}).get("has_duplicates", False),
                            "validation_quality_has_large_distances": validation_results.get("quality_flags", {}).get("has_large_distances", False),
                            "validation_quality_good_quality": validation_results.get("quality_flags", {}).get("good_quality", False),
                            
                            # Duplication info
                            "validation_duplication_phenotype_duplicates": validation_results.get("duplication", {}).get("phenotype_duplicates", 0),
                            "validation_duplication_sbs_duplicates": validation_results.get("duplication", {}).get("sbs_duplicates", 0),
                            "validation_duplication_has_duplicates": validation_results.get("duplication", {}).get("has_duplicates", False),
                        })

                    # Create single-row DataFrame
                    summary_df = pd.DataFrame([summary_dict])
                    summary_df.to_csv(str(snakemake.output.merge_summary), sep="\t", index=False)

                    print(f"✅ Completed successfully!")
                    print(
                        f"Result: {len(final_matches):,} matched cells ready for downstream processing"
                    )

                    # Show final tile distribution
                    if len(final_matches) > 0:
                        unique_tiles = final_matches["tile"].unique()
                        unique_sites = final_matches["site"].unique()
                        print(
                            f"Final tiles represented: {sorted(unique_tiles.tolist())}"
                        )
                        print(
                            f"Final sites represented: {sorted(unique_sites.tolist())}"
                        )

            except Exception as e:
                print(f"❌ ERROR: Cell matching failed: {e}")
                traceback.print_exc()
                create_empty_outputs_with_summary("matching_failed")

print("=== WELL CELL MERGE COMPLETED ===")