"""Step 3: Well Merge Deduplication - 1:1 spatial deduplication using stitched cell IDs.
Production version - operates on stitched cell IDs for proper spatial deduplication
while preserving original cell IDs for downstream processing.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path

from lib.shared.file_utils import validate_dtypes


class DeduplicationError(Exception):
    """Custom exception for deduplication failures."""

    pass


def apply_legacy_compatible_deduplication(raw_matches):
    """Apply legacy-compatible deduplication using stitched cell IDs for spatial accuracy.

    This method:
    1. For each phenotype stitched cell, keeps the best SBS match
    2. For each SBS stitched cell, keeps the best phenotype match
    3. Preserves original cell IDs for downstream compatibility

    Args:
        raw_matches: DataFrame with raw cell matches

    Returns:
        DataFrame with deduplicated matches

    Raises:
        DeduplicationError: If stitched ID columns are missing
    """
    # Require stitched ID columns - no fallback to original IDs
    required_cols = [
        "stitched_cell_id_0",
        "stitched_cell_id_1",
        "cell_0",
        "cell_1",
        "distance",
    ]
    missing_cols = [col for col in required_cols if col not in raw_matches.columns]

    if missing_cols:
        raise DeduplicationError(
            f"Missing required stitched ID columns: {missing_cols}. "
            f"Stitched cell IDs are required for spatial deduplication."
        )

    # Step 1: For each phenotype stitched cell, keep best SBS match
    df_pheno_deduped = raw_matches.sort_values(
        "distance", ascending=True
    ).drop_duplicates("stitched_cell_id_0", keep="first")

    # Step 2: For each SBS stitched cell, keep best phenotype match
    df_final = df_pheno_deduped.sort_values("distance", ascending=True).drop_duplicates(
        "stitched_cell_id_1", keep="first"
    )

    return df_final


def validate_final_matches(final_matches):
    """Validate final matches for spatial accuracy and quality.

    Args:
        final_matches: DataFrame with final cell matches

    Returns:
        dict: Validation results including 1:1 mapping status
    """
    if final_matches.empty:
        return {
            "is_1to1_stitched": True,  # Technically true for empty set
            "match_count": 0,
            "validation_status": "empty",
        }

    # Check for stitched ID duplicates (should be 1:1)
    stitched_pheno_dups = final_matches["stitched_cell_id_0"].duplicated().sum()
    stitched_sbs_dups = final_matches["stitched_cell_id_1"].duplicated().sum()
    is_1to1_stitched = stitched_pheno_dups == 0 and stitched_sbs_dups == 0

    # Check original ID duplicates (may exist - one original cell can map to multiple stitched positions)
    original_pheno_dups = final_matches["cell_0"].duplicated().sum()
    original_sbs_dups = final_matches["cell_1"].duplicated().sum()

    # Distance statistics
    distances = final_matches["distance"]

    return {
        "match_count": len(final_matches),
        "is_1to1_stitched": is_1to1_stitched,
        "stitched_duplicates": {
            "phenotype": int(stitched_pheno_dups),
            "sbs": int(stitched_sbs_dups),
        },
        "original_duplicates": {
            "phenotype": int(original_pheno_dups),
            "sbs": int(original_sbs_dups),
        },
        "distance_stats": {
            "mean": float(distances.mean()),
            "median": float(distances.median()),
            "max": float(distances.max()),
            "std": float(distances.std()),
        },
        "validation_status": "valid"
        if is_1to1_stitched
        else "stitched_duplicates_present",
    }


def validate_quality(matches):
    """Validate match quality and return summary statistics.

    Args:
        matches: DataFrame with final matches

    Returns:
        dict: Quality metrics
    """
    if matches.empty:
        return {"match_count": 0, "mean_distance": 0.0, "quality_tier": "empty"}

    distances = matches["distance"]

    # Calculate distance distribution
    under_5px = (distances < 5).sum()
    under_10px = (distances < 10).sum()
    over_50px = (distances > 50).sum()

    # Determine quality tier
    precision_5px = under_5px / len(distances)
    mean_dist = distances.mean()

    if mean_dist < 2.0 and precision_5px > 0.8 and over_50px == 0:
        quality_tier = "excellent"
    elif mean_dist < 5.0 and precision_5px > 0.6 and over_50px == 0:
        quality_tier = "good"
    elif over_50px == 0:
        quality_tier = "acceptable"
    else:
        quality_tier = "poor"

    return {
        "match_count": len(matches),
        "mean_distance": float(mean_dist),
        "median_distance": float(distances.median()),
        "max_distance": float(distances.max()),
        "precision_5px": float(precision_5px),
        "precision_10px": float(under_10px / len(distances)),
        "large_distance_count": int(over_50px),
        "quality_tier": quality_tier,
    }


def prepare_output_format(matches):
    """Prepare final output with correct column ordering and validation.

    Args:
        matches: DataFrame with deduplicated matches

    Returns:
        DataFrame: Properly formatted output

    Raises:
        DeduplicationError: If required columns are missing
    """
    # Required columns for downstream processing
    required_columns = [
        "plate",
        "well",
        "site",
        "tile",
        "cell_0",
        "i_0",
        "j_0",
        "cell_1",
        "i_1",
        "j_1",
        "distance",
    ]
    optional_columns = ["area_0", "area_1", "stitched_cell_id_0", "stitched_cell_id_1"]

    # Check for missing required columns
    missing_required = [col for col in required_columns if col not in matches.columns]
    if missing_required:
        raise DeduplicationError(f"Missing required output columns: {missing_required}")

    # Select columns for output
    output_columns = [col for col in required_columns if col in matches.columns]
    output_columns.extend([col for col in optional_columns if col in matches.columns])

    return matches[output_columns].copy()


def main():
    """Main deduplication processing function."""
    try:
        # Load inputs
        raw_matches = validate_dtypes(pd.read_parquet(snakemake.input.raw_matches))
        merged_cells = validate_dtypes(pd.read_parquet(snakemake.input.merged_cells))

        plate = snakemake.params.plate
        well = snakemake.params.well

        print(
            f"Processing Well {plate}-{well}: {len(raw_matches):,} raw matches → {len(merged_cells):,} simple matches"
        )

        # Early exit for empty input
        if raw_matches.empty:
            raise DeduplicationError("No raw matches to process")

        # Apply legacy-compatible deduplication
        final_matches = apply_legacy_compatible_deduplication(raw_matches)

        if final_matches.empty:
            raise DeduplicationError("Deduplication eliminated all matches")

        # Validate final matches
        validation_results = validate_final_matches(final_matches)

        # Validate quality
        quality_metrics = validate_quality(final_matches)

        # Prepare output format
        final_output = prepare_output_format(final_matches)

        # Report results
        stitched_status = (
            "✅ 1:1 stitched mapping"
            if validation_results["is_1to1_stitched"]
            else "⚠️  Stitched duplicates present"
        )
        print(
            f"Deduplication complete: {len(final_output):,} matches ({quality_metrics['quality_tier']} quality)"
        )
        print(
            f"Spatial validation: {stitched_status}, Mean distance: {quality_metrics['mean_distance']:.2f}px, "
            f"<5px precision: {quality_metrics['precision_5px']:.1%}"
        )

        # Warn about poor quality matches
        if quality_metrics["large_distance_count"] > 0:
            print(
                f"⚠️  {quality_metrics['large_distance_count']} matches >50px may indicate alignment issues"
            )

        # Save output
        final_output.to_parquet(str(snakemake.output.deduplicated_cells))

        # Create summary
        summary = {
            "status": "success",
            "plate": plate,
            "well": well,
            "processing": {
                "raw_matches_input": len(raw_matches),
                "simple_matches_input": len(merged_cells),
                "final_matches_output": len(final_output),
                "matches_removed": len(raw_matches) - len(final_output),
                "efficiency": len(final_output) / len(raw_matches),
            },
            "deduplication": {
                "method": "legacy_on_stitched_ids",
                "uses_stitched_ids": True,
                "preserves_original_ids": True,
                "achieved_1to1_stitched": validation_results["is_1to1_stitched"],
            },
            "validation": validation_results,
            "quality": quality_metrics,
            "output_format": {
                "columns": list(final_output.columns),
                "ready_for_format_merge": True,
            },
        }

        with open(str(snakemake.output.deduplication_summary), "w") as f:
            yaml.dump(summary, f, default_flow_style=False)

        print(f"✅ Well {plate}-{well} deduplication successful")

    except DeduplicationError as e:
        print(f"❌ Deduplication failed for well {plate}-{well}: {e}")
        create_empty_output(str(e))
        raise

    except Exception as e:
        print(f"❌ Unexpected error in well {plate}-{well}: {e}")
        create_empty_output(f"Unexpected error: {e}")
        raise


def create_empty_output(error_message):
    """Create empty output files when processing fails."""
    # Create empty DataFrame with correct schema
    empty_df = pd.DataFrame(
        columns=[
            "plate",
            "well",
            "site",
            "tile",
            "cell_0",
            "i_0",
            "j_0",
            "area_0",
            "cell_1",
            "i_1",
            "j_1",
            "area_1",
            "distance",
            "stitched_cell_id_0",
            "stitched_cell_id_1",
        ]
    )

    # Add minimal required data
    empty_df["plate"] = snakemake.params.plate
    empty_df["well"] = snakemake.params.well
    empty_df["site"] = 1
    empty_df["tile"] = 1

    empty_df.to_parquet(str(snakemake.output.deduplicated_cells))

    # Create failure summary
    summary = {
        "status": "failed",
        "plate": snakemake.params.plate,
        "well": snakemake.params.well,
        "error": error_message,
        "processing": {"final_matches_output": 0},
        "deduplication": {
            "method": "legacy_on_stitched_ids",
            "achieved_1to1_stitched": False,
        },
        "output_format": {"ready_for_format_merge": False},
    }

    with open(str(snakemake.output.deduplication_summary), "w") as f:
        yaml.dump(summary, f, default_flow_style=False)


if __name__ == "__main__":
    main()
