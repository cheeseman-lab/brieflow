"""Well Merge Deduplication Script.

Performs 1:1 spatial deduplication using stitched cell IDs for well-based merge processing.
This script operates on raw cell matches to produce spatially accurate deduplicated matches
while preserving original cell IDs for downstream processing compatibility.

The deduplication process ensures that each stitched cell ID appears at most once in the
final matches, creating a proper 1:1 spatial mapping between phenotype and SBS datasets.

Input:
    - raw_matches: From well_cell_merge rule output [0]
    - merged_cells: From well_cell_merge rule output [1] (for comparison/validation)

Output:
    - deduplicated_cells: well_merge_deduplicate output [0] - spatially deduplicated matches
    - dedup_summary: well_merge_deduplicate output [1] - comprehensive processing metrics

This script is designed for the well-based merge approach and requires stitched cell IDs
for proper spatial deduplication.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from lib.shared.file_utils import validate_dtypes
from lib.merge.well_deduplication import (
    validate_final_matches,
    deduplicate_matches_by_stitched_ids,
)

print("=== WELL MERGE DEDUPLICATION ===")

# Load and validate input data - using named inputs from rule
try:
    raw_matches = validate_dtypes(pd.read_parquet(snakemake.input.raw_matches))
    merged_cells = validate_dtypes(pd.read_parquet(snakemake.input.merged_cells))

    plate = snakemake.params.plate
    well = snakemake.params.well

    print(f"Processing Well {plate}-{well}")
    print(
        f"Input: {len(raw_matches):,} raw matches → {len(merged_cells):,} simple matches"
    )

    def create_empty_outputs_with_summary(error_message: str):
        """Create empty output files with failure summary."""
        print(f"❌ Deduplication failed for well {plate}-{well}: {error_message}")

        # Create empty DataFrame with correct schema for downstream compatibility
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

        # Add minimal required metadata
        empty_df["plate"] = plate
        empty_df["well"] = well
        empty_df["site"] = 1
        empty_df["tile"] = 1

        # Save to named outputs
        empty_df.to_parquet(str(snakemake.output.deduplicated_cells))

        # Create failure summary as TSV (key-value format for aggregation script)
        summary_data = [
            ["status", "failed"],
            ["plate", plate],
            ["well", well],
            ["error", error_message],
            ["processing_final_matches_output", 0],
            ["deduplication_method", "deduplicate_matches_by_stitched_ids"],
            ["deduplication_achieved_1to1_stitched", False],
            ["output_format_ready_for_format_merge", False],
        ]

        summary_df = pd.DataFrame(summary_data, columns=["metric", "value"])
        summary_df.to_csv(
            str(snakemake.output.deduplication_summary), sep="\t", index=False
        )
        return

    # Early validation - ensure we have data to process
    if raw_matches.empty:
        create_empty_outputs_with_summary("No raw matches to process")
        print("⚠️  No matches to deduplicate - creating empty outputs")
        exit(0)  # Not an error, just no data to process

    # Apply spatial deduplication using stitched cell IDs
    try:
        final_matches = deduplicate_matches_by_stitched_ids(raw_matches)
    except ValueError as e:
        create_empty_outputs_with_summary(str(e))
        raise RuntimeError(f"Deduplication failed: {e}")

    if final_matches.empty:
        create_empty_outputs_with_summary("Deduplication eliminated all matches")
        print("⚠️  All matches were eliminated during deduplication")
        exit(0)  # Not necessarily an error, might be due to poor alignment

    # Validate final matches for spatial accuracy and quality
    validation_results = validate_final_matches(final_matches)

    # Extract quality metrics from comprehensive validation
    quality_metrics = {
        "match_count": validation_results["match_count"],
        "mean_distance": validation_results["distance_stats"]["mean"],
        "median_distance": validation_results["distance_stats"]["median"],
        "max_distance": validation_results["distance_stats"]["max"],
        "precision_5px": validation_results["quality_metrics"]["precision_5px"],
        "precision_10px": validation_results["quality_metrics"]["precision_10px"],
        "large_distance_count": validation_results["distance_distribution"][
            "over_50px"
        ],
        "quality_tier": validation_results["quality_metrics"]["quality_tier"],
    }

    # Prepare final output with correct column ordering and validation
    # Required columns for downstream processing compatibility
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
    # Optional columns that are preserved if present
    optional_columns = ["area_0", "area_1", "stitched_cell_id_0", "stitched_cell_id_1"]

    # Check for missing required columns
    missing_required = [
        col for col in required_columns if col not in final_matches.columns
    ]
    if missing_required:
        create_empty_outputs_with_summary(
            f"Missing required output columns: {missing_required}"
        )
        raise RuntimeError(f"Missing required columns: {missing_required}")

    # Select columns for output (required + available optional)
    output_columns = [col for col in required_columns if col in final_matches.columns]
    output_columns.extend(
        [col for col in optional_columns if col in final_matches.columns]
    )

    final_output = final_matches[output_columns].copy()

    # Generate user-friendly status reporting
    stitched_status = (
        "✅ 1:1 stitched mapping"
        if validation_results["is_1to1_stitched"]
        else "⚠️  Stitched duplicates present"
    )

    print(
        f"Deduplication complete: {len(final_output):,} matches ({quality_metrics['quality_tier']} quality)"
    )
    print(f"Spatial validation: {stitched_status}")
    print(
        f"Quality metrics: mean distance {quality_metrics['mean_distance']:.2f}px, <5px precision {quality_metrics['precision_5px']:.1%}"
    )

    # Warn about potential alignment issues
    if quality_metrics["large_distance_count"] > 0:
        print(
            f"⚠️  {quality_metrics['large_distance_count']} matches >50px may indicate alignment issues"
        )

    # Save deduplicated results - using named outputs
    final_output.to_parquet(str(snakemake.output.deduplicated_cells))
    print(f"✅ Saved deduplicated cells: {snakemake.output.deduplicated_cells}")

    # Create comprehensive summary as TSV for pipeline monitoring and debugging
    summary_data = []

    # Basic information
    summary_data.extend(
        [
            ["status", "success"],
            ["plate", plate],
            ["well", well],
        ]
    )

    # Processing metrics
    summary_data.extend(
        [
            ["processing_raw_matches_input", len(raw_matches)],
            ["processing_simple_matches_input", len(merged_cells)],
            ["processing_final_matches_output", len(final_output)],
            ["processing_matches_removed", len(raw_matches) - len(final_output)],
            [
                "processing_efficiency",
                float(len(final_output) / len(raw_matches))
                if len(raw_matches) > 0
                else 0.0,
            ],
        ]
    )

    # Deduplication information
    summary_data.extend(
        [
            ["deduplication_method", "deduplicate_matches_by_stitched_ids"],
            ["deduplication_uses_stitched_ids", True],
            ["deduplication_preserves_original_ids", True],
            [
                "deduplication_achieved_1to1_stitched",
                validation_results["is_1to1_stitched"],
            ],
        ]
    )

    # Validation results - flatten the nested dictionary
    summary_data.append(
        ["validation_is_1to1_stitched", validation_results["is_1to1_stitched"]]
    )
    summary_data.append(["validation_match_count", validation_results["match_count"]])

    # Distance statistics
    dist_stats = validation_results.get("distance_stats", {})
    for key, value in dist_stats.items():
        summary_data.append([f"validation_distance_{key}", value])

    # Distance distribution
    dist_dist = validation_results.get("distance_distribution", {})
    for key, value in dist_dist.items():
        summary_data.append([f"validation_distribution_{key}", value])

    # Quality metrics
    qual_metrics = validation_results.get("quality_metrics", {})
    for key, value in qual_metrics.items():
        summary_data.append([f"validation_quality_{key}", value])

    # Duplication check
    dup_check = validation_results.get("duplication_check", {})
    for key, value in dup_check.items():
        summary_data.append([f"validation_duplication_{key}", value])

    # Quality metrics (extracted earlier)
    for key, value in quality_metrics.items():
        summary_data.append([f"quality_{key}", value])

    # Output format information
    summary_data.extend(
        [
            ["output_format_columns", ";".join(final_output.columns)],
            ["output_format_ready_for_format_merge", True],
        ]
    )

    summary_df = pd.DataFrame(summary_data, columns=["metric", "value"])
    summary_df.to_csv(
        str(snakemake.output.deduplication_summary), sep="\t", index=False
    )
    print(f"✅ Saved deduplication summary: {snakemake.output.deduplication_summary}")

    print(f"✅ Well {plate}-{well} deduplication completed successfully")
    print(
        f"Final result: {len(final_output):,} deduplicated matches ready for downstream processing"
    )

except Exception as e:
    error_message = f"Unexpected error: {e}"
    print(f"❌ Unexpected error in well {plate}-{well}: {e}")

    # Use fallback values if snakemake params not available
    try:
        plate_val = snakemake.params.plate
        well_val = snakemake.params.well
    except:
        plate_val = "unknown"
        well_val = "unknown"

    # Create empty DataFrame with correct schema for downstream compatibility
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

    # Add minimal required metadata
    empty_df["plate"] = plate_val
    empty_df["well"] = well_val
    empty_df["site"] = 1
    empty_df["tile"] = 1

    empty_df.to_parquet(str(snakemake.output.deduplicated_cells))

    # Create failure summary as TSV (key-value format for aggregation script)
    summary_data = [
        ["status", "failed"],
        ["plate", plate_val],
        ["well", well_val],
        ["error", error_message],
        ["processing_final_matches_output", 0],
        ["deduplication_method", "deduplicate_matches_by_stitched_ids"],
        ["deduplication_achieved_1to1_stitched", False],
        ["output_format_ready_for_format_merge", False],
    ]

    summary_df = pd.DataFrame(summary_data, columns=["metric", "value"])
    summary_df.to_csv(
        str(snakemake.output.deduplication_summary), sep="\t", index=False
    )
    raise

print("=== WELL MERGE DEDUPLICATION COMPLETED ===")
