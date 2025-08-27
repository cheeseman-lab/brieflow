"""Well Merge Deduplication Script.

Performs 1:1 spatial deduplication using stitched cell IDs for well-based merge processing.
This script operates on raw cell matches to produce spatially accurate deduplicated matches
while preserving original cell IDs for downstream processing compatibility.

The deduplication process ensures that each stitched cell ID appears at most once in the
final matches, creating a proper 1:1 spatial mapping between phenotype and SBS datasets.

Input:
    - raw_matches.parquet: Raw cell matches with potential duplicates
    - merged_cells.parquet: Simple merged cells (for comparison/validation)

Output:  
    - deduplicated_cells.parquet: Spatially deduplicated matches
    - dedup_summary.yaml: Comprehensive processing and quality metrics

This script is designed for the well-based merge approach and requires stitched cell IDs
for proper spatial deduplication.
"""

import pandas as pd
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from lib.shared.file_utils import validate_dtypes
from lib.merge.well_deduplication import validate_final_matches, legacy_deduplication_stitched_ids


class DeduplicationError(Exception):
    """Custom exception for deduplication processing failures.
    
    Raised when critical errors occur during deduplication that prevent
    successful processing, such as missing required columns or empty
    input data.
    """
    pass


def apply_legacy_compatible_deduplication(raw_matches: pd.DataFrame) -> pd.DataFrame:
    """Apply spatial deduplication using stitched cell IDs with legacy compatibility.
    
    Wrapper function that calls the library implementation. Maintained for
    backwards compatibility with existing code that calls this function directly.
    
    Args:
        raw_matches: DataFrame containing raw cell matches
        
    Returns:
        DataFrame with deduplicated matches
        
    Raises:
        DeduplicationError: If required stitched ID columns are missing
    """
    try:
        return legacy_deduplication_stitched_ids(raw_matches)
    except ValueError as e:
        # Convert library ValueError to our DeduplicationError for consistency
        raise DeduplicationError(str(e))


# validate_final_matches and validate_quality functions removed - imported from library


def prepare_output_format(matches: pd.DataFrame) -> pd.DataFrame:
    """Prepare final output with correct column ordering and validation.

    Ensures the output DataFrame has the required columns in the expected
    format for downstream processing, while preserving optional columns
    that may be useful for analysis.

    Args:
        matches: DataFrame containing deduplicated matches

    Returns:
        DataFrame with properly formatted output columns

    Raises:
        DeduplicationError: If required columns are missing from input

    Note:
        Required columns are needed for downstream processing compatibility.
        Optional columns (like area measurements) are preserved if present.
    """
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
    missing_required = [col for col in required_columns if col not in matches.columns]
    if missing_required:
        raise DeduplicationError(f"Missing required output columns: {missing_required}")

    # Select columns for output (required + available optional)
    output_columns = [col for col in required_columns if col in matches.columns]
    output_columns.extend([col for col in optional_columns if col in matches.columns])

    return matches[output_columns].copy()


def create_empty_output(error_message: str) -> None:
    """Create empty output files when processing fails.
    
    Generates properly formatted empty output files with minimal required
    data structure to maintain pipeline compatibility when deduplication fails.

    Args:
        error_message: Description of the failure for logging/debugging
        
    Note:
        Uses snakemake.output and snakemake.params which are available
        in the snakemake execution environment.
    """
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
    empty_df["plate"] = snakemake.params.plate
    empty_df["well"] = snakemake.params.well
    empty_df["site"] = 1
    empty_df["tile"] = 1

    empty_df.to_parquet(str(snakemake.output.deduplicated_cells))

    # Create failure summary for debugging and pipeline monitoring
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


def main() -> None:
    """Main deduplication processing function.
    
    Orchestrates the complete deduplication workflow:
    1. Load and validate input data
    2. Apply spatial deduplication using stitched cell IDs
    3. Validate results for quality and spatial accuracy
    4. Prepare and save formatted output
    5. Generate comprehensive summary report
    
    This function handles the complete processing pipeline and provides
    detailed logging and error handling for production use.
    
    Raises:
        DeduplicationError: For recoverable deduplication-specific errors
        Exception: For unexpected system errors
        
    Note:
        Uses snakemake object which provides input/output paths and parameters
        in the snakemake execution environment.
    """
    try:
        # Load and validate input data
        raw_matches = validate_dtypes(pd.read_parquet(snakemake.input.raw_matches))
        merged_cells = validate_dtypes(pd.read_parquet(snakemake.input.merged_cells))

        plate = snakemake.params.plate
        well = snakemake.params.well

        print(
            f"Processing Well {plate}-{well}: "
            f"{len(raw_matches):,} raw matches → {len(merged_cells):,} simple matches"
        )

        # Early validation - ensure we have data to process
        if raw_matches.empty:
            raise DeduplicationError("No raw matches to process")

        # Apply spatial deduplication using stitched cell IDs
        final_matches = apply_legacy_compatible_deduplication(raw_matches)

        if final_matches.empty:
            raise DeduplicationError("Deduplication eliminated all matches")

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
            "large_distance_count": validation_results["distance_distribution"]["over_50px"],
            "quality_tier": validation_results["quality_metrics"]["quality_tier"],
        }

        # Prepare output in the correct format for downstream processing
        final_output = prepare_output_format(final_matches)

        # Generate user-friendly status reporting
        stitched_status = (
            "✅ 1:1 stitched mapping"
            if validation_results["is_1to1_stitched"]
            else "⚠️  Stitched duplicates present"
        )
        
        print(
            f"Deduplication complete: {len(final_output):,} matches "
            f"({quality_metrics['quality_tier']} quality)"
        )
        print(
            f"Spatial validation: {stitched_status}, "
            f"Mean distance: {quality_metrics['mean_distance']:.2f}px, "
            f"<5px precision: {quality_metrics['precision_5px']:.1%}"
        )

        # Warn about potential alignment issues
        if quality_metrics["large_distance_count"] > 0:
            print(
                f"⚠️  {quality_metrics['large_distance_count']} matches >50px "
                f"may indicate alignment issues"
            )

        # Save deduplicated results
        final_output.to_parquet(str(snakemake.output.deduplicated_cells))

        # Create comprehensive summary for pipeline monitoring and debugging
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


if __name__ == "__main__":
    main()