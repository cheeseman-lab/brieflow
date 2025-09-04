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
    - dedup_summary.tsv: Comprehensive processing and quality metrics

This script is designed for the well-based merge approach and requires stitched cell IDs
for proper spatial deduplication.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

from lib.shared.file_utils import validate_dtypes
from lib.merge.well_deduplication import validate_final_matches, deduplicate_matches_by_stitched_ids

# Load and validate input data
try:
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
        error_message = "No raw matches to process"
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
        summary_df.to_csv(str(snakemake.output.deduplication_summary), sep='\t', index=False)
        raise RuntimeError(error_message)
    
    # Apply spatial deduplication using stitched cell IDs
    try:
        final_matches = deduplicate_matches_by_stitched_ids(raw_matches)
    except ValueError as e:
        error_message = str(e)
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
        summary_df.to_csv(str(snakemake.output.deduplication_summary), sep='\t', index=False)
        raise RuntimeError(error_message)
    
    if final_matches.empty:
        error_message = "Deduplication eliminated all matches"
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
        summary_df.to_csv(str(snakemake.output.deduplication_summary), sep='\t', index=False)
        raise RuntimeError(error_message)
    
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
    missing_required = [col for col in required_columns if col not in final_matches.columns]
    if missing_required:
        error_message = f"Missing required output columns: {missing_required}"
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
        summary_df.to_csv(str(snakemake.output.deduplication_summary), sep='\t', index=False)
        raise RuntimeError(error_message)
    
    # Select columns for output (required + available optional)
    output_columns = [col for col in required_columns if col in final_matches.columns]
    output_columns.extend([col for col in optional_columns if col in final_matches.columns])
    
    final_output = final_matches[output_columns].copy()
    
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
    
    # Create comprehensive summary as TSV for pipeline monitoring and debugging
    summary_data = []
    
    # Basic information
    summary_data.extend([
        ["status", "success"],
        ["plate", plate],
        ["well", well],
    ])
    
    # Processing metrics
    summary_data.extend([
        ["processing_raw_matches_input", len(raw_matches)],
        ["processing_simple_matches_input", len(merged_cells)],
        ["processing_final_matches_output", len(final_output)],
        ["processing_matches_removed", len(raw_matches) - len(final_output)],
        ["processing_efficiency", len(final_output) / len(raw_matches)],
    ])
    
    # Deduplication information
    summary_data.extend([
        ["deduplication_method", "deduplicate_matches_by_stitched_ids"],
        ["deduplication_uses_stitched_ids", True],
        ["deduplication_preserves_original_ids", True],
        ["deduplication_achieved_1to1_stitched", validation_results["is_1to1_stitched"]],
    ])
    
    # Validation results - flatten the nested dictionary
    summary_data.append(["validation_is_1to1_stitched", validation_results["is_1to1_stitched"]])
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
    summary_data.extend([
        ["output_format_columns", ";".join(final_output.columns)],
        ["output_format_ready_for_format_merge", True],
    ])
    
    summary_df = pd.DataFrame(summary_data, columns=["metric", "value"])
    summary_df.to_csv(str(snakemake.output.deduplication_summary), sep='\t', index=False)
    
    print(f"✅ Well {plate}-{well} deduplication successful")

except Exception as e:
    error_message = f"Unexpected error: {e}"
    print(f"❌ Unexpected error in well {plate}-{well}: {e}")
    
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
    
    # Create failure summary as TSV (key-value format for aggregation script)
    summary_data = [
        ["status", "failed"],
        ["plate", snakemake.params.plate],
        ["well", snakemake.params.well],
        ["error", error_message],
        ["processing_final_matches_output", 0],
        ["deduplication_method", "deduplicate_matches_by_stitched_ids"],
        ["deduplication_achieved_1to1_stitched", False],
        ["output_format_ready_for_format_merge", False],
    ]
    
    summary_df = pd.DataFrame(summary_data, columns=["metric", "value"])
    summary_df.to_csv(str(snakemake.output.deduplication_summary), sep='\t', index=False)
    raise