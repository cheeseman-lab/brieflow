"""Well Merge Deduplication Script with Comprehensive QC.

Performs 1:1 spatial deduplication using stitched cell IDs for well-based merge processing
with comprehensive QC analysis including matching rates and deduplication statistics.

This script operates on raw cell matches to produce spatially accurate deduplicated matches
while preserving original cell IDs for downstream processing compatibility. It also generates
detailed QC outputs comparable to the tile-based approach.

The deduplication process ensures that each stitched cell ID appears at most once in the
final matches, creating a proper 1:1 spatial mapping between phenotype and SBS datasets.

Input:
    - raw_matches: From well_cell_merge rule output [0]
    - merged_cells: From well_cell_merge rule output [1] (for comparison/validation)
    - sbs_cells: Original SBS cell data for QC analysis
    - phenotype_min_cp: Original phenotype cell data for QC analysis

Output:
    - deduplicated_cells: well_merge_deduplicate output [0] - spatially deduplicated matches
    - dedup_summary: well_merge_deduplicate output [1] - comprehensive processing metrics
    - sbs_matching_rates: well_merge_deduplicate output [2] - SBS cell matching rate analysis
    - phenotype_matching_rates: well_merge_deduplicate output [3] - Phenotype cell matching rate analysis

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
from lib.merge.deduplicate_merge import check_matching_rates
from lib.merge.format_merge import identify_single_gene_mappings

print("=== WELL MERGE DEDUPLICATION WITH QC ===")

# Load and validate input data
try:
    raw_matches = validate_dtypes(pd.read_parquet(snakemake.input.raw_matches))
    merged_cells = validate_dtypes(pd.read_parquet(snakemake.input.merged_cells))
    sbs_cells = validate_dtypes(pd.read_parquet(snakemake.input.sbs_cells))
    phenotype_min_cp = validate_dtypes(
        pd.read_parquet(snakemake.input.phenotype_min_cp)
    )

    plate = snakemake.params.plate
    well = snakemake.params.well

    print(f"Processing Well {plate}-{well}")
    print(
        f"Input: {len(raw_matches):,} raw matches → {len(merged_cells):,} simple matches"
    )
    print(
        f"QC data: {len(sbs_cells):,} SBS cells, {len(phenotype_min_cp):,} phenotype cells"
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
        empty_df["plate"] = plate
        empty_df["well"] = well
        empty_df["site"] = 1
        empty_df["tile"] = 1

        # Save empty deduplicated cells
        empty_df.to_parquet(str(snakemake.output.deduplicated_cells))

        # Create failure summary in wide format
        failure_summary = {
            "plate": plate,
            "well": well,
            "status": "failed",
            "error": error_message,
            "processing_raw_matches_input": len(raw_matches)
            if "raw_matches" in locals()
            else 0,
            "processing_simple_matches_input": len(merged_cells)
            if "merged_cells" in locals()
            else 0,
            "processing_final_matches_output": 0,
            "processing_matches_removed": 0,
            "processing_efficiency": 0.0,
            "qc_sbs_total_cells": 0,
            "qc_sbs_matched_cells": 0,
            "qc_sbs_match_rate": 0.0,
            "qc_phenotype_total_cells": 0,
            "qc_phenotype_matched_cells": 0,
            "qc_phenotype_match_rate": 0.0,
            "deduplication_method": "deduplicate_matches_by_stitched_ids",
            "deduplication_uses_stitched_ids": True,
            "deduplication_preserves_original_ids": True,
            "deduplication_achieved_1to1_stitched": False,
            "validation_is_1to1_stitched": False,
            "validation_match_count": 0,
            "output_format_ready_for_format_merge": False,
            "qc_analysis_completed": False,
            "qc_outputs_generated": False,
        }

        summary_df = pd.DataFrame([failure_summary])
        summary_df.to_csv(
            str(snakemake.output.deduplication_summary), sep="\t", index=False
        )

        # Create empty QC files
        empty_qc = pd.DataFrame({"info": ["No QC data - deduplication failed"]})
        empty_qc.to_csv(str(snakemake.output.sbs_matching_rates), sep="\t", index=False)
        empty_qc.to_csv(
            str(snakemake.output.phenotype_matching_rates), sep="\t", index=False
        )
        return

    # Early validation - ensure we have data to process
    if raw_matches.empty:
        create_empty_outputs_with_summary("No raw matches to process")
        print("⚠️  No matches to deduplicate - creating empty outputs")
        exit(0)

    # STEP 1: Apply spatial deduplication using stitched cell IDs
    try:
        final_matches = deduplicate_matches_by_stitched_ids(raw_matches)
    except ValueError as e:
        create_empty_outputs_with_summary(str(e))
        raise RuntimeError(f"Deduplication failed: {e}")

    if final_matches.empty:
        create_empty_outputs_with_summary("Deduplication eliminated all matches")
        print("⚠️  All matches were eliminated during deduplication")
        exit(0)

    # STEP 2: Validate final matches for spatial accuracy and quality
    validation_results = validate_final_matches(final_matches)
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

    # STEP 3: Prepare final output with correct column ordering
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
    missing_required = [
        col for col in required_columns if col not in final_matches.columns
    ]
    if missing_required:
        create_empty_outputs_with_summary(
            f"Missing required output columns: {missing_required}"
        )
        raise RuntimeError(f"Missing required columns: {missing_required}")

    # Select columns for output
    output_columns = [col for col in required_columns if col in final_matches.columns]
    output_columns.extend(
        [col for col in optional_columns if col in final_matches.columns]
    )
    final_output = final_matches[output_columns].copy()

    # STEP 4: Generate comprehensive QC analysis (similar to tile approach)
    print("\n=== QC ANALYSIS ===")

    # Filter SBS and phenotype data to current well
    well_sbs_cells = sbs_cells[sbs_cells.well == well].copy()
    well_phenotype_cells = phenotype_min_cp[phenotype_min_cp.well == well].copy()

    # Identify single gene mappings for SBS
    well_sbs_cells["mapped_single_gene"] = well_sbs_cells.apply(
        lambda x: identify_single_gene_mappings(x), axis=1
    )

    # Calculate matching rates for SBS cells
    print("Calculating SBS matching rates...")
    try:
        sbs_rates = check_matching_rates(
            well_sbs_cells, final_output, modality="sbs", return_stats=True
        )
        sbs_rates["plate"] = plate
        sbs_rates["well"] = well
        print(f"SBS matching rates calculated: {len(sbs_rates)} entries")
    except Exception as e:
        print(f"⚠️  SBS matching rate calculation failed: {e}")
        sbs_rates = pd.DataFrame(
            {
                "well": [well],
                "total_cells": [len(well_sbs_cells)],
                "matched_cells": [0],
                "match_rate": [0.0],
                "plate": [plate],
                "error": [str(e)],
            }
        )

    # Calculate matching rates for phenotype cells
    print("Calculating phenotype matching rates...")
    try:
        phenotype_rates = check_matching_rates(
            well_phenotype_cells, final_output, modality="phenotype", return_stats=True
        )
        phenotype_rates["plate"] = plate
        phenotype_rates["well"] = well
        print(f"Phenotype matching rates calculated: {len(phenotype_rates)} entries")
    except Exception as e:
        print(f"⚠️  Phenotype matching rate calculation failed: {e}")
        phenotype_rates = pd.DataFrame(
            {
                "well": [well],
                "total_cells": [len(well_phenotype_cells)],
                "matched_cells": [0],
                "match_rate": [0.0],
                "plate": [plate],
                "error": [str(e)],
            }
        )

    # STEP 5: Generate status reporting
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

    if quality_metrics["large_distance_count"] > 0:
        print(
            f"⚠️  {quality_metrics['large_distance_count']} matches >50px may indicate alignment issues"
        )

    # STEP 6: Save all outputs

    # Save deduplicated cells
    final_output.to_parquet(str(snakemake.output.deduplicated_cells))
    print(f"✅ Saved deduplicated cells: {snakemake.output.deduplicated_cells}")

    # Save QC outputs
    sbs_rates.to_csv(str(snakemake.output.sbs_matching_rates), sep="\t", index=False)
    print(f"✅ Saved SBS matching rates: {snakemake.output.sbs_matching_rates}")

    phenotype_rates.to_csv(
        str(snakemake.output.phenotype_matching_rates), sep="\t", index=False
    )
    print(
        f"✅ Saved phenotype matching rates: {snakemake.output.phenotype_matching_rates}"
    )

    # Create comprehensive deduplication summary in wide format
    summary_dict = {
        "plate": plate,
        "well": well,
        "status": "success",
        # Processing metrics
        "processing_raw_matches_input": len(raw_matches),
        "processing_simple_matches_input": len(merged_cells),
        "processing_final_matches_output": len(final_output),
        "processing_matches_removed": len(raw_matches) - len(final_output),
        "processing_efficiency": float(len(final_output) / len(raw_matches))
        if len(raw_matches) > 0
        else 0.0,
        # QC metrics
        "qc_sbs_total_cells": len(well_sbs_cells),
        "qc_sbs_matched_cells": sbs_rates["matched_cells"].iloc[0]
        if not sbs_rates.empty
        else 0,
        "qc_sbs_match_rate": sbs_rates["match_rate"].iloc[0]
        if not sbs_rates.empty
        else 0.0,
        "qc_phenotype_total_cells": len(well_phenotype_cells),
        "qc_phenotype_matched_cells": phenotype_rates["matched_cells"].iloc[0]
        if not phenotype_rates.empty
        else 0,
        "qc_phenotype_match_rate": phenotype_rates["match_rate"].iloc[0]
        if not phenotype_rates.empty
        else 0.0,
        # Deduplication information
        "deduplication_method": "deduplicate_matches_by_stitched_ids",
        "deduplication_uses_stitched_ids": True,
        "deduplication_preserves_original_ids": True,
        "deduplication_achieved_1to1_stitched": validation_results["is_1to1_stitched"],
        # Validation results
        "validation_is_1to1_stitched": validation_results["is_1to1_stitched"],
        "validation_match_count": validation_results["match_count"],
        # Output format information
        "output_format_columns": ";".join(final_output.columns),
        "output_format_ready_for_format_merge": True,
        "qc_analysis_completed": True,
        "qc_outputs_generated": True,
    }

    # Add distance statistics
    dist_stats = validation_results.get("distance_stats", {})
    for key, value in dist_stats.items():
        summary_dict[f"validation_distance_{key}"] = value

    # Add distance distribution
    dist_dist = validation_results.get("distance_distribution", {})
    for key, value in dist_dist.items():
        summary_dict[f"validation_distribution_{key}"] = value

    # Add quality metrics
    qual_metrics = validation_results.get("quality_metrics", {})
    for key, value in qual_metrics.items():
        summary_dict[f"validation_quality_{key}"] = value

    # Add duplication check
    stitched_duplicates = validation_results.get("stitched_duplicates", {})
    original_duplicates = validation_results.get("original_duplicates", {})

    summary_dict.update(
        {
            "validation_stitched_duplicates_phenotype": stitched_duplicates.get(
                "phenotype", 0
            ),
            "validation_stitched_duplicates_sbs": stitched_duplicates.get("sbs", 0),
            "validation_original_duplicates_phenotype": original_duplicates.get(
                "phenotype", 0
            ),
            "validation_original_duplicates_sbs": original_duplicates.get("sbs", 0),
        }
    )

    # Add quality metrics (extracted earlier)
    for key, value in quality_metrics.items():
        summary_dict[f"quality_{key}"] = value

    # Create single-row DataFrame
    summary_df = pd.DataFrame([summary_dict])
    summary_df.to_csv(
        str(snakemake.output.deduplication_summary), sep="\t", index=False
    )
    print(f"✅ Saved deduplication summary: {snakemake.output.deduplication_summary}")

    print(f"✅ Well {plate}-{well} deduplication with QC completed successfully")
    print(
        f"Final result: {len(final_output):,} deduplicated matches ready for downstream processing"
    )
    print(
        f"QC summary: SBS {sbs_rates['match_rate'].iloc[0]:.1f}% match rate, Phenotype {phenotype_rates['match_rate'].iloc[0]:.1f}% match rate"
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

    # Create empty outputs
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
    empty_df["plate"] = plate_val
    empty_df["well"] = well_val
    empty_df["site"] = 1
    empty_df["tile"] = 1
    empty_df.to_parquet(str(snakemake.output.deduplicated_cells))

    # Create failure summary in wide format
    failure_summary = {
        "plate": plate_val,
        "well": well_val,
        "status": "failed",
        "error": error_message,
        "processing_raw_matches_input": 0,
        "processing_simple_matches_input": 0,
        "processing_final_matches_output": 0,
        "processing_matches_removed": 0,
        "processing_efficiency": 0.0,
        "qc_sbs_total_cells": 0,
        "qc_sbs_matched_cells": 0,
        "qc_sbs_match_rate": 0.0,
        "qc_phenotype_total_cells": 0,
        "qc_phenotype_matched_cells": 0,
        "qc_phenotype_match_rate": 0.0,
        "deduplication_method": "deduplicate_matches_by_stitched_ids",
        "deduplication_uses_stitched_ids": True,
        "deduplication_preserves_original_ids": True,
        "deduplication_achieved_1to1_stitched": False,
        "validation_is_1to1_stitched": False,
        "validation_match_count": 0,
        "output_format_ready_for_format_merge": False,
        "qc_analysis_completed": False,
        "qc_outputs_generated": False,
    }

    summary_df = pd.DataFrame([failure_summary])
    summary_df.to_csv(
        str(snakemake.output.deduplication_summary), sep="\t", index=False
    )

    # Create empty QC files
    empty_qc = pd.DataFrame({"info": ["No QC data - processing failed"]})
    empty_qc.to_csv(str(snakemake.output.sbs_matching_rates), sep="\t", index=False)
    empty_qc.to_csv(
        str(snakemake.output.phenotype_matching_rates), sep="\t", index=False
    )
    raise

print("=== WELL MERGE DEDUPLICATION WITH QC COMPLETED ===")
