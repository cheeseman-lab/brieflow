"""Aggregate Well Summaries Script.

Aggregates individual well summary TSV files from the 3-step well merge pipeline into
consolidated plate-level summaries. All input summaries are now in wide format 
(one row per well) so this script simply loads and concatenates them.

This script processes:
1. Alignment summaries (wide format) - from well_alignment rule output [4]
2. Cell merge summaries (wide format) - from well_cell_merge rule output [2]
3. Deduplication summaries (wide format) - from well_merge_deduplicate rule output [1]
4. SBS matching rates (wide format) - from well_merge_deduplicate rule output [2]
5. Phenotype matching rates (wide format) - from well_merge_deduplicate rule output [3]

Input files (per well):
- alignment_summary.tsv: Well alignment metrics (wide format)
- merge_summary.tsv: Cell merge metrics (wide format)
- dedup_summary.tsv: Deduplication metrics (wide format)
- sbs_matching_rates.tsv: SBS matching rate analysis (wide format)
- phenotype_matching_rates.tsv: Phenotype matching rate analysis (wide format)

Output files (per plate):
- alignment_summaries.tsv: Aggregated alignment data across all wells (output [0])
- cell_merge_summaries.tsv: Aggregated cell merge data across all wells (output [1])
- dedup_summaries.tsv: Aggregated deduplication data across all wells (output [2])
- sbs_matching_summaries.tsv: Aggregated SBS matching data across all wells (output [3])
- phenotype_matching_summaries.tsv: Aggregated phenotype matching data across all wells (output [4])

Each output file contains one row per well with plate and well identifier columns.
Failed wells are included with status='failed' and placeholder values.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
import warnings
import re
import traceback


def print_section_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n--- {title} ---")


def extract_well_id_from_path(file_path: str) -> Tuple[str, str]:
    """Extract plate and well identifiers from standardized file path.

    Expects the standard naming convention: P-{plate}_W-{well}
    Example: P-1_W-A3__merge_final.parquet -> ('1', 'A3')

    Args:
        file_path: Path to summary file

    Returns:
        Tuple of (plate, well) identifiers

    Raises:
        ValueError: If plate/well cannot be extracted from path
    """
    filename = Path(file_path).name

    # Standard pattern: P-{plate}_W-{well}
    match = re.search(r"P-(\d+)_W-([A-H]\d{1,2})", filename, re.IGNORECASE)
    if match:
        plate_id = match.group(1)
        well_id = match.group(2).upper()
        # Ensure well is zero-padded (A3 -> A03)
        if len(well_id) == 2:  # e.g., "A3"
            well_id = well_id[0] + well_id[1:].zfill(2)  # "A03"
        return plate_id, well_id

    raise ValueError(
        f"Could not extract plate/well from standardized path: {file_path}"
    )


def load_wide_format_summary(file_path: str, summary_type: str) -> Optional[pd.DataFrame]:
    """Load wide format summary file (one row per well).

    Args:
        file_path: Path to summary TSV file
        summary_type: Type of summary for error messages

    Returns:
        DataFrame with summary data or None if loading fails
    """
    try:
        if not Path(file_path).exists():
            print(f"⚠️  {summary_type} summary file not found: {file_path}")
            return None

        df = pd.read_csv(file_path, sep="\t")

        if df.empty:
            print(f"⚠️  Empty {summary_type} summary file: {file_path}")
            return None

        # Ensure required columns exist
        if "plate" not in df.columns or "well" not in df.columns:
            try:
                plate_id, well_id = extract_well_id_from_path(file_path)
                if "plate" not in df.columns:
                    df["plate"] = plate_id
                if "well" not in df.columns:
                    df["well"] = well_id
            except ValueError as e:
                print(f"⚠️  Could not add plate/well to {summary_type} summary: {e}")
                return None

        return df

    except Exception as e:
        print(f"⚠️  Failed to load {summary_type} summary {file_path}: {e}")
        return None


def create_failed_well_placeholder(
    plate_id: str, well_id: str, summary_type: str
) -> pd.DataFrame:
    """Create placeholder row for failed wells.

    Args:
        plate_id: Plate identifier
        well_id: Well identifier
        summary_type: Type of summary for appropriate default columns

    Returns:
        DataFrame with single placeholder row
    """
    base_data = {
        "plate": plate_id,
        "well": well_id,
        "status": "failed",
    }

    if summary_type == "alignment":
        # Add alignment-specific placeholder columns
        placeholder_data = {
            **base_data,
            "failure_reason": "file_missing",
            "scale_factor": np.nan,
            "overlap_fraction": np.nan,
            "phenotype_triangles": 0,
            "sbs_triangles": 0,
            "alignment_score": np.nan,
            "determinant": np.nan,
            "approach": "failed",
            "transformation_type": "failed",
        }
    elif summary_type == "merge":
        # Add merge-specific placeholder columns
        placeholder_data = {
            **base_data,
            "failure_reason": "file_missing",
            "distance_threshold_pixels": np.nan,
            "phenotype_cells_before_filtering": 0,
            "sbs_cells_before_filtering": 0,
            "raw_matches_found": 0,
            "mean_match_distance": np.nan,
            "alignment_approach": "failed",
            "alignment_transformation_type": "failed",
            "alignment_score": np.nan,
            "alignment_determinant": np.nan,
        }
    elif summary_type == "dedup":
        # Add deduplication-specific placeholder columns
        placeholder_data = {
            **base_data,
            "error": "file_missing",
            "processing_final_matches_output": 0,
            "deduplication_achieved_1to1_stitched": False,
            "validation_match_count": 0,
            "validation_distance_mean": np.nan,
            "deduplication_method": "failed",
        }
    elif summary_type in ["sbs_matching", "phenotype_matching"]:
        # Add matching rate placeholder columns
        placeholder_data = {
            **base_data,
            "error": "file_missing",
            "total_cells": 0,
            "matched_cells": 0,
            "match_rate": 0.0,
        }
    else:
        placeholder_data = base_data

    return pd.DataFrame([placeholder_data])


def get_all_expected_wells(file_paths: List[str]) -> List[Tuple[str, str]]:
    """Extract all expected well identifiers from file paths.

    Args:
        file_paths: List of file paths to process

    Returns:
        List of (plate, well) tuples for all expected wells
    """
    wells = []
    for path in file_paths:
        try:
            plate_id, well_id = extract_well_id_from_path(path)
            wells.append((plate_id, well_id))
        except ValueError:
            print(f"⚠️  Could not extract well ID from path: {path}")
            continue

    return sorted(set(wells))  # Remove duplicates and sort


def aggregate_summaries(file_paths: List[str], summary_type: str) -> pd.DataFrame:
    """Aggregate summary files into a single DataFrame.

    Args:
        file_paths: List of paths to summary files
        summary_type: Type of summary for logging and error handling

    Returns:
        DataFrame with aggregated summaries (one row per well)
    """
    print(f"Aggregating {len(file_paths)} {summary_type} summary files...")

    if not file_paths:
        print(f"⚠️  No {summary_type} summary files provided")
        return pd.DataFrame()

    all_wells = get_all_expected_wells(file_paths)
    print(f"Expected wells: {len(all_wells)}")

    aggregated_rows = []
    processed_wells = set()

    # Process existing files
    for file_path in file_paths:
        try:
            plate_id, well_id = extract_well_id_from_path(file_path)
            well_key = (plate_id, well_id)

            df = load_wide_format_summary(file_path, summary_type)

            if df is not None and not df.empty:
                # Ensure plate/well columns are present and correct
                df["plate"] = plate_id
                df["well"] = well_id
                aggregated_rows.append(df)
                processed_wells.add(well_key)
                print(f"✅ Processed {summary_type} summary for {plate_id}-{well_id}")
            else:
                print(
                    f"⚠️  Empty or invalid {summary_type} summary for {plate_id}-{well_id}"
                )
                # Create placeholder for invalid file
                placeholder = create_failed_well_placeholder(
                    plate_id, well_id, summary_type
                )
                aggregated_rows.append(placeholder)
                processed_wells.add(well_key)

        except Exception as e:
            print(f"❌ Error processing {file_path}: {e}")
            continue

    # Create placeholders for missing wells
    missing_wells = set(all_wells) - processed_wells
    for plate_id, well_id in missing_wells:
        print(
            f"⚠️  Creating placeholder for missing {summary_type} summary: {plate_id}-{well_id}"
        )
        placeholder = create_failed_well_placeholder(plate_id, well_id, summary_type)
        aggregated_rows.append(placeholder)

    # Combine all rows
    if aggregated_rows:
        result = pd.concat(aggregated_rows, ignore_index=True)

        # Sort by plate and well for consistent output
        result = result.sort_values(["plate", "well"]).reset_index(drop=True)

        print(f"✅ Aggregated {len(result)} {summary_type} summaries")
        return result
    else:
        print(f"❌ No {summary_type} summaries could be processed")
        return pd.DataFrame()


def save_summary_with_fallback(
    df: pd.DataFrame, output_path: str, summary_type: str
) -> None:
    """Save summary DataFrame with fallback for empty data.

    Args:
        df: DataFrame to save
        output_path: Output file path
        summary_type: Type of summary for logging
    """
    # Create output directory if needed
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    if not df.empty:
        df.to_csv(output_path, sep="\t", index=False)
        print(f"✅ Saved {summary_type} summaries: {output_path}")
        print(f"   {len(df)} wells processed")
    else:
        # Create empty file with basic headers
        pd.DataFrame(columns=["plate", "well", "status"]).to_csv(
            output_path, sep="\t", index=False
        )
        print(f"⚠️  Saved empty {summary_type} summaries: {output_path}")


def print_success_statistics(
    alignment_df: pd.DataFrame, 
    merge_df: pd.DataFrame, 
    dedup_df: pd.DataFrame,
    sbs_matching_df: pd.DataFrame,
    phenotype_matching_df: pd.DataFrame
) -> None:
    """Print summary statistics for successful processing.

    Args:
        alignment_df: Aggregated alignment summaries
        merge_df: Aggregated merge summaries
        dedup_df: Aggregated dedup summaries
        sbs_matching_df: Aggregated SBS matching summaries
        phenotype_matching_df: Aggregated phenotype matching summaries
    """
    successful_wells = []

    if not alignment_df.empty:
        successful_alignment = len(
            alignment_df[alignment_df.get("status", "") != "failed"]
        )
        successful_wells.append(
            f"Alignment: {successful_alignment}/{len(alignment_df)}"
        )

    if not merge_df.empty:
        successful_merge = len(merge_df[merge_df.get("status", "") != "failed"])
        successful_wells.append(f"Merge: {successful_merge}/{len(merge_df)}")

    if not dedup_df.empty:
        successful_dedup = len(dedup_df[dedup_df.get("status", "") != "failed"])
        successful_wells.append(f"Dedup: {successful_dedup}/{len(dedup_df)}")

    if not sbs_matching_df.empty:
        successful_sbs = len(sbs_matching_df[sbs_matching_df.get("error", "").isna() | (sbs_matching_df.get("error", "") == "")])
        successful_wells.append(f"SBS Matching: {successful_sbs}/{len(sbs_matching_df)}")

    if not phenotype_matching_df.empty:
        successful_pheno = len(phenotype_matching_df[phenotype_matching_df.get("error", "").isna() | (phenotype_matching_df.get("error", "") == "")])
        successful_wells.append(f"Phenotype Matching: {successful_pheno}/{len(phenotype_matching_df)}")

    if successful_wells:
        print(f"Success rates: {', '.join(successful_wells)}")


def main():
    """Main execution function."""
    print("=== AGGREGATE WELL SUMMARIES ===")

    plate = snakemake.params.plate
    print(f"Processing plate: {plate}")

    # Get input file paths - using named inputs from rule
    alignment_paths = snakemake.input.alignment_summary_paths
    merge_paths = snakemake.input.merge_summary_paths
    dedup_paths = snakemake.input.dedup_summary_paths
    sbs_matching_paths = snakemake.input.sbs_matching_rates_paths
    phenotype_matching_paths = snakemake.input.phenotype_matching_rates_paths

    print(f"Input files:")
    print(f"  Alignment summaries: {len(alignment_paths)}")
    print(f"  Merge summaries: {len(merge_paths)}")
    print(f"  Deduplication summaries: {len(dedup_paths)}")
    print(f"  SBS matching summaries: {len(sbs_matching_paths)}")
    print(f"  Phenotype matching summaries: {len(phenotype_matching_paths)}")

    try:
        # Process all summaries (now all in wide format)
        print_section_header("Processing Alignment Summaries")
        alignment_df = aggregate_summaries(alignment_paths, "alignment")

        print_section_header("Processing Cell Merge Summaries")
        merge_df = aggregate_summaries(merge_paths, "merge")

        print_section_header("Processing Deduplication Summaries")
        dedup_df = aggregate_summaries(dedup_paths, "dedup")

        print_section_header("Processing SBS Matching Summaries")
        sbs_matching_df = aggregate_summaries(sbs_matching_paths, "sbs_matching")

        print_section_header("Processing Phenotype Matching Summaries")
        phenotype_matching_df = aggregate_summaries(phenotype_matching_paths, "phenotype_matching")

        # Save aggregated summaries
        print_section_header("Saving Aggregated Summaries")

        save_summary_with_fallback(
            alignment_df, snakemake.output.alignment_summaries, "alignment"
        )
        save_summary_with_fallback(
            merge_df, snakemake.output.cell_merge_summaries, "cell merge"
        )
        save_summary_with_fallback(
            dedup_df, snakemake.output.dedup_summaries, "deduplication"
        )
        save_summary_with_fallback(
            sbs_matching_df, snakemake.output.sbs_matching_summaries, "SBS matching"
        )
        save_summary_with_fallback(
            phenotype_matching_df, snakemake.output.phenotype_matching_summaries, "phenotype matching"
        )

        print(f"\n🎉 Successfully aggregated summaries for plate {plate}")

        # Print summary statistics
        print_success_statistics(alignment_df, merge_df, dedup_df, sbs_matching_df, phenotype_matching_df)

    except Exception as e:
        print(f"❌ Error during aggregation: {e}")
        traceback.print_exc()
        raise

    print(f"\n🎉 Successfully completed aggregation for plate {plate}")
    print("=== AGGREGATE WELL SUMMARIES COMPLETED ===")


if __name__ == "__main__":
    main()