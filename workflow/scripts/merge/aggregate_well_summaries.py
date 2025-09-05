"""Aggregate Well Summaries Script.

Aggregates individual well summary TSV files from the 3-step well merge pipeline into
consolidated plate-level summaries. Converts key-value format summaries (merge/dedup)
to one-row-per-well format for easier analysis.

This script processes:
1. Alignment summaries (already in row format) - from well_alignment rule output [4]
2. Cell merge summaries (converts from key-value to row format) - from well_cell_merge rule output [2]
3. Deduplication summaries (converts from key-value to row format) - from well_merge_deduplicate rule output [1]

Input files (per well):
- alignment_summary.tsv: Well alignment metrics (row format)
- merge_summary.tsv: Cell merge metrics (key-value format)
- dedup_summary.tsv: Deduplication metrics (key-value format)

Output files (per plate):
- alignment_summaries.tsv: Aggregated alignment data across all wells (output [0])
- cell_merge_summaries.tsv: Aggregated cell merge data across all wells (output [1])
- dedup_summaries.tsv: Aggregated deduplication data across all wells (output [2])

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


def load_alignment_summary(file_path: str) -> Optional[pd.DataFrame]:
    """Load alignment summary file (already in row format).

    Args:
        file_path: Path to alignment summary TSV

    Returns:
        DataFrame with alignment summary or None if loading fails
    """
    try:
        if not Path(file_path).exists():
            print(f"⚠️  Alignment summary file not found: {file_path}")
            return None

        df = pd.read_csv(file_path, sep="\t")

        if df.empty:
            print(f"⚠️  Empty alignment summary file: {file_path}")
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
                print(f"⚠️  Could not add plate/well to alignment summary: {e}")
                return None

        return df

    except Exception as e:
        print(f"⚠️  Failed to load alignment summary {file_path}: {e}")
        return None


def load_key_value_summary(file_path: str, summary_type: str) -> Optional[pd.DataFrame]:
    """Load and convert key-value format summary to row format.

    Args:
        file_path: Path to summary TSV file
        summary_type: Type of summary ('merge' or 'dedup') for error messages

    Returns:
        DataFrame with single row containing all metrics or None if loading fails
    """
    try:
        if not Path(file_path).exists():
            print(f"⚠️  {summary_type} summary file not found: {file_path}")
            return None

        print(f"DEBUG - Loading {summary_type} file: {file_path}")
        df = pd.read_csv(file_path, sep="\t")
        print(f"DEBUG - Loaded {summary_type} file shape: {df.shape}")
        print(f"DEBUG - {summary_type} file columns: {list(df.columns)}")

        if df.empty:
            print(f"⚠️  Empty {summary_type} summary file: {file_path}")
            return None

        # Expect columns: metric, value
        if "metric" not in df.columns or "value" not in df.columns:
            print(f"⚠️  {summary_type} summary {file_path} missing metric/value columns")
            print(f"DEBUG - Available columns: {list(df.columns)}")
            return None

        print(f"DEBUG - {summary_type} file first few rows:\n{df.head()}")

        # Convert from key-value to single row
        row_data = {}
        for _, row in df.iterrows():
            metric = str(row["metric"])
            value = row["value"]

            # Handle different value types
            if pd.isna(value):
                row_data[metric] = None
            elif isinstance(value, str):
                # Try to convert numeric strings
                try:
                    if "." in value or "e" in value.lower():
                        row_data[metric] = float(value)
                    else:
                        row_data[metric] = int(value)
                except (ValueError, AttributeError):
                    row_data[metric] = value
            else:
                row_data[metric] = value

        print(f"DEBUG - Converted {len(row_data)} metrics to row format")

        # Extract plate/well if not present in the data
        if "plate" not in row_data or "well" not in row_data:
            try:
                plate_id, well_id = extract_well_id_from_path(file_path)
                if "plate" not in row_data:
                    row_data["plate"] = plate_id
                if "well" not in row_data:
                    row_data["well"] = well_id
                print(f"DEBUG - Added plate/well from path: {plate_id}, {well_id}")
            except ValueError as e:
                print(f"⚠️  Could not extract well ID for {summary_type} summary: {e}")
                return None

        result_df = pd.DataFrame([row_data])
        print(f"DEBUG - Final {summary_type} dataframe shape: {result_df.shape}")
        return result_df

    except Exception as e:
        print(f"⚠️  Failed to load {summary_type} summary {file_path}: {e}")
        import traceback

        traceback.print_exc()
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
            "reason": "file_missing",
            "distance_threshold_pixels": np.nan,
            "phenotype_cells_before_filtering": 0,
            "sbs_cells_before_filtering": 0,
            "raw_matches_found": 0,
            "mean_match_distance": np.nan,
        }
    elif summary_type == "dedup":
        # Add deduplication-specific placeholder columns
        placeholder_data = {
            **base_data,
            "error": "file_missing",
            "processing_final_matches_output": 0,
            "deduplication_achieved_1to1_stitched": False,
            "quality_match_count": 0,
            "quality_mean_distance": np.nan,
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
        summary_type: Type of summary ('alignment', 'merge', or 'dedup')

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

            if summary_type == "alignment":
                df = load_alignment_summary(file_path)
            else:
                df = load_key_value_summary(file_path, summary_type)

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


def create_error_plot(
    output_path: str, plate: str, data_type: str, error_msg: str
) -> None:
    """Create a standardized error plot.

    Args:
        output_path: Path to save the plot
        plate: Plate identifier
        data_type: Type of data (e.g., 'phenotype', 'sbs')
        error_msg: Error message to display
    """
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"{error_msg}\nfor plate {plate}",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title(f"{data_type.title()} Plate {plate} - Error")
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception as e:
        print(f"❌ Could not create error plot: {e}")


def generate_plate_qc_plots(
    plate: str,
    phenotype_files: List[str],
    sbs_files: List[str],
    phenotype_output: str,
    sbs_output: str,
) -> None:
    """Generate plate-level QC plots for both phenotype and SBS data.

    Args:
        plate: Plate identifier
        phenotype_files: List of phenotype position files
        sbs_files: List of SBS position files
        phenotype_output: Output path for phenotype plot
        sbs_output: Output path for SBS plot
    """
    try:
        from lib.merge.eval_stitch import plot_cell_positions_plate_scatter

        # Generate phenotype plate QC plot
        print("Creating phenotype plate scatter plot...")
        try:
            if phenotype_files:
                plot_cell_positions_plate_scatter(
                    parquet_files=phenotype_files,
                    output_path=phenotype_output,
                    data_type="phenotype",
                    plate=plate,
                    title=f"Phenotype Cell Positions - Plate {plate}",
                    point_size=0.1,
                    alpha=0.8,
                    cmap="tab20",
                    colorbar_label="Original Tile ID",
                    figsize=(16, 11),
                )
                print(f"✅ Phenotype plate QC plot saved: {phenotype_output}")
            else:
                print("⚠️  No phenotype position files available")
                create_error_plot(
                    phenotype_output, plate, "phenotype", "No phenotype data available"
                )
        except Exception as e:
            print(f"❌ Error creating phenotype plate QC plot: {e}")
            create_error_plot(
                phenotype_output, plate, "phenotype", "Error generating phenotype plot"
            )

        # Generate SBS plate QC plot
        print("Creating SBS plate scatter plot...")
        try:
            if sbs_files:
                plot_cell_positions_plate_scatter(
                    parquet_files=sbs_files,
                    output_path=sbs_output,
                    data_type="sbs",
                    plate=plate,
                    title=f"SBS Cell Positions - Plate {plate}",
                    point_size=0.1,
                    alpha=0.8,
                    cmap="tab20",
                    colorbar_label="Original Tile ID",
                    figsize=(16, 11),
                )
                print(f"✅ SBS plate QC plot saved: {sbs_output}")
            else:
                print("⚠️  No SBS position files available")
                create_error_plot(sbs_output, plate, "sbs", "No SBS data available")
        except Exception as e:
            print(f"❌ Error creating SBS plate QC plot: {e}")
            create_error_plot(sbs_output, plate, "sbs", "Error generating SBS plot")

    except ImportError as e:
        print(f"❌ Could not import plot_cell_positions_plate_scatter: {e}")
        # Create placeholder plots for both outputs
        create_error_plot(
            phenotype_output, plate, "phenotype", "Plot function not available"
        )
        create_error_plot(sbs_output, plate, "sbs", "Plot function not available")
    except Exception as e:
        print(f"❌ Unexpected error during plate QC plot generation: {e}")
        traceback.print_exc()
        # Create error plots for both outputs
        create_error_plot(phenotype_output, plate, "phenotype", "Unexpected error")
        create_error_plot(sbs_output, plate, "sbs", "Unexpected error")


def print_success_statistics(
    alignment_df: pd.DataFrame, merge_df: pd.DataFrame, dedup_df: pd.DataFrame
) -> None:
    """Print summary statistics for successful processing.

    Args:
        alignment_df: Aggregated alignment summaries
        merge_df: Aggregated merge summaries
        dedup_df: Aggregated dedup summaries
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

    print(f"Input files:")
    print(f"  Alignment summaries: {len(alignment_paths)}")
    print(f"  Merge summaries: {len(merge_paths)}")
    print(f"  Deduplication summaries: {len(dedup_paths)}")

    try:
        # Process alignment summaries (already in row format)
        print_section_header("Processing Alignment Summaries")
        alignment_df = aggregate_summaries(alignment_paths, "alignment")

        # Process merge summaries (convert from key-value to row format)
        print_section_header("Processing Cell Merge Summaries")
        merge_df = aggregate_summaries(merge_paths, "merge")

        # Process deduplication summaries (convert from key-value to row format)
        print_section_header("Processing Deduplication Summaries")
        dedup_df = aggregate_summaries(dedup_paths, "dedup")

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

        print(f"\n🎉 Successfully aggregated summaries for plate {plate}")

        # Print summary statistics
        print_success_statistics(alignment_df, merge_df, dedup_df)

    except Exception as e:
        print(f"❌ Error during aggregation: {e}")
        traceback.print_exc()
        raise

    # Generate plate-level QC plots
    print_section_header("Generating Plate-Level QC Plots")

    try:
        phenotype_files = snakemake.input.phenotype_positions_paths
        sbs_files = snakemake.input.sbs_positions_paths

        generate_plate_qc_plots(
            plate=plate,
            phenotype_files=phenotype_files,
            sbs_files=sbs_files,
            phenotype_output=snakemake.output.phenotype_plate_qc,
            sbs_output=snakemake.output.sbs_plate_qc,
        )

    except Exception as e:
        print(f"❌ Error during QC plot generation: {e}")
        traceback.print_exc()
        # Continue execution even if plots fail

    print(
        f"\n🎉 Successfully completed aggregation and QC plot generation for plate {plate}"
    )
    print("=== AGGREGATE WELL SUMMARIES COMPLETED ===")


if __name__ == "__main__":
    main()
