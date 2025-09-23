"""Aggregate Well Summaries Script for Snakemake.

Simple aggregation of individual well summary TSV files into plate-level summaries.
All input summaries are in wide format (one row per well).
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import matplotlib.pyplot as plt

# Import the plotting function from your library
from lib.merge.eval_stitch import plot_cell_positions_plate_scatter

# Extract plate from params
plate = snakemake.params.plate
print(f"Aggregating summaries for plate {plate}")


# Helper function to extract well ID from path
def extract_well_info(file_path):
    """Extract plate and well from file path."""
    filename = Path(file_path).name
    match = re.search(r"P-(\d+)_W-([A-H]\d{1,2})", filename, re.IGNORECASE)
    if match:
        plate_id = match.group(1)
        well_id = match.group(2).upper()
        # Zero-pad well if needed (A3 -> A03)
        if len(well_id) == 2:
            well_id = well_id[0] + well_id[1:].zfill(2)
        return plate_id, well_id
    return None, None


# Process each summary type
summary_types = [
    (
        "alignment",
        snakemake.input.alignment_summary_paths,
        snakemake.output.alignment_summaries,
    ),
    (
        "merge",
        snakemake.input.merge_summary_paths,
        snakemake.output.cell_merge_summaries,
    ),
    ("dedup", snakemake.input.dedup_summary_paths, snakemake.output.dedup_summaries),
    (
        "sbs_matching",
        snakemake.input.sbs_matching_rates_paths,
        snakemake.output.sbs_matching_summaries,
    ),
    (
        "phenotype_matching",
        snakemake.input.phenotype_matching_rates_paths,
        snakemake.output.phenotype_matching_summaries,
    ),
]

for summary_type, input_paths, output_path in summary_types:
    print(f"\nProcessing {len(input_paths)} {summary_type} files...")

    dataframes = []

    for file_path in input_paths:
        try:
            # Load the file
            if Path(file_path).exists():
                df = pd.read_csv(file_path, sep="\t")

                if not df.empty:
                    # Extract and add plate/well info if missing
                    filename = Path(file_path).name
                    match = re.search(
                        r"P-(\d+)_W-([A-H]\d{1,2})", filename, re.IGNORECASE
                    )
                    if match:
                        plate_id = match.group(1)
                        well_id = match.group(2).upper()
                        # Zero-pad well if needed (A3 -> A03)
                        if len(well_id) == 2:
                            well_id = well_id[0] + well_id[1:].zfill(2)
                        df["plate"] = plate_id
                        df["well"] = well_id
                        dataframes.append(df)
                        print(f"  ✅ {plate_id}-{well_id}")
                    else:
                        print(f"  ⚠️  Could not extract well info from {file_path}")
                else:
                    print(f"  ⚠️  Empty file: {file_path}")
            else:
                print(f"  ⚠️  Missing file: {file_path}")

        except Exception as e:
            print(f"  ❌ Error loading {file_path}: {e}")

    # Combine and save
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(["plate", "well"]).reset_index(drop=True)

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        # Save
        combined_df.to_csv(output_path, sep="\t", index=False)
        print(f"  📁 Saved {len(combined_df)} rows to {output_path}")

        # Quick success rate calculation
        if "status" in combined_df.columns:
            success_count = len(combined_df[combined_df["status"] != "failed"])
            print(f"  📊 Success rate: {success_count}/{len(combined_df)}")
        elif "error" in combined_df.columns:
            success_count = len(
                combined_df[combined_df["error"].isna() | (combined_df["error"] == "")]
            )
            print(f"  📊 Success rate: {success_count}/{len(combined_df)}")
    else:
        # Create empty output file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["plate", "well"]).to_csv(
            output_path, sep="\t", index=False
        )
        print(f"  📁 Saved empty file to {output_path}")

print(f"\nCompleted aggregation for plate {plate}")
