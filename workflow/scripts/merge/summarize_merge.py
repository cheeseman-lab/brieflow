"""Summarize Merge.

Simple aggregation of individual well summary TSV files into plate-level summaries.
All input summaries are in wide format (one row per well).
"""

import pandas as pd
from pathlib import Path
import re

# Get params directly
plate = snakemake.params.plate
wells = snakemake.params.wells

print(f"Aggregating summaries for plate {plate}")
print(f"Expected wells: {wells}")


def extract_well_from_path(file_path):
    """Extract well identifier from file path.
    
    Assumes file paths contain pattern like 'plate-{plate}_well-{well}'
    """
    path_str = str(file_path)
    match = re.search(r'well-([A-Z0-9]+)', path_str)
    if match:
        return match.group(1)
    return None


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
    print(f"\nProcessing {summary_type}...")
    print(f"  Found {len(input_paths)} input files")

    dataframes = []

    for file_path in input_paths:
        # Extract well from the file path itself
        well = extract_well_from_path(file_path)
        
        if well is None:
            print(f"  ⚠️  Could not extract well from path: {file_path}")
            continue
            
        try:
            if Path(file_path).exists():
                df = pd.read_csv(file_path, sep="\t")

                if not df.empty:
                    # Add plate/well extracted from path
                    df["plate"] = plate
                    df["well"] = well
                    dataframes.append(df)
                    print(f"  ✅ {plate}-{well}")
                else:
                    print(f"  ⚠️  Empty file: {well} ({file_path})")
            else:
                print(f"  ⚠️  Missing file: {well} ({file_path})")

        except Exception as e:
            print(f"  ❌ Error loading {well}: {e}")

    # Combine and save
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(["plate", "well"]).reset_index(drop=True)

        # Create output directory and save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
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
        print(f"  ⚠️  No valid data found - saved empty file to {output_path}")

print(f"\nCompleted aggregation for plate {plate}")