"""Summarize Merge.

Simple aggregation of individual well summary TSV files into plate-level summaries.
All input summaries are in wide format (one row per well).
"""

import pandas as pd
from pathlib import Path

# Get params directly
plate = snakemake.params.plate
wells = snakemake.params.wells

# Convert wells to list of strings to ensure consistency
wells = [str(w) for w in wells]

print(f"Aggregating summaries for plate {plate}")

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
    dataframes = []

    for well, file_path in zip(wells, input_paths):
        try:
            if Path(file_path).exists():
                df = pd.read_csv(file_path, sep="\t")

                if not df.empty:
                    df["plate"] = plate
                    df["well"] = well
                    dataframes.append(df)

        except Exception as e:
            pass

    # Combine and save
    if dataframes:
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(["plate", "well"]).reset_index(drop=True)

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(output_path, sep="\t", index=False)
    else:
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["plate", "well"]).to_csv(
            output_path, sep="\t", index=False
        )

print(f"\nCompleted aggregation for plate {plate}")