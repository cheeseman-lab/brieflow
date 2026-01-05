"""Summarize Stitch.

Aggregates stitch-specific summary files (alignment and merge summaries)
from individual wells into plate-level summaries.
"""

import pandas as pd
from pathlib import Path

# Get params
plate = snakemake.params.plate
wells = snakemake.params.wells

# Convert wells to list of strings
wells = [str(w) for w in wells]

print(f"Aggregating stitch summaries for plate {plate}")

# Process alignment summaries
alignment_dfs = []
for well, file_path in zip(wells, snakemake.input.alignment_summary_paths):
    try:
        if Path(file_path).exists():
            df = pd.read_csv(file_path, sep="\t")
            if not df.empty:
                df["plate"] = plate
                df["well"] = well
                alignment_dfs.append(df)
    except Exception:
        pass

if alignment_dfs:
    alignment_summaries = pd.concat(alignment_dfs, ignore_index=True)
    alignment_summaries = alignment_summaries.sort_values(
        ["plate", "well"]
    ).reset_index(drop=True)
else:
    alignment_summaries = pd.DataFrame(columns=["plate", "well"])

Path(snakemake.output.alignment_summaries).parent.mkdir(parents=True, exist_ok=True)
alignment_summaries.to_csv(snakemake.output.alignment_summaries, sep="\t", index=False)

# Process cell merge summaries
merge_dfs = []
for well, file_path in zip(wells, snakemake.input.merge_summary_paths):
    try:
        if Path(file_path).exists():
            df = pd.read_csv(file_path, sep="\t")
            if not df.empty:
                df["plate"] = plate
                df["well"] = well
                merge_dfs.append(df)
    except Exception:
        pass

if merge_dfs:
    cell_merge_summaries = pd.concat(merge_dfs, ignore_index=True)
    cell_merge_summaries = cell_merge_summaries.sort_values(
        ["plate", "well"]
    ).reset_index(drop=True)
else:
    cell_merge_summaries = pd.DataFrame(columns=["plate", "well"])

Path(snakemake.output.cell_merge_summaries).parent.mkdir(parents=True, exist_ok=True)
cell_merge_summaries.to_csv(
    snakemake.output.cell_merge_summaries, sep="\t", index=False
)

print(f"Completed stitch summary aggregation for plate {plate}")
