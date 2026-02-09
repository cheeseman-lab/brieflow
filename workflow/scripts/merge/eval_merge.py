"""Evaluate Merge Results.

Evaluates final merge results (after deduplication) by analyzing SBS and
phenotype matching rates and generating summary statistics and plots.
"""

import pandas as pd
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.format_merge import identify_single_gene_mappings
from lib.merge.eval_merge import plot_sbs_ph_matching_heatmap, plot_cell_positions

# Load deduplicated merge data for summary statistics
merge_deduplicated = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.deduplicated_merge_paths],
        ignore_index=True,
    )
)

# Load formatted (pre-dedup) merge data for matching rate heatmaps
merge_formatted = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.formatted_merge_paths],
        ignore_index=True,
    )
)

# Standardize coordinate column names (stitch uses global_i_0/global_j_0, fast uses i_0/j_0)
if "global_i_0" in merge_deduplicated.columns:
    merge_deduplicated = merge_deduplicated.rename(
        columns={
            "global_i_0": "i_0",
            "global_j_0": "j_0",
            "global_i_1": "i_1",
            "global_j_1": "j_1",
        }
    )

# Load SBS data
sbs_cells = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.combine_cells_paths],
        ignore_index=True,
    )
)
sbs_info = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.sbs_info_paths],
        ignore_index=True,
    )
)

# Load phenotype data
phenotype_min_cp = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.min_phenotype_cp_paths],
        ignore_index=True,
    )
)
phenotype_info = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.phenotype_info_paths],
        ignore_index=True,
    )
)

# Load dedup stats by well
dedup_stats = {}
for p in snakemake.input.dedup_stats_paths:
    df = pd.read_csv(p, sep="\t")
    # Extract well from filename: P-1_W-A1__deduplication_stats.tsv
    well = Path(p).stem.split("__")[0].split("_W-")[1]
    dedup_stats[well] = df

# Identify single gene mappings in merge_deduplicated
merge_deduplicated["mapped_single_gene"] = merge_deduplicated.apply(
    lambda x: identify_single_gene_mappings(x), axis=1
)

# Build per-well summary rows
rows = []
for well in sorted(merge_deduplicated["well"].unique()):
    well_merge = merge_deduplicated[merge_deduplicated["well"] == well]
    well_ph = phenotype_min_cp[phenotype_min_cp["well"] == well]
    well_sbs = sbs_cells[sbs_cells["well"] == well]
    well_dedup = dedup_stats.get(well)

    # Get matched_raw from dedup stats (Initial stage = before deduplication)
    if well_dedup is not None and "Initial" in well_dedup["stage"].values:
        matched_raw = int(
            well_dedup[well_dedup["stage"] == "Initial"]["total_cells"].iloc[0]
        )
    else:
        matched_raw = len(well_merge)  # Fallback if dedup stats not available

    # Get cell counts from merge input data (sbs_info and phenotype_info)
    well_sbs_info = sbs_info[sbs_info["well"] == well]
    well_ph_info = phenotype_info[phenotype_info["well"] == well]

    # matched_final is the count AFTER deduplication (from final_merge)
    matched_final = len(well_merge)
    ph_cells = len(well_ph_info)  # Use phenotype_info (all segmented cells)
    sbs_cells_count = len(well_sbs_info)  # Use sbs_info (all segmented cells)

    # Count unique cells in merge (use full cell identifier, not just label)
    # Phenotype cells: (plate, well, tile, site, cell_0)
    unique_ph_in_merge = (
        well_merge[["plate", "well", "tile", "site", "cell_0"]]
        .drop_duplicates()
        .shape[0]
    )
    # SBS cells: (plate, well, tile, cell_1) - no site dimension in SBS
    unique_sbs_in_merge = (
        well_merge[["plate", "well", "tile", "cell_1"]].drop_duplicates().shape[0]
    )

    rows.append(
        {
            "well": well,
            "ph_cells": ph_cells,
            "sbs_cells": sbs_cells_count,
            "matched_raw": matched_raw,
            "total_match_pairs": matched_final,
            "unique_ph_in_merge": unique_ph_in_merge,
            "unique_sbs_in_merge": unique_sbs_in_merge,
            "ph_recovery_rate": round(unique_ph_in_merge / ph_cells, 3)
            if ph_cells > 0
            else 0,
            "sbs_recovery_rate": (
                round(unique_sbs_in_merge / sbs_cells_count, 3)
                if sbs_cells_count > 0
                else 0
            ),
            "dist_mean": round(well_merge["distance"].mean(), 2),
            "dist_median": round(well_merge["distance"].median(), 2),
            "cells_with_barcode": int(well_merge["gene_symbol_0"].notna().sum()),
            "single_gene_count": int(well_merge["mapped_single_gene"].sum()),
            "single_gene_rate": round(well_merge["mapped_single_gene"].mean(), 3),
        }
    )

# TOTAL row removed per user request

# Save comprehensive merge summary
summary_df = pd.DataFrame(rows)
summary_df.to_csv(snakemake.output.merge_summary, sep="\t", index=False)

# Evaluate minimal merge data (use formatted/pre-dedup merge for matching rates)
merge_minimal = merge_formatted[
    ["plate", "well", "tile", "site", "cell_0", "cell_1", "distance"]
]

# Eval SBS matching rates (use sbs_info - all segmented cells that went into merge)
sbs_summary, fig = plot_sbs_ph_matching_heatmap(
    merge_minimal,
    sbs_info.rename(columns={"cell": "cell_1"}),  # sbs_info uses "cell" not "cell_1"
    target="sbs",
    shape=snakemake.params.heatmap_shape_sbs,
    plate=snakemake.params.heatmap_plate_sbs,
    return_summary=True,
)
sbs_summary.to_csv(snakemake.output.sbs_to_ph_matching_rates_tsv, sep="\t", index=False)
fig.savefig(snakemake.output.sbs_to_ph_matching_rates_png)

# Eval phenotype matching rates (use phenotype_info - all segmented cells that went into merge)
ph_summary, fig = plot_sbs_ph_matching_heatmap(
    merge_minimal,
    phenotype_info.rename(
        columns={"cell": "cell_0"}
    ),  # phenotype_info uses "cell" not "cell_0"
    target="phenotype",
    shape=snakemake.params.heatmap_shape_ph,
    plate=snakemake.params.heatmap_plate_ph,
    return_summary=True,
)
ph_summary.to_csv(snakemake.output.ph_to_sbs_matching_rates_tsv, sep="\t", index=False)
fig.savefig(snakemake.output.ph_to_sbs_matching_rates_png)

# Evaluate all formatted merge data
fig = plot_cell_positions(merge_deduplicated, title="All Cells by Channel Min")
fig.savefig(snakemake.output.all_cells_by_channel_min)
fig = plot_cell_positions(
    merge_deduplicated.query("channels_min==0"),
    title="Cells with Channel Min = 0",
    color="red",
)
fig.savefig(snakemake.output.cells_with_channel_min_0)

# Aggregate dedup stats across all wells (already loaded above)
dedup_dfs = []
for well, df in dedup_stats.items():
    df = df.copy()
    df["well"] = well
    dedup_dfs.append(df)

if dedup_dfs:
    dedup_summaries = pd.concat(dedup_dfs, ignore_index=True)
    stage_order = ["Initial", "After SBS dedup", "After phenotype dedup"]
    dedup_summaries["stage"] = pd.Categorical(
        dedup_summaries["stage"], categories=stage_order, ordered=True
    )
    dedup_summaries = dedup_summaries.sort_values(["well", "stage"]).reset_index(
        drop=True
    )
else:
    dedup_summaries = pd.DataFrame(columns=["well", "stage", "total_cells"])

dedup_summaries.to_csv(snakemake.output.dedup_summaries, sep="\t", index=False)
