"""Evaluate Merge Results.

Evaluates merge results by analyzing SBS and phenotype matching rates
and generating summary statistics and plots.
"""

import pandas as pd
from pathlib import Path

from lib.shared.file_utils import validate_dtypes
from lib.merge.format_merge import identify_single_gene_mappings
from lib.merge.eval_merge import plot_sbs_ph_matching_heatmap, plot_cell_positions

# Load data for evaluating merge
merge_formatted = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.format_merge_paths],
        ignore_index=True,
    )
)
sbs_cells = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.combine_cells_paths],
        ignore_index=True,
    )
)
phenotype_min_cp = validate_dtypes(
    pd.concat(
        [pd.read_parquet(p) for p in snakemake.input.min_phenotype_cp_paths],
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

# Identify single gene mappings in merge_formatted
merge_formatted["mapped_single_gene"] = merge_formatted.apply(
    lambda x: identify_single_gene_mappings(x), axis=1
)

# Build per-well summary rows
rows = []
for well in sorted(merge_formatted["well"].unique()):
    well_merge = merge_formatted[merge_formatted["well"] == well]
    well_ph = phenotype_min_cp[phenotype_min_cp["well"] == well]
    well_sbs = sbs_cells[sbs_cells["well"] == well]
    well_dedup = dedup_stats.get(well)

    # Get matched_raw from dedup stats (Initial stage)
    if well_dedup is not None and "Initial" in well_dedup["stage"].values:
        matched_raw = int(
            well_dedup[well_dedup["stage"] == "Initial"]["total_cells"].iloc[0]
        )
    else:
        matched_raw = len(well_merge)

    matched_final = len(well_merge)
    ph_cells = len(well_ph)
    sbs_cells_count = len(well_sbs)

    rows.append(
        {
            "well": well,
            "ph_cells": ph_cells,
            "sbs_cells": sbs_cells_count,
            "matched_raw": matched_raw,
            "matched_final": matched_final,
            "ph_match_rate": round(matched_final / ph_cells, 3) if ph_cells > 0 else 0,
            "sbs_match_rate": (
                round(matched_final / sbs_cells_count, 3) if sbs_cells_count > 0 else 0
            ),
            "dist_mean": round(well_merge["distance"].mean(), 2),
            "dist_median": round(well_merge["distance"].median(), 2),
            "cells_with_barcode": int(well_merge["gene_symbol_0"].notna().sum()),
            "single_gene_count": int(well_merge["mapped_single_gene"].sum()),
            "single_gene_rate": round(well_merge["mapped_single_gene"].mean(), 3),
        }
    )

# Add TOTAL row
total_ph = len(phenotype_min_cp)
total_sbs = len(sbs_cells)
total_matched_final = len(merge_formatted)
total_matched_raw = sum(r["matched_raw"] for r in rows)

total_row = {
    "well": "TOTAL",
    "ph_cells": total_ph,
    "sbs_cells": total_sbs,
    "matched_raw": total_matched_raw,
    "matched_final": total_matched_final,
    "ph_match_rate": round(total_matched_final / total_ph, 3) if total_ph > 0 else 0,
    "sbs_match_rate": round(total_matched_final / total_sbs, 3) if total_sbs > 0 else 0,
    "dist_mean": round(merge_formatted["distance"].mean(), 2),
    "dist_median": round(merge_formatted["distance"].median(), 2),
    "cells_with_barcode": int(merge_formatted["gene_symbol_0"].notna().sum()),
    "single_gene_count": int(merge_formatted["mapped_single_gene"].sum()),
    "single_gene_rate": round(merge_formatted["mapped_single_gene"].mean(), 3),
}
rows.append(total_row)

# Save comprehensive merge summary
summary_df = pd.DataFrame(rows)
summary_df.to_csv(snakemake.output.merge_summary, sep="\t", index=False)

# Evaluate minimal merge data
merge_minimal = merge_formatted[
    ["plate", "well", "tile", "site", "cell_0", "cell_1", "distance"]
]

# Eval SBS matching rates
sbs_summary, fig = plot_sbs_ph_matching_heatmap(
    merge_minimal,
    sbs_cells,
    target="sbs",
    shape="6W_sbs",
    return_summary=True,
)
sbs_summary.to_csv(snakemake.output.sbs_to_ph_matching_rates_tsv, sep="\t", index=False)
fig.savefig(snakemake.output.sbs_to_ph_matching_rates_png)

# Eval phenotype matching rates
ph_summary, fig = plot_sbs_ph_matching_heatmap(
    merge_minimal,
    phenotype_min_cp.rename(columns={"label": "cell_0"}),
    target="phenotype",
    shape="6W_ph",
    return_summary=True,
)
ph_summary.to_csv(snakemake.output.ph_to_sbs_matching_rates_tsv, sep="\t", index=False)
fig.savefig(snakemake.output.ph_to_sbs_matching_rates_png)

# Evaluate all formatted merge data
fig = plot_cell_positions(merge_formatted, title="All Cells by Channel Min")
fig.savefig(snakemake.output.all_cells_by_channel_min)
fig = plot_cell_positions(
    merge_formatted.query("channels_min==0"),
    title="Cells with Channel Min = 0",
    color="red",
)
fig.savefig(snakemake.output.cells_with_channel_min_0)
