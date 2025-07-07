import pandas as pd
import matplotlib.pyplot as plt

from lib.sbs.standardize_barcode_design import get_barcode_list
from lib.sbs.eval_mapping import (
    plot_mapping_vs_threshold,
    plot_read_mapping_heatmap,
    plot_cell_mapping_heatmap,
    plot_cell_metric_histogram,
    plot_gene_symbol_histogram,
    mapping_overview,
)

# Read barcodes
df_barcode_library = pd.read_csv(snakemake.params.df_barcode_library_fp, sep="\t")
barcodes = get_barcode_list(
    df_barcode_library, sequencing_order=snakemake.params.sequencing_order
)

# Load SBS processing files
reads = pd.concat(
    [pd.read_parquet(p) for p in snakemake.input.reads_paths], ignore_index=True
)
cells = pd.concat(
    [pd.read_parquet(p) for p in snakemake.input.cells_paths], ignore_index=True
)
sbs_info = pd.concat(
    [pd.read_parquet(p) for p in snakemake.input.sbs_info_paths], ignore_index=True
)

_, fig = plot_mapping_vs_threshold(reads, barcodes, "peak", num_thresholds=10)
fig.savefig(snakemake.output[0])

_, fig = plot_mapping_vs_threshold(reads, barcodes, "Q_min", num_thresholds=10)
fig.savefig(snakemake.output[1])

fig = plot_read_mapping_heatmap(reads, barcodes, shape="6W_sbs")
fig.savefig(snakemake.output[2])

df_summary_one, fig = plot_cell_mapping_heatmap(
    cells,
    sbs_info,
    barcodes,
    mapping_to="one",
    mapping_strategy="gene symbols",
    shape="6W_sbs",
    return_summary=True,
)
df_summary_one.to_csv(snakemake.output[3], index=False, sep="\t")
fig.savefig(snakemake.output[4])

df_summary_any, fig = plot_cell_mapping_heatmap(
    cells,
    sbs_info,
    barcodes,
    mapping_to="any",
    mapping_strategy="gene symbols",
    shape="6W_sbs",
    return_summary=True,
)
df_summary_any.to_csv(snakemake.output[5], index=False, sep="\t")
fig.savefig(snakemake.output[6])

_, fig = plot_cell_metric_histogram(cells, sort_by=snakemake.params.sort_by)
fig.savefig(snakemake.output[7])

_, fig = plot_gene_symbol_histogram(cells)
fig.savefig(snakemake.output[8])

mapping_overview_df = mapping_overview(
    sbs_info, cells, sort_by=snakemake.params.sort_by
)
mapping_overview_df.to_csv(snakemake.output[9], sep="\t", index=False)
