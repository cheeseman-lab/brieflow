import pandas as pd
import matplotlib as plt

from lib.sbs.eval_mapping import (
    plot_mapping_vs_threshold,
    plot_read_mapping_heatmap,
    plot_cell_mapping_heatmap,
    plot_reads_per_cell_histogram,
    plot_gene_symbol_histogram,
    mapping_overview,
)

# Read barcodes
df_design = pd.read_csv(snakemake.params.df_design_path, index_col=None)
df_pool = df_design.drop(columns=['Unnamed: 0']).rename(columns={'target':'gene_symbol'})
df_pool['prefix_map'] = df_pool['iBAR_2']
df_pool['prefix_recomb'] = df_pool['iBAR_1'].str.slice(0,3)
barcodes = df_pool["prefix_recomb"] + df_pool["prefix_map"]

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

# _, fig = plot_reads_per_cell_histogram(cells, x_cutoff=20)
fig = plt.figure()
plt.text(0.5, 0.5, 'Placeholder figure', horizontalalignment='center')
fig.savefig(snakemake.output[7])

_, fig = plot_gene_symbol_histogram(cells)
fig.savefig(snakemake.output[8])

mapping_overview_df = mapping_overview(sbs_info, cells)
mapping_overview_df.to_csv(snakemake.output[9], sep="\t", index=False)
