import pandas as pd

from lib.sbs_process.eval_mapping import (
    plot_mapping_vs_threshold,
    plot_mapping_vs_threshold,
    plot_read_mapping_heatmap,
    plot_cell_mapping_heatmap,
    plot_reads_per_cell_histogram,
    plot_gene_symbol_histogram,
    mapping_overview,
)

# Read barcodes
df_design = pd.read_csv(snakemake.params.df_design_path, sep="\t")
df_pool = df_design.query("dialout==[0,1]").drop_duplicates("sgRNA")
df_pool["prefix"] = df_pool.apply(lambda x: x.sgRNA[: x.prefix_length], axis=1)
barcodes = df_pool["prefix"]

# Concatenate files
reads = pd.read_hdf(snakemake.input[0])
cells = pd.read_hdf(snakemake.input[1])
sbs_info = pd.read_hdf(snakemake.input[2])

_, fig = plot_mapping_vs_threshold(reads, barcodes, "peak")
fig.savefig(snakemake.output[0])

_, fig = plot_mapping_vs_threshold(reads, barcodes, "Q_min")
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

_, fig = plot_reads_per_cell_histogram(cells, x_cutoff=20)
fig.savefig(snakemake.output[7])

_, fig = plot_gene_symbol_histogram(cells, x_cutoff=30)
fig.savefig(snakemake.output[8])

mapping_overview_df = mapping_overview(sbs_info, cells)
mapping_overview_df.to_csv(snakemake.output[9], sep="\t", index=False)