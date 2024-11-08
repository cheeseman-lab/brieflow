import pandas as pd
import matplotlib.pyplot as plt

from lib.sbs_process.eval import (
    plot_mapping_vs_threshold,
    plot_mapping_vs_threshold,
    plot_read_mapping_heatmap,
    plot_cell_mapping_heatmap,
    plot_reads_per_cell_histogram,
    plot_gene_symbol_histogram,
)

# Read barcodes
df_design = pd.read_csv(snakemake.params.df_design_path, sep="\t")
df_pool = df_design.query("dialout==[0,1]").drop_duplicates("sgRNA")
df_pool["prefix"] = df_pool.apply(lambda x: x.sgRNA[: x.prefix_length], axis=1)
barcodes = df_pool["prefix"]

# Concatenate files
reads = pd.read_hdf(snakemake.input[0])
cells = pd.read_hdf(snakemake.input[1])
minimal_phenotype_info = pd.read_hdf(snakemake.input[2])

_, fig = plot_mapping_vs_threshold(reads, barcodes, "peak")
fig.savefig(snakemake.output[0])

_, fig = plot_mapping_vs_threshold(reads, barcodes, "Q_min")
fig.savefig(snakemake.output[1])

fig = plot_read_mapping_heatmap(reads, barcodes, shape="6W_sbs")
fig.savefig(snakemake.output[2])

df_summary_one, fig = plot_cell_mapping_heatmap(
    cells,
    minimal_phenotype_info,
    barcodes,
    mapping_to="one",
    mapping_strategy="gene_symbols",
    shape="6W_sbs",
    return_summary=True,
)
df_summary_one.to_csv(snakemake.output[3], index=False, sep="\t")
fig.savefig(snakemake.output[4])

df_summary_any, fig = plot_cell_mapping_heatmap(
    cells,
    minimal_phenotype_info,
    barcodes,
    mapping_to="any",
    mapping_strategy="gene_symbols",
    shape="6W_sbs",
    return_summary=True,
)
df_summary_any.to_csv(snakemake.output[5], index=False, sep="\t")
fig.savefig(snakemake.output[6])

_, fig = plot_reads_per_cell_histogram(cells, x_cutoff=20)
fig.savefig(snakemake.output[7])

_, fig = plot_gene_symbol_histogram(cells, x_cutoff=30)
fig.savefig(snakemake.output[8])

# Calculate and print mapped single gene statistics
print("Calculating mapped single gene statistics...")
cells["mapped_single_gene"] = cells.apply(
    lambda x: (
        True
        if (pd.notnull(x.gene_symbol_0) & pd.isnull(x.gene_symbol_1))
        | (x.gene_symbol_0 == x.gene_symbol_1)
        else False
    ),
    axis=1,
)
print(cells.mapped_single_gene.value_counts())

num_rows = len(minimal_phenotype_info)

with open(snakemake.output[9], "w") as eval_stats_file:
    eval_stats_file.write(f"Number of cells extracted in sbs step: {num_rows}\n")
    eval_stats_file.write("Mapped single gene statistics:\n")
    eval_stats_file.write(cells.mapped_single_gene.value_counts().to_string())
