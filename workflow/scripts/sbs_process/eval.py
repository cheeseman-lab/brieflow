import pandas as pd
import matplotlib.pyplot as plt

from lib.sbs_process.eval import (
    load_and_concatenate_hdfs,
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
print("Concatenating files...")
reads = load_and_concatenate_hdfs(snakemake.input.read_files)
cells = load_and_concatenate_hdfs(snakemake.input.cell_files)
minimal_phenotype_info = load_and_concatenate_hdfs(
    snakemake.input.minimal_phenotype_info_files
)

# Generate plots
print("Generating plots...")

plot_mapping_vs_threshold(reads, barcodes, "peak")
plt.gcf().savefig(snakemake.output[0])
plt.close()

plot_mapping_vs_threshold(reads, barcodes, "Q_min")
plt.gcf().savefig(snakemake.output[1])
plt.close()

plot_read_mapping_heatmap(reads, barcodes, shape="6W_sbs")
plt.gcf().savefig(snakemake.output[2])
plt.close()

df_summary_one, _ = plot_cell_mapping_heatmap(
    cells,
    minimal_phenotype_info,
    barcodes,
    mapping_to="one",
    mapping_strategy="gene_symbols",
    shape="6W_sbs",
    return_summary=True,
)
df_summary_one.to_csv(snakemake.output[3], index=False, sep="\t")
plt.gcf().savefig(snakemake.output[4])
plt.close()

df_summary_any, _ = plot_cell_mapping_heatmap(
    cells,
    minimal_phenotype_info,
    barcodes,
    mapping_to="any",
    mapping_strategy="gene_symbols",
    shape="6W_sbs",
    return_summary=True,
)
df_summary_any.to_csv(snakemake.output[5], index=False, sep="\t")
plt.gcf().savefig(snakemake.output[6])
plt.close()

outliers = plot_reads_per_cell_histogram(cells, x_cutoff=20)
plt.savefig(snakemake.output[7])
plt.close()

outliers = plot_gene_symbol_histogram(cells, x_cutoff=30)
plt.gcf().savefig(snakemake.output[8])
plt.close()

num_rows = len(minimal_phenotype_info)
print(f"The number of cells extracted in the sbs step is: {num_rows}")

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

print("QC analysis completed.")
