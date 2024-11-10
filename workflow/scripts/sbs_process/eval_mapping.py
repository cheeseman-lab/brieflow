import pandas as pd

from lib.sbs_process.eval_mapping import (
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


def mapping_overview(sbs_info, cells):
    # Count the total number of cells per well
    cell_counts = sbs_info.groupby("well").size().reset_index(name="total_cells__count")

    # Count and calculate percent of cells with 1 barcode mapping per well
    one_barcode_mapping = (
        cells[cells["barcode_count"] == 1]
        .groupby("well")
        .size()
        .reset_index(name="1_barcode_cells__count")
    )
    one_barcode_mapping["1_barcode_cells__percent"] = (
        one_barcode_mapping["1_barcode_cells__count"]
        / cell_counts["total_cells__count"]
        * 100
    )

    # Count and calculate percent of cells with >=1 barcode mapping per well
    multiple_barcode_mapping = (
        cells[cells["barcode_count"] >= 1]
        .groupby("well")
        .size()
        .reset_index(name="1_or_more_barcodes__count")
    )
    multiple_barcode_mapping["1_or_more_barcodes__percent"] = (
        multiple_barcode_mapping["1_or_more_barcodes__count"]
        / cell_counts["total_cells__count"]
        * 100
    )

    # Count and calculate percent of cells with 1 gene symbol mapping per well
    one_gene_mapping = (
        cells[(~cells["gene_symbol_0"].isna()) & (cells["gene_symbol_1"].isna())]
        .groupby("well")
        .size()
        .reset_index(name="1_gene_cells__count")
    )
    one_gene_mapping["1_gene_cells__percent"] = (
        one_gene_mapping["1_gene_cells__count"]
        / cell_counts["total_cells__count"]
        * 100
    )

    # Count and calculate percent of cells with >=1 gene symbol mapping per well
    multiple_gene_mapping = (
        cells[(~cells["gene_symbol_0"].isna()) | (~cells["gene_symbol_1"].isna())]
        .groupby("well")
        .size()
        .reset_index(name="1_or_more_genes__count")
    )
    multiple_gene_mapping["1_or_more_genes__percent"] = (
        multiple_gene_mapping["1_or_more_genes__count"]
        / cell_counts["total_cells__count"]
        * 100
    )

    # Merge all counts and percents into a single DataFrame
    overview_df = (
        cell_counts.merge(one_barcode_mapping, on="well", how="left")
        .merge(multiple_barcode_mapping, on="well", how="left")
        .merge(one_gene_mapping, on="well", how="left")
        .merge(multiple_gene_mapping, on="well", how="left")
    )

    # Fill NaN values with 0 (for cases where no cells meet criteria)
    overview_df.fillna(0, inplace=True)

    return overview_df


# Get the mapping overview
mapping_overview_df = mapping_overview(sbs_info, cells)
# Save the mapping overview
mapping_overview_df.to_csv(snakemake.output[9], sep="\t", index=False)
