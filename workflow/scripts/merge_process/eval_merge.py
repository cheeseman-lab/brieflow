import pandas as pd

from lib.merge.format_merge import identify_single_gene_mappings
from lib.merge.eval_merge import plot_sbs_ph_matching_heatmap, plot_cell_positions

# Load data for evaluating merge
merge_formatted = pd.read_hdf(snakemake.input[0])
sbs_cells = pd.read_hdf(snakemake.input[1])
phenotype_min_cp = pd.read_hdf(snakemake.input[2])

# Identify single gene mappings in SBS
sbs_cells["mapped_single_gene"] = sbs_cells.apply(
    lambda x: identify_single_gene_mappings(x), axis=1
)
mapping_counts = sbs_cells.mapped_single_gene.value_counts()
mapping_percentages = sbs_cells.mapped_single_gene.value_counts(normalize=True)
# Save cell mapping statistics
mapping_stats = pd.DataFrame(
    {
        "category": ["mapped_cells", "unmapped_cells"],
        "count": [mapping_counts[True], mapping_counts[False]],
        "percentage": [mapping_percentages[True], mapping_percentages[False]],
    }
)
mapping_stats.to_csv(snakemake.output[0], sep="\t", index=False)

# Evaluate minimal merge data
merge_minimal = merge_formatted[
    ["well", "tile", "site", "cell_0", "cell_1", "distance"]
]

# Eval SBS matching rates
sbs_summary, fig = plot_sbs_ph_matching_heatmap(
    merge_minimal,
    sbs_cells,
    target="sbs",
    shape="6W_sbs",
    return_summary=True,
)
sbs_summary.to_csv(snakemake.output[1], sep="\t", index=False)
fig.savefig(snakemake.output[2])


# Eval phenotype matching rates
ph_summary, fig = plot_sbs_ph_matching_heatmap(
    merge_minimal,
    phenotype_min_cp.rename(columns={"label": "cell_0"}),
    target="phenotype",
    shape="6W_ph",
    return_summary=True,
)
ph_summary.to_csv(snakemake.output[3], sep="\t", index=False)
fig.savefig(snakemake.output[4])

# Evaluate all formatted merge data
fig = plot_cell_positions(merge_formatted, title="All Cells by Channel Min")
fig.savefig(snakemake.output[5])
fig = plot_cell_positions(
    merge_formatted.query("channels_min==0"),
    title="Cells with Channel Min = 0",
    color="red",
)
fig.savefig(snakemake.output[6])

# TODO: Add all output targets
