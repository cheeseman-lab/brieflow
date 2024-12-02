import pandas as pd

from lib.shared.eval_segmentation import (
    segmentation_overview,
    plot_cell_density_heatmap,
)


# Get the segmentation overview
segmentation_overview_df = segmentation_overview(
    snakemake.input.segmentation_stats_paths
)
# Save the segmentation overview
segmentation_overview_df.to_csv(snakemake.output[0], sep="\t", index=False)


# load cell data
cells = pd.read_hdf(snakemake.input.cells_path)

# plot cell density heatmap
cell_density_summary, fig = plot_cell_density_heatmap(cells, shape="square", plate="6W")
cell_density_summary.to_csv(snakemake.output[1], index=False, sep="\t")
fig.savefig(snakemake.output[2])
