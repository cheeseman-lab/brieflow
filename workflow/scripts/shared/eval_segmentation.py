import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    }
)

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
cells = pd.concat(
    [pd.read_parquet(p) for p in snakemake.input.cells_paths], ignore_index=True
)

# Load metadata for spatial heatmap plotting
metadata = pd.concat(
    [pd.read_parquet(p) for p in snakemake.input.metadata_paths], ignore_index=True
).drop_duplicates(subset=["well", "tile"])

# plot cell density heatmap
cell_density_summary, fig = plot_cell_density_heatmap(
    cells,
    metadata=metadata,
)
cell_density_summary.to_csv(snakemake.output[1], index=False, sep="\t")
fig.savefig(snakemake.output[2], dpi=300, bbox_inches="tight", transparent=True)
