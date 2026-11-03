import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Nimbus Sans", "Liberation Sans", "DejaVu Sans"],
    }
)

from lib.phenotype.eval_features import plot_feature_heatmap
from lib.shared.parquet_io import read_parquets


# Load phenotype processing files
phenotype_cp_min = read_parquets(snakemake.input.cells_paths)

# Load metadata for spatial heatmap plotting
metadata = read_parquets(snakemake.input.metadata_paths).drop_duplicates(
    subset=["well", "tile"]
)

# Generate and save feature heatmaps
min_feature_names = [col for col in phenotype_cp_min.columns if col.endswith("_min")]
for feature_name in min_feature_names:
    # Generate feature heatmap evaluation
    df_summary_one, fig = plot_feature_heatmap(
        phenotype_cp_min,
        feature=feature_name,
        metadata=metadata,
        return_summary=True,
    )

    # Determine the save paths from snakemake.output
    tsv_path = next(
        path
        for path in snakemake.output
        if feature_name in path and path.endswith(".tsv")
    )
    png_path = next(
        path
        for path in snakemake.output
        if feature_name in path and path.endswith(".png")
    )

    # Save df summary and figure
    df_summary_one.to_csv(tsv_path, index=False, sep="\t")
    fig.savefig(png_path, dpi=300, bbox_inches="tight", transparent=True)
