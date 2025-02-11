import pandas as pd

from lib.phenotype.eval_features import plot_feature_heatmap


# Load SBS processing files
phenotype_cp_min = pd.read_hdf(snakemake.input[0])

# Generate and save feature heatmaps
min_feature_names = [col for col in phenotype_cp_min.columns if col.endswith("_min")]
for feature_name in min_feature_names:
    # Generate feature heatmap evaluation
    df_summary_one, fig = plot_feature_heatmap(
        phenotype_cp_min, feature=feature_name, shape="6W_ph", return_summary=True
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
    fig.savefig(png_path)
