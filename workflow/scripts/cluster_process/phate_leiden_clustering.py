import pandas as pd

from lib.cluster.phate_leiden_clustering import (
    select_features,
    normalize_to_controls,
    perform_pca_analysis,
    phate_leiden_pipeline,
    dimensionality_reduction,
    merge_phate_uniprot,
)

# load cleaned gene data
cleaned_data = pd.read_csv(snakemake.input[0], sep="\t")

# select features
filtered_data, removed_features = select_features(
    cleaned_data,
    correlation_threshold=snakemake.params.correlation_threshold,
    variance_threshold=snakemake.params.variance_threshold,
    min_unique_values=snakemake.params.min_unique_values,
)

# normalize filtered data
normalized_data = normalize_to_controls(filtered_data, snakemake.params.control_prefix)

# threshold data with pca
pca_thresholded_data, n_components, pca = perform_pca_analysis(
    normalized_data,
    save_plot_path=snakemake.output[0],
)

# perform phate leiden clustering
phate_leiden_clustering = phate_leiden_pipeline(
    pca_thresholded_data, resolution=snakemake.params.leiden_resolution
)

# create and save plot with phate leiden clustering
dimensionality_reduction(
    phate_leiden_clustering,
    x="PHATE_0",
    y="PHATE_1",
    control_query=f'{snakemake.params.population_feature}.str.startswith("{snakemake.params.control_prefix}")',
    control_color="lightgray",
    control_legend=True,
    label_query=f'~{snakemake.params.population_feature}.str.startswith("{snakemake.params.control_prefix}")',
    label_hue="cluster",
    label_palette="husl",
    s=25,
    hide_axes=False,
    label_legend=False,
    legend_kwargs={"loc": "center left", "bbox_to_anchor": (1, 0.5)},
    save_plot_path=snakemake.output[1],
)

# add uniprot data to cluster data
phate_leiden_uniprot = merge_phate_uniprot(
    phate_leiden_clustering, snakemake.params.uniprot_data_fp
)
phate_leiden_uniprot.to_csv(snakemake.output[2], sep="\t", index=False)
