import pandas as pd

from lib.cluster.cluster_eval import plot_cluster_sizes
from lib.cluster.phate_leiden_clustering import (
    phate_leiden_pipeline,
    plot_phate_leiden_clusters,
)

# load aggregated data
aggregated_data = pd.read_csv(snakemake.input[0], sep="\t")

# cluster aggregated data
phate_leiden_clustering = phate_leiden_pipeline(
    aggregated_data,
    int(snakemake.params.leiden_resolution),
    snakemake.params.phate_distance_metric,
)
phate_leiden_clustering.to_csv(snakemake.output[0], sep="\t", index=False)

# plot cluster sizes
cluster_size_fig = plot_cluster_sizes(phate_leiden_clustering)
cluster_size_fig.savefig(snakemake.output[1])

# plot clusters
clusters_fig = plot_phate_leiden_clusters(
    phate_leiden_clustering,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
)
clusters_fig.savefig(snakemake.output[2])
