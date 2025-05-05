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

# add uniprot information
uniprot_data = pd.read_csv(snakemake.params.uniprot_data_fp, sep="\t")
# Expand the rows of the uniprot data to deal with synonyms and homologs
uniprot_data = uniprot_data.dropna(subset=["gene_names"])
expanded_rows = []
for _, row in uniprot_data.iterrows():
    gene_names = row["gene_names"].split()
    for position, gene in enumerate(gene_names):
        expanded_rows.append(
            {
                "gene_name": gene,
                "position": position,
                "uniprot_entry": row["entry"],
                "uniprot_function": row["function"],
                "uniprot_link": row["link"],
            }
        )
expanded_df = pd.DataFrame(expanded_rows)
uniprot_data = expanded_df.sort_values(["gene_name", "position"]).drop_duplicates(
    "gene_name", keep="first"
)
uniprot_data = uniprot_data[
    ["gene_name", "uniprot_entry", "uniprot_function", "uniprot_link"]
]
# merge uniprot data with clustering results
phate_leiden_clustering = phate_leiden_clustering.merge(
    uniprot_data, how="left", left_on="gene_symbol_0", right_on="gene_name"
).drop(columns="gene_name")

# save clustering results
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
