import pandas as pd

from lib.cluster.benchmark_clusters import (
    create_cluster_gene_table,
    analyze_differential_features,
    process_interactions,
)

# load cluster data with uniprot annotations
phate_leiden_uniprot = pd.read_csv(snakemake.input[0], sep="\t")

# create cluster gene table
cluster_gene_table = create_cluster_gene_table(
    phate_leiden_uniprot,
    columns_to_combine=[snakemake.params.population_feature, "STRING"],
)

# analyze differential features
cleaned_gene_data = pd.read_csv(snakemake.input[1], sep="\t")
cluster_gene_table, diff_results = analyze_differential_features(
    cluster_gene_table, cleaned_gene_data
)

# process interactions and get enrichment results
cluster_gene_table, global_metrics = process_interactions(
    cluster_gene_table, snakemake.params.string_data_fp, snakemake.params.corum_data_fp
)

# save cluster analysis results
cluster_gene_table.to_csv(snakemake.output[0], sep="\t", index=False)
global_metrics.to_csv(snakemake.output[1], sep="\t", index=False)
