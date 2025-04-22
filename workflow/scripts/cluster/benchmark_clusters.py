import pandas as pd
import numpy as np

from lib.cluster.phate_leiden_clustering import phate_leiden_pipeline
from lib.cluster.benchmark_clusters import plot_benchmark_results

aggregated_data = pd.read_csv(snakemake.input[0], sep="\t")

# create baseline data by shuffling columns independently
shuffled_aggregated_data = aggregated_data.copy()
feature_start_idx = shuffled_aggregated_data.columns.get_loc("PC_0")
feature_cols = shuffled_aggregated_data.columns[feature_start_idx:]
for col in feature_cols:
    shuffled_aggregated_data[col] = np.random.permutation(
        shuffled_aggregated_data[col].values
    )

phate_leiden_clustering = pd.read_csv(snakemake.input[1], sep="\t")

phate_leiden_clustering_shuffled = phate_leiden_pipeline(
    shuffled_aggregated_data,
    int(snakemake.params.leiden_resolution),
    snakemake.params.phate_distance_metric,
)

cluster_datasets = {
    "Real": phate_leiden_clustering,
    "Shuffled": phate_leiden_clustering_shuffled,
}

string_pair_benchmark = pd.read_csv(snakemake.params.string_pair_benchmark_fp, sep="\t")
pair_recall_benchmarks = {
    "STRING": string_pair_benchmark,
}

corum_group_benchmark = pd.read_csv(snakemake.params.corum_group_benchmark_fp, sep="\t")
kegg_group_benchmark = pd.read_csv(snakemake.params.kegg_group_benchmark_fp, sep="\t")
group_enrichment_benchmarks = {
    "CORUM": corum_group_benchmark,
    "KEGG": kegg_group_benchmark,
}

benchmark_results_fig = plot_benchmark_results(
    cluster_datasets,
    pair_recall_benchmarks,
    group_enrichment_benchmarks,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
)
benchmark_results_fig.savefig(snakemake.output[0])
