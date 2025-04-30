import pandas as pd
import numpy as np

from lib.cluster.phate_leiden_clustering import phate_leiden_pipeline
from lib.cluster.benchmark_clusters import (
    run_benchmark_analysis,
    save_json_results,
)
from lib.cluster.scrape_benchmarks import (
    simplify_ampersand_genes,
    filter_complexes,
)

aggregated_data = pd.read_csv(snakemake.input[0], sep="\t")
phate_leiden_clustering = pd.read_csv(snakemake.input[1], sep="\t")

# create baseline data by shuffling columns independently
shuffled_aggregated_data = aggregated_data.copy()
feature_start_idx = shuffled_aggregated_data.columns.get_loc("PC_0")
feature_cols = shuffled_aggregated_data.columns[feature_start_idx:]
for col in feature_cols:
    shuffled_aggregated_data[col] = np.random.permutation(
        shuffled_aggregated_data[col].values
    )

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
    "CORUM": filter_complexes(
        corum_group_benchmark,
        phate_leiden_clustering,
        snakemake.params.perturbation_name_col,
        snakemake.params.control_key,
    ),
    "KEGG": filter_complexes(
        kegg_group_benchmark,
        phate_leiden_clustering,
        snakemake.params.perturbation_name_col,
        snakemake.params.control_key,
    ),
}

(
    integrated_results,
    combined_tables,
    global_metrics,
    enrichment_pie_charts,
    enrichment_bar_charts,
) = run_benchmark_analysis(
    cluster_datasets,
    string_pair_benchmark,
    corum_group_benchmark,
    kegg_group_benchmark,
    snakemake.params.perturbation_name_col,
    snakemake.params.control_key,
)

# Save the results
save_json_results(integrated_results["Real"], snakemake.output[0])
save_json_results(integrated_results["Shuffled"], snakemake.output[1])
combined_tables["Real"].to_csv(snakemake.output[2], sep="\t", index=False)
combined_tables["Shuffled"].to_csv(snakemake.output[3], sep="\t", index=False)
save_json_results(global_metrics["Shuffled"], snakemake.output[4])
save_json_results(global_metrics["Shuffled"], snakemake.output[5])
enrichment_pie_charts["Real"].savefig(snakemake.output[6])
enrichment_pie_charts["Shuffled"].savefig(snakemake.output[7])
enrichment_bar_charts["Real"].savefig(snakemake.output[8])
enrichment_bar_charts["Shuffled"].savefig(snakemake.output[9])
