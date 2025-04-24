"""Benchmark utilities for evaluating gene clustering performance.

This module provides functions to evaluate clustering results using external benchmarks
such as known gene pairs and gene groups. It includes methods for calculating recall
of known gene pairs, measuring group enrichment, and visualizing benchmark performance
across different clustering parameters.

Statistical Methodology:
- Pair Recall: Evaluates clustering by measuring the fraction of known gene pairs that appear
  in the same cluster, providing a direct measure of how well the clustering preserves known
  relationships.
- Group Enrichment: Uses Fisher's exact test to identify statistically significant
  over-representation of known gene groups within clusters, followed by multiple testing
  correction using the Benjamini-Hochberg procedure to control false discovery rate.
"""

import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

from lib.cluster.phate_leiden_clustering import phate_leiden_pipeline


def calculate_pair_recall(
    pair_benchmark,
    phate_leiden_clustering,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
):
    """Calculate recall for known gene pairs within clusters.

    Evaluates how well the clustering recovers known gene-gene relationships
    by measuring the fraction of known gene pairs that are assigned to the same cluster.

    Statistical Methodology:
    Calculates the true positive rate (recall) as TP/(TP+FN), where TP is the number of
    known gene pairs assigned to the same cluster, and FN is the number of known gene
    pairs assigned to different clusters.

    Args:
        pair_benchmark (pd.DataFrame): DataFrame with 'pair' and 'gene_name' columns
            representing known gene pairs.
        phate_leiden_clustering (pd.DataFrame): Clustering results with cluster assignments.
        perturbation_col_name (str, optional): Column name for gene identifiers.
            Defaults to "gene_symbol_0".
        control_key (str, optional): Prefix for control perturbations to filter out.
            Defaults to None.

    Returns:
        float: Recall score, fraction of known gene pairs found in the same cluster.
    """
    # Filter non-targeting genes if requested
    if control_key is not None:
        cluster_df = phate_leiden_clustering[
            ~phate_leiden_clustering[perturbation_col_name].str.startswith(control_key)
        ]
    else:
        cluster_df = phate_leiden_clustering

    # 1. Convert the string pairs DataFrame to a set of positive pairs
    positive_pairs = set()

    # Group string pairs by their pair ID
    for pair_id, pair in pair_benchmark.groupby("pair"):
        # Get all genes in this pair
        genes = pair["gene_name"].unique()
        # Create all pairwise combinations from these genes and add to positive pairs
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene_pair = tuple(sorted([genes[i], genes[j]]))
                positive_pairs.add(gene_pair)

    # 2. Extract gene clusters from the clustering DataFrame
    # Create a dictionary mapping genes to their cluster
    gene_to_cluster = dict(
        zip(
            cluster_df[perturbation_col_name],
            cluster_df["cluster"],
        )
    )

    # 3. Evaluate all known positive pairs from STRING
    true_positives = 0
    false_negatives = 0

    # Only examine the pairs we know are positive according to STRING
    for gene_pair in positive_pairs:
        gene_a, gene_b = gene_pair

        # Skip pairs where either gene isn't in our clustering data
        if gene_a not in gene_to_cluster or gene_b not in gene_to_cluster:
            continue

        # Check if they're in the same cluster
        same_cluster = gene_to_cluster[gene_a] == gene_to_cluster[gene_b]

        # Update counts
        if same_cluster:
            true_positives += 1
        else:
            false_negatives += 1

    # 4. Calculate recall
    total_evaluated_pairs = true_positives + false_negatives
    recall = true_positives / total_evaluated_pairs

    return recall


def calculate_pair_recall_global(
    pair_benchmark,
    phate_leiden_clustering,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
):
    """Compute the average recall across all clusters for a pair benchmark.

    For each cluster, get recall as TP / (TP + FN) and return the overall mean recall.

    Statistical Methodology:
    Calculates recall separately for each cluster and then averages these values,
    giving equal weight to each cluster regardless of size. This approach measures
    how consistently the clustering algorithm identifies true relationships across
    all clusters.

    Args:
        pair_benchmark (pd.DataFrame): DataFrame with 'pair' and 'gene_name'.
        phate_leiden_clustering (pd.DataFrame): Clustering result with 'cluster' and gene column.
        perturbation_col_name (str, optional): Column name for gene identifiers.
            Defaults to "gene_symbol_0".
        control_key (str or None, optional): Prefix for control perturbations to filter out.
            Defaults to None.

    Returns:
        float: Average recall across all clusters.
    """
    if control_key is not None:
        cluster_df = phate_leiden_clustering[
            ~phate_leiden_clustering[perturbation_col_name].str.startswith(control_key)
        ]
    else:
        cluster_df = phate_leiden_clustering

    gene_to_cluster = dict(
        zip(cluster_df[perturbation_col_name], cluster_df["cluster"])
    )

    positive_pairs = set()
    for _, group in pair_benchmark.groupby("pair"):
        genes = group["gene_name"].unique()
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                pair = tuple(sorted((genes[i], genes[j])))
                positive_pairs.add(pair)

    from collections import defaultdict

    cluster_tp = defaultdict(int)
    cluster_fn = defaultdict(int)
    cluster_true_pairs = defaultdict(list)

    for gene_a, gene_b in positive_pairs:
        cluster_a = gene_to_cluster.get(gene_a)
        cluster_b = gene_to_cluster.get(gene_b)
        if cluster_a is None or cluster_b is None:
            continue
        if cluster_a == cluster_b:
            cluster_tp[cluster_a] += 1
            pair_str = f"{gene_a},{gene_b}"
            cluster_true_pairs[cluster_a].append(pair_str)
        else:
            cluster_fn[cluster_a] += 1
            cluster_fn[cluster_b] += 1

    cluster_sizes = cluster_df.groupby("cluster").size().to_dict()

    data = []
    for cluster_id in (
        set(cluster_tp.keys()) | set(cluster_fn.keys()) | set(cluster_sizes.keys())
    ):
        tp = cluster_tp.get(cluster_id, 0)
        fn = cluster_fn.get(cluster_id, 0)
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        size = cluster_sizes.get(cluster_id, 0)
        true_pairs_str = "; ".join(cluster_true_pairs.get(cluster_id, []))
        data.append(
            {
                "cluster": cluster_id,
                "recall": recall,
                "cluster_size": size,
                "true_pairs": true_pairs_str,
            }
        )

    pair_recall_global = (
        pd.DataFrame(data)
        .sort_values(by="recall", ascending=False)
        .reset_index(drop=True)
    )

    return pair_recall_global["recall"].mean()


def calculate_group_enrichment(
    group_benchmark,
    phate_leiden_clustering,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
):
    """Calculate enrichment of known gene groups within clusters.

    Evaluates cluster quality by measuring how many known gene groups are
    significantly enriched within each cluster using Fisher's exact test.

    Statistical Methodology:
    For each cluster-group pair, constructs a 2×2 contingency table and performs
    Fisher's exact test to determine if the group is significantly over-represented
    in the cluster. P-values are adjusted for multiple testing using the
    Benjamini-Hochberg method, and groups with adjusted p < 0.05 are considered
    significantly enriched.

    Args:
        group_benchmark (pd.DataFrame): DataFrame with 'group' and 'gene_name' columns
            representing known gene groupings.
        phate_leiden_clustering (pd.DataFrame): Clustering results with cluster assignments.
        perturbation_col_name (str, optional): Column name for gene identifiers.
            Defaults to "gene_symbol_0".
        control_key (str, optional): Prefix for control perturbations to filter out.
            Defaults to None.

    Returns:
        float: Average number of enriched groups per cluster.
    """
    print(group_benchmark)
    print(phate_leiden_clustering)
    cluster_df = phate_leiden_clustering.sort_values(by="cluster")
    if control_key is not None:
        cluster_df = phate_leiden_clustering[
            ~phate_leiden_clustering[perturbation_col_name].str.startswith(control_key)
        ]

    background_genes = set(phate_leiden_clustering[perturbation_col_name])
    group_df = group_benchmark.sort_values(by="group")

    group_to_genes = {
        group: set(df["gene_name"]) for group, df in group_df.groupby("group")
    }

    cluster_to_genes = {
        cluster: set(df[perturbation_col_name])
        for cluster, df in cluster_df.groupby("cluster")
    }

    enriched_rows = []

    for cluster_id, cluster_genes in cluster_to_genes.items():
        pvals = []
        group_ids = list(group_to_genes.keys())

        for group_id in group_ids:
            group_genes = group_to_genes[group_id]
            a = len(cluster_genes & group_genes)
            b = len(group_genes) - a
            c = len(cluster_genes) - a
            d = max(0, len(background_genes) - (a + b + c))

            contingency = [[a, b], [c, d]]
            _, p = fisher_exact(contingency)
            pvals.append(p)

        fdrs = multipletests(pvals, method="fdr_bh")[1]
        enriched = [str(g) for g, f in zip(group_ids, fdrs) if f < 0.05]
        enriched_rows.append(
            {
                "cluster": cluster_id,
                "cluster_size": len(cluster_genes),
                "num_enriched_groups": len(enriched),
                "cluster_genes": "; ".join(sorted(cluster_genes)),
                "enriched_groups": "; ".join(enriched),
            }
        )

        if cluster_id > 5:
            break

    cluster_enrichment = (
        pd.DataFrame(enriched_rows).sort_values("cluster").reset_index(drop=True)
    )

    return cluster_enrichment["num_enriched_groups"].mean()


def perform_resolution_thresholding(
    aggregated_data,
    shuffled_aggregated_data,
    phate_distance_metric,
    leiden_resolutions,
    pair_benchmark,
    perturbation_col_name,
    control_key=None,
):
    """Evaluate clustering at different Leiden resolution parameters.

    Performs clustering across multiple resolution values and evaluates recall
    performance on both real and shuffled (control) data.

    Args:
        aggregated_data (pd.DataFrame): Data matrix to cluster.
        shuffled_aggregated_data (pd.DataFrame): Shuffled data for control comparison.
        phate_distance_metric (str): Distance metric for PHATE dimensionality reduction.
        leiden_resolutions (list): List of resolution parameters to evaluate.
        pair_benchmark (pd.DataFrame): DataFrame with known gene pairs for benchmarking.
        perturbation_col_name (str): Column name for gene identifiers.
        control_key (str, optional): Prefix for control perturbations to filter out.
            Defaults to None.

    Returns:
        tuple:
            pd.DataFrame: Results with resolution, cluster count, and recall metrics.
            matplotlib.figure.Figure: Visualization of results.
    """
    # perform phate leiden clustering for each resolution
    results = []
    for resolution in leiden_resolutions:
        print(f"Creating clusters for resolution: {resolution}")

        phate_leiden_clustering = phate_leiden_pipeline(
            aggregated_data, resolution, phate_distance_metric
        )

        phate_leiden_clustering_shuffled = phate_leiden_pipeline(
            shuffled_aggregated_data, resolution, phate_distance_metric
        )

        statistics = {
            "resolution": resolution,
            "num_clusters": phate_leiden_clustering["cluster"].nunique(),
            "recall": calculate_pair_recall(
                pair_benchmark,
                phate_leiden_clustering,
                perturbation_col_name,
                control_key,
            ),
            "recall_shuffled": calculate_pair_recall(
                pair_benchmark,
                phate_leiden_clustering_shuffled,
                perturbation_col_name,
                control_key,
            ),
        }

        results.append(statistics)

    results_df = pd.DataFrame(results)

    # Create a figure with 2 subplots sharing x-axis
    fig = plt.figure(figsize=(16, 6))
    ax1 = fig.add_subplot(1, 2, 1)
    ax2 = fig.add_subplot(1, 2, 2, sharex=ax1)

    # First subplot for recall
    sns.lineplot(
        data=results_df,
        x="resolution",
        y="recall",
        marker="o",
        label="Real Data Recall",
        ax=ax1,
    )
    sns.lineplot(
        data=results_df,
        x="resolution",
        y="recall_shuffled",
        marker="s",
        label="Shuffled Data Recall",
        ax=ax1,
    )
    ax1.set_title("Recall vs Resolution")
    ax1.set_xlabel("Resolution")
    ax1.set_ylabel("Recall Score")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Second subplot for number of clusters
    sns.lineplot(
        data=results_df,
        x="resolution",
        y="num_clusters",
        marker="D",
        color="green",
        ax=ax2,
    )
    ax2.set_title("Number of Clusters vs Resolution")
    ax2.set_xlabel("Resolution")
    ax2.set_ylabel("Number of Clusters")
    ax2.grid(True, linestyle="--", alpha=0.7)

    plt.tight_layout()

    # Return without showing the plot
    return results_df, fig


def plot_benchmark_results(
    cluster_datasets,
    pair_recall_benchmarks,
    group_enrichment_benchmarks,
    perturbation_col_name,
    control_key,
):
    """Plot benchmark results for pair recall and group enrichment across clustering datasets.

    Creates a side-by-side bar plot comparing multiple clustering approaches against
    different benchmarking datasets.

    Statistical Methodology:
    Visualizes pair recall scores and group enrichment metrics across different datasets,
    enabling direct comparison of clustering approaches. The visualization helps identify
    which clustering method best captures known biological relationships and functional
    groupings.

    Args:
        cluster_datasets (dict): Mapping from dataset name to cluster DataFrame.
        pair_recall_benchmarks (dict): Mapping from benchmark name to pair benchmark DataFrame.
        group_enrichment_benchmarks (dict): Mapping from benchmark name to group benchmark DataFrame.
        perturbation_col_name (str): Column name for gene identifiers.
        control_key (str): Prefix for control perturbations to filter out.

    Returns:
        matplotlib.figure.Figure: Figure with benchmark bar plots.
    """
    # Collect data
    pair_data = []
    for benchmark_name, benchmark_df in pair_recall_benchmarks.items():
        for dataset_name, cluster_df in cluster_datasets.items():
            score = calculate_pair_recall(
                benchmark_df,
                cluster_df,
                perturbation_col_name=perturbation_col_name,
                control_key=control_key,
            )
            pair_data.append(
                {
                    "Score": score,
                    "Dataset": dataset_name,
                    "Benchmark": benchmark_name,
                    "Type": "Pair Recall",
                }
            )

    group_data = []
    for benchmark_name, benchmark_df in group_enrichment_benchmarks.items():
        for dataset_name, cluster_df in cluster_datasets.items():
            score = calculate_group_enrichment(
                benchmark_df,
                cluster_df,
                perturbation_col_name=perturbation_col_name,
                control_key=control_key,
            )
            group_data.append(
                {
                    "Score": score,
                    "Dataset": dataset_name,
                    "Benchmark": benchmark_name,
                    "Type": "Group Enrichment",
                }
            )

    df = pd.DataFrame(pair_data + group_data)

    # Plot using seaborn
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    sns.barplot(
        data=df[df["Type"] == "Pair Recall"],
        x="Dataset",
        y="Score",
        hue="Benchmark",
        ax=axes[0],
    )
    axes[0].set_title("Pair Recall")
    axes[0].set_ylabel("Recall Score")
    axes[0].tick_params(axis="x", rotation=45)

    sns.barplot(
        data=df[df["Type"] == "Group Enrichment"],
        x="Dataset",
        y="Score",
        hue="Benchmark",
        ax=axes[1],
    )
    axes[1].set_title("Group Enrichment")
    axes[1].set_ylabel("Enrichment Score")
    axes[1].tick_params(axis="x", rotation=45)

    fig.tight_layout()
    return fig
