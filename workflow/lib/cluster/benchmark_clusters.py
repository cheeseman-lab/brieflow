"""Module for analyzing clusters generated during the clustering module.

This module provides functions to analyze, interpret, and validate clusters. It supports creating cluster-specific
gene tables, identifying differential features between clusters, and validating clusters against external databases
such as STRING and CORUM. The results include both detailed cluster metrics and global metrics for performance evaluation.

Functions:
    - create_cluster_gene_table: Generate a table summarizing clusters with combined gene information and counts.
    - analyze_differential_features: Identify and analyze differential features between clusters.
    - process_interactions: Validate cluster data against STRING and CORUM databases, providing enrichment and interaction metrics.
"""

from itertools import combinations

import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, fisher_exact
from statsmodels.stats.multitest import multipletests


def calculate_pair_recall(
    pair_benchmark, phate_leiden_clustering, pertubration_col_name
):
    # 1. Convert the string pairs DataFrame to a set of positive pairs
    positive_pairs = set()

    # Group string pairs by their pair ID
    for pair_id, group in pair_benchmark.groupby("pair"):
        # Get all genes in this pair
        genes = group["gene_name"].unique()
        # Create all pairwise combinations from these genes and add to positive pairs
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene_pair = tuple(sorted([genes[i], genes[j]]))
                positive_pairs.add(gene_pair)

    # 2. Extract gene clusters from the clustering DataFrame
    # Create a dictionary mapping genes to their cluster
    gene_to_cluster = dict(
        zip(
            phate_leiden_clustering["gene_symbol_0"],  # Using the correct column name
            phate_leiden_clustering["cluster"],
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


from matplotlib import pyplot as plt
import seaborn as sns


def perform_resolution_thresholding(
    aggregated_data, shuffled_aggregated_data, leiden_resolutions
):
    # perform phate leiden clustering for each resolution
    results = []
    for resolution in leiden_resolutions:
        print(f"Creating clusters for resolution: {resolution}")

        phate_leiden_clustering = phate_leiden_pipeline(
            aggregated_data, resolution=resolution
        )

        phate_leiden_clustering_shuffled = phate_leiden_pipeline(
            shuffled_aggregated_data, resolution=resolution
        )

        statistics = {
            "resolution": resolution,
            "num_clusters": phate_leiden_clustering["cluster"].nunique(),
            "recall": calculate_pair_recall(
                string_pair_benchmark, phate_leiden_clustering
            ),
            "recall_shuffled": calculate_pair_recall(
                string_pair_benchmark, phate_leiden_clustering_shuffled
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
        label="Real Recall",
        ax=ax1,
    )
    sns.lineplot(
        data=results_df,
        x="resolution",
        y="recall_shuffled",
        marker="s",
        label="Shuffled Recall",
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
