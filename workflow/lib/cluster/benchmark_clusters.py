import pandas as pd
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests

from lib.cluster.phate_leiden_clustering import phate_leiden_pipeline


def calculate_pair_recall(
    pair_benchmark,
    phate_leiden_clustering,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
):
    # Filter non-targeting genes if requested
    if control_key is not None:
        cluster_df = phate_leiden_clustering[
            phate_leiden_clustering[perturbation_col_name] != control_key
        ]
    else:
        cluster_df = phate_leiden_clustering

    cluster_genes_in_group = len(
        set(pair_benchmark["gene_name"]) & set(cluster_df[perturbation_col_name])
    )
    print(f"Number of cluster genes found in group dataset: {cluster_genes_in_group}")

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


from matplotlib import pyplot as plt
import seaborn as sns


def perform_resolution_thresholding(
    aggregated_data,
    shuffled_aggregated_data,
    leiden_resolutions,
    pair_benchmark,
    perturbation_col_name,
    control_key=None,
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
            "recall": calculate_pair_recall(pair_benchmark, phate_leiden_clustering),
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


def calculate_group_enrichment(
    group_benchmark,
    phate_leiden_clustering,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
):
    # Sort cluster df by cluster
    cluster_df = phate_leiden_clustering.sort_values(by="cluster")

    # Get all unique genes in the cluster_df
    cluster_genes = set(cluster_df[perturbation_col_name])

    cluster_genes_in_group = len(set(group_benchmark["gene_name"]) & cluster_genes)
    print(f"Number of cluster genes found in group dataset: {cluster_genes_in_group}")

    # Filter non-targeting genes if requested
    if control_key is not None:
        cluster_df = cluster_df[cluster_df[perturbation_col_name] != control_key]

    # get background genes
    background_genes = set(cluster_df[perturbation_col_name])

    # Initialize an empty DataFrame to store the enriched groups for each gene in each cluster
    enriched_groups_df = pd.DataFrame(
        columns=["cluster", "enriched_groups", "cluster_genes"]
    )

    # Sort group df by group
    group_df = group_benchmark.sort_values(by="group")

    # Iterate over unique clusters in the DataFrame
    for cluster_id in cluster_df["cluster"].unique():
        cluster_genes = cluster_df[cluster_df["cluster"] == cluster_id][
            perturbation_col_name
        ].tolist()

        # Perform Fisher's exact test for each group and the current cluster
        group_pvalues = []
        for group_id in group_df["group"].unique():
            group_genes = group_df[group_df["group"] == group_id]["gene_name"].to_list()
            group_genes_in_cluster = len(set(cluster_genes).intersection(group_genes))
            group_genes_not_in_cluster = len(group_genes) - group_genes_in_cluster
            genes_in_cluster_not_in_group = len(cluster_genes) - group_genes_in_cluster
            genes_not_in_cluster_not_in_group = (
                len(background_genes) - len(cluster_genes) - group_genes_not_in_cluster
            )

            contingency_table = [
                [group_genes_in_cluster, group_genes_not_in_cluster],
                [genes_in_cluster_not_in_group, genes_not_in_cluster_not_in_group],
            ]

            _, p_value = fisher_exact(contingency_table)
            group_pvalues.append(p_value)
        # Perform Benjamini-Hochberg correction for multiple testing
        group_fdrs = multipletests(group_pvalues, method="fdr_bh")[1]

        # Get the unique groups and store in a list for later reference
        group_ids = group_df["group"].unique().tolist()

        # Create a DataFrame to store the results for the current cluster
        cluster_result_df = pd.DataFrame(
            {
                "cluster": cluster_id,
                "enriched_groups": "; ".join(
                    [
                        str(group_id)
                        for group_id, group_fdr in zip(group_ids, group_fdrs)
                        if group_fdr < 0.05
                    ]
                ),
                "cluster_genes": "; ".join(cluster_genes),
            },
            index=[0],
        )

        # Append the current cluster's results to the main DataFrame
        enriched_groups_df = pd.concat(
            [enriched_groups_df, cluster_result_df], ignore_index=True
        )

        # TODO: remove when want to run all clusters
        if cluster_id == 10:
            break

    # Sort enriched_groups_df by cluster number
    enriched_groups_df.sort_values(by="cluster", inplace=True)

    # Calculate the metric: Number of clusters associated with each group
    num_groups_per_cluster = enriched_groups_df["enriched_groups"].apply(
        lambda x: len(x.split(";")) if x.strip() else 0
    )

    return num_groups_per_cluster.mean()
