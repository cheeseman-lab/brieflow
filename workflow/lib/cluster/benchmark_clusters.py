"""Benchmark evaluation and visualization of gene clustering against curated datasets.

This module contains methods to benchmark gene clustering results against
biological ground truth datasets such as STRING, CORUM, and KEGG. It includes
functions for computing enrichment of known gene groups and gene pairs,
integrating metrics across datasets, and generating summary visualizations.

Key functions:
    - run_benchmark_analysis: Perform integrated benchmarking and plotting.
    - evaluate_resolution: Compare clustering resolutions using PR metrics.
    - calculate_group_enrichment: Enrich gene groups within clusters.
    - calculate_pair_enrichment: Compute pairwise precision and recall.
    - run_integrated_benchmarks: Consolidate enrichment and performance metrics.
    - generate_combined_enrichment_table: Assemble detailed benchmark table.
    - enrichment_pie_chart: Visualize enrichment category distribution.
    - enrichment_bar_chart: Plot enrichment scores by cluster.
    - save_json_results: Export data with type-safe JSON serialization.
"""

import json

import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
from statsmodels.stats.multitest import multipletests
import seaborn as sns
import matplotlib.pyplot as plt

from lib.cluster.phate_leiden_clustering import phate_leiden_pipeline
from lib.cluster.scrape_benchmarks import filter_complexes


def run_benchmark_analysis(
    cluster_datasets,
    string_pair_benchmark,
    corum_group_benchmark,
    kegg_group_benchmark,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
    max_clusters=None,
):
    """Run comprehensive benchmark analysis and generate outputs.

    Args:
        cluster_datasets (dict): Dictionary of cluster datasets.
        string_pair_benchmark (pd.DataFrame): STRING pair benchmark.
        corum_group_benchmark (pd.DataFrame): CORUM group benchmark.
        kegg_group_benchmark (pd.DataFrame): KEGG group benchmark.
        perturbation_col_name (str): Column name for gene identifiers.
        control_key (str): Prefix for control perturbations.
        max_clusters (int): Maximum clusters to analyze.

    Returns:
        tuple: (integrated_results, combined_tables, metrics, pie_charts, cluster_enrichment_plots)
    """
    # Set up benchmarks
    pair_benchmarks = {"STRING": string_pair_benchmark}
    group_benchmarks = {"CORUM": corum_group_benchmark, "KEGG": kegg_group_benchmark}

    # Run integrated benchmarks
    print("Running integrated benchmarks...")
    integrated_results, cluster_details, global_metrics = run_integrated_benchmarks(
        cluster_datasets,
        pair_benchmarks,
        group_benchmarks,
        perturbation_col_name,
        control_key,
        max_clusters,
    )

    # Generate combined tables
    print("Generating combined tables...")
    combined_tables = {}
    for dataset_name in cluster_datasets.keys():
        combined_tables[dataset_name] = generate_combined_enrichment_table(
            integrated_results,
            dataset_name,
            cluster_datasets[dataset_name],  # Pass the original dataframe
        )

    # Generate visualizations
    print("Generating visualizations...")
    pie_charts = {}
    cluster_enrichment_plots = {}

    for dataset_name in cluster_datasets.keys():
        # Create single pie chart with both views (with and without unenriched)
        pie_charts[dataset_name] = enrichment_pie_chart(
            integrated_results, dataset_name
        )

        # Generate cluster enrichment plots
        cluster_enrichment_plots[dataset_name] = enrichment_bar_chart(
            combined_tables[dataset_name]
        )

    # Print summary metrics
    print("\nSummary Metrics:")
    for dataset_name in cluster_datasets.keys():
        print(f"\n{dataset_name}:")

        # Group enrichment
        for benchmark in group_benchmarks.keys():
            if benchmark in global_metrics[dataset_name]:
                metrics = global_metrics[dataset_name][benchmark]
                print(
                    f"  {benchmark} num enriched clusters: {metrics['num_enriched_clusters']}"
                )
                print(
                    f"  {benchmark} proportion enriched clusters: {metrics['proportion_enriched']:.2f}"
                )

        # Pair precision-recall
        for benchmark in pair_benchmarks.keys():
            if benchmark in global_metrics[dataset_name]:
                metrics = global_metrics[dataset_name][benchmark]
                print(f"  {benchmark} precision: {metrics['precision']:.4f}")
                print(f"  {benchmark} recall: {metrics['recall']:.4f}")
                print(f"  {benchmark} F1 score: {metrics['f1_score']:.4f}")

    return (
        integrated_results,
        combined_tables,
        global_metrics,
        pie_charts,
        cluster_enrichment_plots,
    )


def evaluate_resolution(
    aggregated_data,
    phate_distance_metric,
    leiden_resolutions,
    group_benchmarks,
    perturbation_col_name,
    control_key=None,
):
    """Evaluate clustering at different Leiden resolution parameters using pair benchmarks.

    Performs clustering across multiple resolution values and evaluates
    precision and recall performance using pair benchmarks derived from group data.

    Args:
        aggregated_data (pd.DataFrame): Data matrix to cluster.
        phate_distance_metric (str): Distance metric for PHATE dimensionality reduction.
        leiden_resolutions (list): List of resolution parameters to evaluate.
        group_benchmarks (dict): Dictionary of benchmark DataFrames with known gene groups. Must have a group column.
        perturbation_col_name (str): Column name for gene identifiers.
        control_key (str, optional): Prefix for control perturbations to filter out.

    Returns:
        tuple:
            pd.DataFrame: Results with resolution, cluster count, and precision-recall metrics.
            matplotlib.figure.Figure: Visualization of precision-recall curves.
    """
    # Initialize data structures to store results
    all_results = {}

    # Set up a dictionary to hold results for each benchmark
    for benchmark_name in group_benchmarks.keys():
        all_results[benchmark_name] = []

    # Perform clustering for each resolution
    for resolution in leiden_resolutions:
        print(f"Creating clusters for resolution: {resolution}")

        # Generate clustering for this resolution
        clustering = phate_leiden_pipeline(
            aggregated_data, resolution, phate_distance_metric
        )

        # For each benchmark, calculate metrics
        for benchmark_name, group_df in group_benchmarks.items():
            # Filter the benchmark if needed (for group benchmarks)
            filtered_group_benchmark = filter_complexes(
                group_df,
                clustering,
                perturbation_col_name=perturbation_col_name,
                control_key=control_key,
            )

            # Convert group benchmark to pair benchmark format
            pair_rows = []
            for group_name, group_data in filtered_group_benchmark.groupby("group"):
                genes = group_data["gene_name"].unique()
                # Create all possible pairs within this group
                for i in range(len(genes)):
                    for j in range(i + 1, len(genes)):
                        pair_rows.append(
                            {"pair": group_name, "gene_name": str(genes[i])}
                        )
                        pair_rows.append(
                            {"pair": group_name, "gene_name": str(genes[j])}
                        )

            # Create pair benchmark DataFrame
            pair_benchmark = pd.DataFrame(pair_rows)

            # Skip if no pairs were created
            if len(pair_benchmark) == 0:
                continue

            # Calculate adjusted precision/recall metrics
            adjusted_metrics = calculate_pair_enrichment(
                pair_benchmark,
                clustering,
                perturbation_col_name,
                control_key,
                max_clusters=None,
                return_cluster_details=False,
                adjust_precision=True,
            )

            # Collect statistics
            statistics = {
                "resolution": resolution,
                "num_clusters": clustering["cluster"].nunique(),
                "precision": adjusted_metrics["adjusted_precision"],
                "recall": adjusted_metrics["recall"],
                "f1_score": adjusted_metrics["adjusted_f1_score"],
                "benchmark": benchmark_name,
            }

            all_results[benchmark_name].append(statistics)

    # Combine all results into a single DataFrame
    combined_results = []
    for benchmark_name, results in all_results.items():
        if results:  # Skip empty results
            benchmark_df = pd.DataFrame(results)
            benchmark_df["benchmark"] = benchmark_name
            combined_results.append(benchmark_df)

    results_df = (
        pd.concat(combined_results, ignore_index=True)
        if combined_results
        else pd.DataFrame()
    )

    # If no results, return early
    if results_df.empty:
        return results_df, None

    # Create visualization figure - smaller figure size for cleaner presentation
    fig, ax = plt.subplots(dpi=300)

    # Define colors and markers for each benchmark - using more standard colors
    benchmark_styles = {
        "CORUM": {"color": "#1f77b4", "marker": "o"},  # blue
        "KEGG": {"color": "#ff7f0e", "marker": "s"},  # orange
    }

    # Plot each benchmark
    for benchmark_name, style in benchmark_styles.items():
        if benchmark_name in all_results and all_results[benchmark_name]:
            # Get data for this benchmark
            benchmark_df = pd.DataFrame(all_results[benchmark_name])
            sorted_df = benchmark_df.sort_values(by="resolution")

            # Plot adjusted precision curve
            ax.plot(
                sorted_df["recall"],
                sorted_df["precision"],
                color=style["color"],
                linestyle="-",
                linewidth=2,
                label=f"{benchmark_name}",
            )

            # Add points with resolution labels
            for _, row in sorted_df.iterrows():
                ax.scatter(
                    row["recall"],
                    row["precision"],
                    color=style["color"],
                    s=60,
                    marker=style["marker"],
                    zorder=5,
                )

                # Add resolution labels next to points
                ax.annotate(
                    f"{row['resolution']}",
                    (row["recall"], row["precision"]),
                    xytext=(5, 0),
                    textcoords="offset points",
                    fontsize=9,
                    ha="left",
                    va="center",
                )

    # Customize plot for cleaner appearance
    ax.set_title("Precision-Recall by Resolution", fontsize=14)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)

    # Set axis limits with some padding
    ax.set_xlim(0, max(results_df["recall"]) * 1.05)
    ax.set_ylim(0, max(results_df["precision"]) * 1.05)

    # Add grid but make it subtle
    ax.grid(True, linestyle="--", alpha=0.3)

    # Add legend with cleaner formatting
    ax.legend(fontsize=10, framealpha=0.7, loc="best")

    # Use tight layout to maximize space usage
    plt.tight_layout()

    return results_df, fig


def calculate_group_enrichment(
    group_benchmark,
    phate_leiden_clustering,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
    return_full_table=False,
    max_clusters=None,
):
    """Calculate enrichment of known gene groups within clusters.

    Args:
        group_benchmark (pd.DataFrame): DataFrame with 'group' and 'gene_name' columns.
        phate_leiden_clustering (pd.DataFrame): Clustering results.
        perturbation_col_name (str, optional): Column name for gene identifiers.
        control_key (str, optional): Prefix for control perturbations to filter out.
        return_full_table (bool, optional): Whether to return the full table.
        max_clusters (int, optional): Maximum number of clusters to analyze before stopping.

    Returns:
        Union[float, pd.DataFrame]: Either the average number of enriched groups
            per cluster, or the full enrichment table.
    """
    cluster_df = phate_leiden_clustering.sort_values(by="cluster")
    if control_key is not None:
        cluster_df = cluster_df[
            ~cluster_df[perturbation_col_name].str.startswith(control_key)
        ]

    background_genes = set(phate_leiden_clustering[perturbation_col_name])
    group_df = group_benchmark.sort_values(by="group")
    group_to_genes = {
        group: set(df["gene_name"]) for group, df in group_df.groupby("group")
    }

    # Get clusters sorted by size (descending) and limit to max_clusters
    cluster_sizes = cluster_df.groupby("cluster").size().sort_values(ascending=False)
    if max_clusters is not None:
        cluster_ids = cluster_sizes.index[:max_clusters]
    else:
        cluster_ids = cluster_sizes.index

    # Only process selected clusters
    cluster_to_genes = {
        cluster: set(df[perturbation_col_name])
        for cluster, df in cluster_df.groupby("cluster")
        if cluster in cluster_ids
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
            d = len(background_genes) - (a + b + c)

            contingency = [[a, b], [c, d]]
            _, p = fisher_exact(contingency)
            pvals.append(p)

        # BH correction for each cluster individually
        fdrs = multipletests(pvals, method="fdr_bh")[1]
        enriched = [f"{g}" for g, f in zip(group_ids, fdrs) if f < 0.05]

        enriched_rows.append(
            {
                "cluster": cluster_id,
                "cluster_size": len(cluster_genes),
                "num_enriched_groups": len(enriched),
                "enriched_groups": "; ".join(enriched) if enriched else "",
                "cluster_genes": "; ".join(sorted(cluster_genes)),
            }
        )

    cluster_enrichment = (
        pd.DataFrame(enriched_rows).sort_values("cluster").reset_index(drop=True)
    )

    if return_full_table:
        return cluster_enrichment
    else:
        return cluster_enrichment["num_enriched_groups"].mean()


def calculate_pair_enrichment(
    pair_benchmark,
    phate_leiden_clustering,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
    max_clusters=None,
    return_cluster_details=False,
    adjust_precision=False,
):
    """Calculate precision and recall for gene pairs within clusters with early stopping.

    Args:
        pair_benchmark (pd.DataFrame): DataFrame with 'pair' and 'gene_name' columns.
        phate_leiden_clustering (pd.DataFrame): Clustering results.
        perturbation_col_name (str, optional): Column name for gene identifiers.
        control_key (str, optional): Prefix for control perturbations to filter out.
        max_clusters (int, optional): Maximum number of clusters to analyze.
        return_cluster_details (bool, optional): Whether to return per-cluster details.
        adjust_precision (bool, optional): If True, calculates precision more conservatively by
            only considering pairs as false positives when both genes are known to belong to
            different pairs in the benchmark data. This avoids penalizing potential novel
            interactions between genes not fully characterized in the benchmark dataset.

    Returns:
        dict or tuple: Dictionary with metrics or tuple of (metrics, cluster_details)
    """
    # Filter non-targeting genes
    if control_key is not None:
        cluster_df = phate_leiden_clustering[
            ~phate_leiden_clustering[perturbation_col_name].str.startswith(control_key)
        ]
    else:
        cluster_df = phate_leiden_clustering

    # Create positive pairs and track which genes belong to each complex
    positive_pairs = set()
    gene_to_complex = {}  # For adjusted precision

    for pair_id, pair in pair_benchmark.groupby("pair"):
        genes = pair["gene_name"].unique()

        # Track which complex each gene belongs to
        if adjust_precision:
            for gene in genes:
                gene_to_complex[gene] = pair_id

        # Create all gene pairs within this complex
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene_pair = tuple(sorted([genes[i], genes[j]]))
                positive_pairs.add(gene_pair)

    # Get a set of all genes that belong to any complex
    genes_in_complexes = set(gene_to_complex.keys()) if adjust_precision else set()

    # Get clusters sorted by size (descending) and limit to max_clusters
    cluster_sizes = cluster_df.groupby("cluster").size().sort_values(ascending=False)
    if max_clusters is not None:
        cluster_ids = cluster_sizes.index[:max_clusters]
    else:
        cluster_ids = cluster_sizes.index

    # Filter to only include selected clusters
    cluster_df = cluster_df[cluster_df["cluster"].isin(cluster_ids)]

    # Create gene to cluster mapping
    gene_to_cluster = dict(
        zip(
            cluster_df[perturbation_col_name],
            cluster_df["cluster"],
        )
    )

    # Initialize counters for global metrics
    true_positives = 0
    false_positives = 0
    adjusted_false_positives = 0  # For adjusted precision
    false_negatives = 0

    # Initialize per-cluster metrics if requested
    cluster_metrics = {} if return_cluster_details else None

    # Count all gene pairs within the same cluster
    for cluster_id, cluster_genes in cluster_df.groupby("cluster")[
        perturbation_col_name
    ]:
        cluster_tp = 0
        cluster_fp = 0
        cluster_adjusted_fp = 0

        genes = list(cluster_genes)
        # Generate all pairs within this cluster
        for i in range(len(genes)):
            for j in range(i + 1, len(genes)):
                gene_pair = tuple(sorted([genes[i], genes[j]]))

                # Check if this is a true positive
                if gene_pair in positive_pairs:
                    true_positives += 1
                    cluster_tp += 1
                else:
                    # Standard false positive count
                    false_positives += 1
                    cluster_fp += 1

                    # For adjusted precision calculation
                    if adjust_precision:
                        gene_a, gene_b = gene_pair

                        # Only count as false positive if both genes are in the benchmark set but in different complexes
                        if (
                            gene_a in genes_in_complexes
                            and gene_b in genes_in_complexes
                            and gene_to_complex.get(gene_a)
                            != gene_to_complex.get(gene_b)
                        ):
                            adjusted_false_positives += 1
                            cluster_adjusted_fp += 1

        if return_cluster_details:
            # Store cluster-specific metrics
            cluster_metrics[cluster_id] = {
                "true_positives": cluster_tp,
                "false_positives": cluster_fp,
                "adjusted_false_positives": cluster_adjusted_fp,
                "cluster_size": len(genes),
                "total_pairs": len(genes) * (len(genes) - 1) // 2,
                "precision": cluster_tp / (cluster_tp + cluster_fp)
                if (cluster_tp + cluster_fp) > 0
                else 0,
                "adjusted_precision": cluster_tp / (cluster_tp + cluster_adjusted_fp)
                if (cluster_tp + cluster_adjusted_fp) > 0
                else 0,
            }

    # Count false negatives (known pairs in different clusters)
    for gene_pair in positive_pairs:
        gene_a, gene_b = gene_pair
        if gene_a not in gene_to_cluster or gene_b not in gene_to_cluster:
            continue
        if gene_to_cluster[gene_a] != gene_to_cluster[gene_b]:
            false_negatives += 1

            # Add to cluster metrics for the appropriate clusters
            if return_cluster_details:
                cluster_a = gene_to_cluster[gene_a]
                cluster_b = gene_to_cluster[gene_b]

                if cluster_a in cluster_metrics:
                    if "false_negatives" not in cluster_metrics[cluster_a]:
                        cluster_metrics[cluster_a]["false_negatives"] = 0
                    cluster_metrics[cluster_a]["false_negatives"] += 1

                if cluster_b in cluster_metrics and cluster_b != cluster_a:
                    if "false_negatives" not in cluster_metrics[cluster_b]:
                        cluster_metrics[cluster_b]["false_negatives"] = 0
                    cluster_metrics[cluster_b]["false_negatives"] += 1

    # Calculate precision and recall
    standard_precision = (
        true_positives / (true_positives + false_positives)
        if (true_positives + false_positives) > 0
        else 0
    )
    adjusted_precision = (
        true_positives / (true_positives + adjusted_false_positives)
        if (true_positives + adjusted_false_positives) > 0
        else 0
    )
    recall = (
        true_positives / (true_positives + false_negatives)
        if (true_positives + false_negatives) > 0
        else 0
    )

    # Calculate F1 scores
    standard_f1 = (
        2 * (standard_precision * recall) / (standard_precision + recall)
        if (standard_precision + recall) > 0
        else 0
    )
    adjusted_f1 = (
        2 * (adjusted_precision * recall) / (adjusted_precision + recall)
        if (adjusted_precision + recall) > 0
        else 0
    )

    # Finalize per-cluster metrics
    if return_cluster_details:
        for cluster_id in cluster_metrics:
            # Some clusters might not have false negatives
            if "false_negatives" not in cluster_metrics[cluster_id]:
                cluster_metrics[cluster_id]["false_negatives"] = 0

            cluster_tp = cluster_metrics[cluster_id]["true_positives"]
            cluster_fn = cluster_metrics[cluster_id]["false_negatives"]

            # Calculate recall for this cluster
            cluster_metrics[cluster_id]["recall"] = (
                cluster_tp / (cluster_tp + cluster_fn)
                if (cluster_tp + cluster_fn) > 0
                else 0
            )

            # Calculate F1 scores
            p = cluster_metrics[cluster_id]["precision"]
            r = cluster_metrics[cluster_id]["recall"]
            cluster_metrics[cluster_id]["f1_score"] = (
                2 * (p * r) / (p + r) if (p + r) > 0 else 0
            )

            p_adj = cluster_metrics[cluster_id]["adjusted_precision"]
            cluster_metrics[cluster_id]["adjusted_f1_score"] = (
                2 * (p_adj * r) / (p_adj + r) if (p_adj + r) > 0 else 0
            )

    # Create metrics dictionary with both standard and adjusted metrics
    global_metrics = {
        "precision": adjusted_precision if adjust_precision else standard_precision,
        "standard_precision": standard_precision,
        "adjusted_precision": adjusted_precision,
        "recall": recall,
        "f1_score": adjusted_f1 if adjust_precision else standard_f1,
        "standard_f1_score": standard_f1,
        "adjusted_f1_score": adjusted_f1,
        "true_positives": true_positives,
        "false_positives": false_positives,
        "adjusted_false_positives": adjusted_false_positives,
        "false_negatives": false_negatives,
        "using_adjusted_precision": adjust_precision,
    }

    if return_cluster_details:
        # Convert to DataFrame
        cluster_df = pd.DataFrame.from_dict(cluster_metrics, orient="index")
        return global_metrics, cluster_df
    else:
        return global_metrics


def run_integrated_benchmarks(
    cluster_datasets,
    pair_benchmarks,
    group_benchmarks,
    perturbation_col_name="gene_symbol_0",
    control_key=None,
    max_clusters=None,
):
    """Run all benchmarks with early stopping and return integrated results.

    Args:
        cluster_datasets (dict): Mapping from dataset name to cluster DataFrame.
        pair_benchmarks (dict): Mapping from benchmark name to pair benchmark DataFrame.
        group_benchmarks (dict): Mapping from benchmark name to group benchmark DataFrame.
        perturbation_col_name (str): Column name for gene identifiers.
        control_key (str): Prefix for control perturbations to filter out.
        max_clusters (int): Maximum number of clusters to analyze.

    Returns:
        tuple: (integrated_results, cluster_details, global_metrics)
    """
    # Store results
    global_metrics = {}
    cluster_details = {}
    enrichment_tables = {}

    # Integrated results with all benchmarks combined
    integrated_results = {}

    # For each dataset
    for dataset_name, cluster_df in cluster_datasets.items():
        print(f"Processing dataset: {dataset_name}")

        # Initialize integrated results for this dataset
        integrated_results[dataset_name] = {
            "clusters": {},
            "enrichment_categories": {},
            "total_clusters_analyzed": 0,
        }

        # Get clusters sorted by size (descending) and limit to max_clusters
        cluster_sizes = (
            cluster_df.groupby("cluster").size().sort_values(ascending=False)
        )
        if max_clusters is not None:
            cluster_ids = list(cluster_sizes.index[:max_clusters])
        else:
            cluster_ids = list(cluster_sizes.index)

        integrated_results[dataset_name]["total_clusters_analyzed"] = len(cluster_ids)

        # Initialize cluster data
        for cluster_id in cluster_ids:
            integrated_results[dataset_name]["clusters"][cluster_id] = {
                "size": cluster_sizes[cluster_id],
                "enriched_in": set(),
                "details": {},
            }

        # Run pair benchmarks
        for benchmark_name, benchmark_df in pair_benchmarks.items():
            print(f"  - Running {benchmark_name} pair benchmark")

            # Get global metrics and per-cluster details
            pr_metrics, cluster_pr_df = calculate_pair_enrichment(
                benchmark_df,
                cluster_df,
                perturbation_col_name,
                control_key,
                max_clusters=max_clusters,
                return_cluster_details=True,
            )

            # Store global metrics
            dataset_key = f"{dataset_name}"
            if dataset_key not in global_metrics:
                global_metrics[dataset_key] = {}
            global_metrics[dataset_key][benchmark_name] = pr_metrics

            # Store cluster details
            for cluster_id in cluster_ids:
                if cluster_id in cluster_pr_df.index:
                    cluster_metrics = cluster_pr_df.loc[cluster_id].to_dict()

                    # Mark as enriched if precision is above a threshold (e.g., global average)
                    is_enriched = cluster_metrics["precision"] > pr_metrics["precision"]

                    if is_enriched:
                        integrated_results[dataset_name]["clusters"][cluster_id][
                            "enriched_in"
                        ].add(benchmark_name)

                    # Store detailed metrics
                    integrated_results[dataset_name]["clusters"][cluster_id]["details"][
                        benchmark_name
                    ] = {
                        "precision": cluster_metrics["precision"],
                        "recall": cluster_metrics["recall"]
                        if "recall" in cluster_metrics
                        else None,
                        "true_positives": cluster_metrics["true_positives"],
                        "false_positives": cluster_metrics["false_positives"],
                        "false_negatives": cluster_metrics["false_negatives"]
                        if "false_negatives" in cluster_metrics
                        else None,
                    }

        # Run group enrichment benchmarks
        for benchmark_name, benchmark_df in group_benchmarks.items():
            print(f"  - Running {benchmark_name} group benchmark")

            # Calculate enrichment with early stopping
            enrichment_table = calculate_group_enrichment(
                benchmark_df,
                cluster_df,
                perturbation_col_name,
                control_key,
                return_full_table=True,
                max_clusters=max_clusters,
            )

            # Store enrichment table
            dataset_key = f"{dataset_name}"
            enrichment_tables[f"{dataset_key}_{benchmark_name}"] = enrichment_table

            # Calculate global metrics
            num_enriched = sum(enrichment_table["num_enriched_groups"] > 0)
            proportion_enriched = sum(
                enrichment_table["num_enriched_groups"] > 0
            ) / len(enrichment_table)

            if dataset_key not in global_metrics:
                global_metrics[dataset_key] = {}
            global_metrics[dataset_key][benchmark_name] = {
                "num_enriched_clusters": num_enriched,
                "proportion_enriched": proportion_enriched,
            }

            # Update integrated results
            for _, row in enrichment_table.iterrows():
                cluster_id = row["cluster"]
                is_enriched = row["num_enriched_groups"] > 0

                if (
                    is_enriched
                    and cluster_id in integrated_results[dataset_name]["clusters"]
                ):
                    integrated_results[dataset_name]["clusters"][cluster_id][
                        "enriched_in"
                    ].add(benchmark_name)
                    integrated_results[dataset_name]["clusters"][cluster_id]["details"][
                        benchmark_name
                    ] = {
                        "num_enriched_groups": row["num_enriched_groups"],
                        "enriched_groups": row["enriched_groups"].split("; ")
                        if row["enriched_groups"]
                        else [],
                    }

        # Count clusters by enrichment category combinations
        for cluster_id, cluster_data in integrated_results[dataset_name][
            "clusters"
        ].items():
            categories = cluster_data["enriched_in"]
            category_key = tuple(sorted(categories)) if categories else ("None",)

            if (
                category_key
                not in integrated_results[dataset_name]["enrichment_categories"]
            ):
                integrated_results[dataset_name]["enrichment_categories"][
                    category_key
                ] = 0

            integrated_results[dataset_name]["enrichment_categories"][category_key] += 1

    return integrated_results, cluster_details, global_metrics


def generate_combined_enrichment_table(integrated_results, dataset_name, cluster_df):
    """Generate a comprehensive table of cluster enrichment across all benchmarks.

    Args:
        integrated_results (dict): Results from run_integrated_benchmarks.
        dataset_name (str): Name of dataset to generate table for.
        cluster_df (pd.DataFrame): Original clustering data with gene names.

    Returns:
        pd.DataFrame: Table with cluster enrichment details.
    """
    if dataset_name not in integrated_results:
        raise ValueError(f"Dataset {dataset_name} not found in results")

    rows = []
    for cluster_id, cluster_data in integrated_results[dataset_name][
        "clusters"
    ].items():
        # Get genes for this cluster
        cluster_genes = cluster_df[cluster_df["cluster"] == cluster_id][
            "gene_symbol_0"
        ].tolist()

        row = {
            "cluster": cluster_id,
            "genes": "; ".join(sorted(cluster_genes)),
            "cluster_size": cluster_data["size"],
            "enriched_in": ", ".join(sorted(cluster_data["enriched_in"]))
            if cluster_data["enriched_in"]
            else "None",
        }

        # Add details for each benchmark
        for benchmark, details in cluster_data["details"].items():
            if "precision" in details:  # Pair benchmark
                row[f"{benchmark}_precision"] = details["precision"]
                row[f"{benchmark}_true_positives"] = details["true_positives"]
            elif "num_enriched_groups" in details:  # Group benchmark
                row[f"{benchmark}_enriched_groups"] = len(details["enriched_groups"])
                row[f"{benchmark}_enriched_ids"] = ", ".join(details["enriched_groups"])

        rows.append(row)

    return pd.DataFrame(rows).sort_values("cluster").reset_index(drop=True)


def enrichment_pie_chart(integrated_results, dataset_name):
    """Create a single pie chart showing cluster enrichment with and without unenriched clusters.

    Args:
        integrated_results (dict): Results from run_integrated_benchmarks.
        dataset_name (str): Name of dataset to plot.

    Returns:
        matplotlib.figure.Figure: Figure with a single pie chart.
    """
    if dataset_name not in integrated_results:
        raise ValueError(f"Dataset {dataset_name} not found in results")

    # Create a figure with a single plot
    fig, ax = plt.subplots(dpi=300)

    # Extract all clusters data
    clusters_data = integrated_results[dataset_name]["clusters"]

    # Count enrichment categories directly from cluster data
    category_counts = {
        "Not Enriched": 0,
        "CORUM": 0,
        "KEGG": 0,
        "STRING": 0,
        "Multiple": 0,
    }

    for cluster_id, cluster_info in clusters_data.items():
        enriched_in = cluster_info["enriched_in"]

        if not enriched_in:
            category_counts["Not Enriched"] += 1
        elif len(enriched_in) == 1:
            # Single category - keep as is
            category = next(iter(enriched_in))
            if category in category_counts:
                category_counts[category] += 1
            else:
                print(f"Warning: Unknown category {category}")
                # Fallback
                category_counts["Multiple"] += 1
        else:
            # Multiple categories
            category_counts["Multiple"] += 1

    # Define clear, distinguishable colors for the categories
    color_map = {
        "Not Enriched": "#D3D3D3",  # Light gray
        "CORUM": "#4C78A8",  # Blue
        "KEGG": "#F58518",  # Orange
        "STRING": "#72B7B2",  # Teal
        "Multiple": "#54A24B",  # Green
    }

    # Prepare data for pie chart
    labels = []
    sizes = []
    colors = []

    # Sort by size, keeping Not Enriched first
    sorted_categories = sorted(
        category_counts.items(),
        key=lambda x: (0 if x[0] == "Not Enriched" else 1, -x[1]),
    )

    for category, count in sorted_categories:
        if count > 0:  # Only include categories with non-zero count
            labels.append(category)
            sizes.append(count)
            colors.append(color_map[category])

    # Create a function that will map the percentage to the actual count
    def make_autopct(values):
        def autopct(pct):
            # Calculate the absolute value based on the percentage
            absolute = int(round(pct * sum(values) / 100.0))
            return f"{absolute}"

        return autopct

    # Draw the pie chart
    wedges, texts, autotexts = ax.pie(
        sizes,
        labels=None,  # We'll use a legend instead
        autopct=make_autopct(sizes),
        startangle=90,
        colors=colors,
    )

    # Customize text for better readability
    for i, autotext in enumerate(autotexts):
        # Choose text color based on background color brightness
        if colors[i] == "#D3D3D3":  # Light gray (Not Enriched)
            autotext.set_color("black")
        else:
            autotext.set_color("white")
        autotext.set_fontsize(11)

    # Equal aspect ratio
    ax.axis("equal")

    # Calculate total clusters for percentages
    total_clusters = sum(category_counts.values())

    # Create legend entries with counts and percentages
    legend_entries = [
        f"{label} ({count}, {count / total_clusters:.1%})"
        for label, count in sorted_categories
        if count > 0
    ]

    # Add legend
    ax.legend(
        wedges,
        legend_entries,
        title="Categories",
        loc="center right",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=11,
    )

    # Main title for the figure
    ax.set_title(
        f"Cluster Enrichment Categories for {dataset_name}", fontsize=16, pad=20
    )

    # Adjust layout
    plt.tight_layout()

    return fig


def enrichment_bar_chart(df: pd.DataFrame, figsize=(12, 8)) -> plt.Figure:
    """Plot enrichment bar plots for STRING, CORUM, and KEGG per cluster.

    Args:
        df: DataFrame with columns 'cluster', 'STRING_true_positives',
            'CORUM_enriched_groups', and 'KEGG_enriched_groups'.
        figsize: Size of the matplotlib figure.

    Returns:
        Matplotlib Figure object.
    """
    df = df.copy()
    for col in [
        "STRING_true_positives",
        "CORUM_enriched_groups",
        "KEGG_enriched_groups",
    ]:
        if col not in df.columns:
            df[col] = 0
        df[col] = df[col].fillna(0)

    fig, axes = plt.subplots(3, 1, figsize=figsize, sharex=True)
    fig.suptitle("Cluster Enrichment")

    axes[0].bar(df["cluster"], df["STRING_true_positives"])
    axes[0].set_ylabel("STRING True Positives")

    axes[1].bar(df["cluster"], df["CORUM_enriched_groups"])
    axes[1].set_ylabel("CORUM Enriched Groups")

    axes[2].bar(df["cluster"], df["KEGG_enriched_groups"])
    axes[2].set_ylabel("KEGG Enriched Groups")
    axes[2].set_xlabel("Cluster")

    clusters = df["cluster"].astype(int)
    xticks = clusters[clusters % 20 == 0].unique()
    axes[2].set_xticks(xticks)

    return fig


def save_json_results(data, output_path):
    """Save data as a JSON file, converting unsupported types to serializable forms.

    This function recursively converts sets to lists, NumPy numeric types to
    native Python types, and ensures dictionary keys are strings before saving
    the result as a JSON file to the specified path.

    Args:
        data: The data structure to save. May include nested dicts, lists, sets,
            and NumPy scalar types.
        output_path (str or Path): The file path where the JSON will be written.
    """

    def clean(obj):
        if isinstance(obj, set):
            return list(obj)
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, dict):
            return {str(k): clean(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [clean(i) for i in obj]
        return obj

    cleaned_data = clean(data)
    with open(output_path, "w") as f:
        json.dump(cleaned_data, f, indent=2)
