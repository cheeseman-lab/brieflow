"""Evaluation utilities for clustering quality assessment.

This module provides visualization and evaluation functions for assessing
the quality of gene clustering results, including cell distribution analysis,
cluster size visualization, bootstrap filtering, and optimal resolution finding.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from lib.shared.file_utils import parse_filename, get_filename


def find_optimal_resolution(
    root_fp,
    channel_combo,
    cell_class,
    use_filtered=False,
    metric="corum_enrichment",
    resolutions=None,
):
    """Find the optimal Leiden resolution by reading benchmark outputs.

    Scans benchmark result files across resolutions and ranks them by the
    specified metric.

    Args:
        root_fp (Path): Root output directory (config["all"]["root_fp"]).
        channel_combo (str): Channel combination name.
        cell_class (str): Cell class name.
        use_filtered (bool): Whether to look in filtered/ subdirectory.
        metric (str): Metric to optimize. Options:
            - "corum_enrichment": Proportion of clusters enriched for CORUM complexes
            - "kegg_enrichment": Proportion of clusters enriched for KEGG pathways
            - "string_f1": STRING pair F1 score
            - "combined": Average of all normalized metrics
        resolutions (list, optional): List of resolutions to check.
            If None, auto-discovers from directory structure.

    Returns:
        dict: Dictionary with:
            - optimal_resolution: Best resolution for the metric
            - all_results: DataFrame with metrics for all resolutions
            - metric_used: Which metric was used for ranking
    """
    root_fp = Path(root_fp)

    # Build base cluster path
    if use_filtered:
        base_path = root_fp / "cluster" / channel_combo / cell_class / "filtered"
    else:
        base_path = root_fp / "cluster" / channel_combo / cell_class

    # Auto-discover resolutions if not provided
    if resolutions is None:
        resolutions = []
        if base_path.exists():
            for d in base_path.iterdir():
                if d.is_dir() and d.name.isdigit():
                    resolutions.append(int(d.name))
        resolutions = sorted(resolutions)

    if not resolutions:
        raise ValueError(f"No resolution directories found in {base_path}")

    # Collect metrics for each resolution
    results = []
    for res in resolutions:
        res_path = base_path / str(res)

        # Look for Real global_metrics.json
        metrics_file = res_path / get_filename(
            {"cluster_benchmark": "Real"}, "global_metrics", "json"
        )

        if not metrics_file.exists():
            continue

        with open(metrics_file) as f:
            metrics = json.load(f)

        result = {
            "resolution": res,
            "corum_enrichment": metrics.get("CORUM", {}).get("proportion_enriched", 0),
            "kegg_enrichment": metrics.get("KEGG", {}).get("proportion_enriched", 0),
            "string_f1": metrics.get("STRING", {}).get("f1_score", 0),
            "string_precision": metrics.get("STRING", {}).get("precision", 0),
            "string_recall": metrics.get("STRING", {}).get("recall", 0),
            "corum_num_enriched": metrics.get("CORUM", {}).get(
                "num_enriched_clusters", 0
            ),
            "kegg_num_enriched": metrics.get("KEGG", {}).get(
                "num_enriched_clusters", 0
            ),
        }
        results.append(result)

    if not results:
        raise ValueError(f"No benchmark results found in {base_path}")

    results_df = pd.DataFrame(results)

    # Calculate combined score (normalized average)
    for col in ["corum_enrichment", "kegg_enrichment", "string_f1"]:
        max_val = results_df[col].max()
        if max_val > 0:
            results_df[f"{col}_norm"] = results_df[col] / max_val
        else:
            results_df[f"{col}_norm"] = 0

    results_df["combined"] = (
        results_df["corum_enrichment_norm"]
        + results_df["kegg_enrichment_norm"]
        + results_df["string_f1_norm"]
    ) / 3

    # Find optimal based on metric
    if metric == "combined":
        optimal_idx = results_df["combined"].idxmax()
    elif metric == "corum_enrichment":
        optimal_idx = results_df["corum_enrichment"].idxmax()
    elif metric == "kegg_enrichment":
        optimal_idx = results_df["kegg_enrichment"].idxmax()
    elif metric == "string_f1":
        optimal_idx = results_df["string_f1"].idxmax()
    else:
        raise ValueError(f"Unknown metric: {metric}")

    optimal_resolution = int(results_df.loc[optimal_idx, "resolution"])

    return {
        "optimal_resolution": optimal_resolution,
        "all_results": results_df.sort_values("resolution"),
        "metric_used": metric,
    }


def plot_resolution_comparison(results_df, metric="corum_enrichment", figsize=(12, 5)):
    """Plot benchmark metrics across resolutions.

    Args:
        results_df (pandas.DataFrame): DataFrame from find_optimal_resolution.
        metric (str): Primary metric to highlight.
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    # CORUM enrichment
    axes[0].plot(
        results_df["resolution"],
        results_df["corum_enrichment"],
        "o-",
        color="steelblue",
    )
    axes[0].set_xlabel("Leiden Resolution")
    axes[0].set_ylabel("Proportion Enriched")
    axes[0].set_title("CORUM Complex Enrichment")
    axes[0].grid(True, alpha=0.3)

    # KEGG enrichment
    axes[1].plot(
        results_df["resolution"],
        results_df["kegg_enrichment"],
        "o-",
        color="forestgreen",
    )
    axes[1].set_xlabel("Leiden Resolution")
    axes[1].set_ylabel("Proportion Enriched")
    axes[1].set_title("KEGG Pathway Enrichment")
    axes[1].grid(True, alpha=0.3)

    # STRING F1
    axes[2].plot(results_df["resolution"], results_df["string_f1"], "o-", color="coral")
    axes[2].set_xlabel("Leiden Resolution")
    axes[2].set_ylabel("F1 Score")
    axes[2].set_title("STRING Pair F1 Score")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def merge_bootstrap_with_genes(
    bootstrap_df,
    genes_df,
    perturbation_name_col,
    bootstrap_gene_col="gene",
):
    """Merge bootstrap results with gene-level feature table.

    Only includes feature columns from genes_df that were actually bootstrapped
    (determined by matching _pval columns in bootstrap_df).

    Args:
        bootstrap_df (pandas.DataFrame): Bootstrap results with significance stats.
        genes_df (pandas.DataFrame): Gene features table (e.g., features_genes.tsv).
        perturbation_name_col (str): Column name for gene identifiers in genes_df.
        bootstrap_gene_col (str): Column name for gene identifiers in bootstrap_df.

    Returns:
        pandas.DataFrame: Merged dataframe with genes_df columns + bootstrap stats.
    """
    bootstrap_copy = bootstrap_df.copy()

    if (
        bootstrap_gene_col in bootstrap_copy.columns
        and perturbation_name_col not in bootstrap_copy.columns
    ):
        bootstrap_copy = bootstrap_copy.rename(
            columns={bootstrap_gene_col: perturbation_name_col}
        )

    bootstrapped_features = [
        col.replace("_pval", "")
        for col in bootstrap_copy.columns
        if col.endswith("_pval")
    ]

    metadata_cols = [perturbation_name_col, "cell_count"]
    genes_cols_to_keep = [
        col
        for col in genes_df.columns
        if col in metadata_cols or col in bootstrapped_features
    ]
    genes_subset = genes_df[genes_cols_to_keep]

    merged = genes_subset.merge(
        bootstrap_copy,
        on=perturbation_name_col,
        how="left",
        suffixes=("", "_bootstrap"),
    )

    return merged


def filter_genes_by_bootstrap(
    merged_data,
    perturbation_name_col,
    control_patterns,
    zscore_threshold,
    zscore_direction,
    fdr_threshold,
    filter_mode,
):
    """Filter genes based on bootstrap significance thresholds.

    Args:
        merged_data (pandas.DataFrame): DataFrame with bootstrap results merged.
        perturbation_name_col (str): Column name for gene identifiers.
        control_patterns (list): List of regex patterns for control genes.
        zscore_threshold (float): Z-score threshold for raw feature values.
        zscore_direction (str): "positive", "negative", or "both".
        fdr_threshold (float): FDR cutoff.
        filter_mode (str): "zscore", "fdr", or "both".

    Returns:
        tuple: (filtered_df, filter_stats)
    """
    fdr_cols = [c for c in merged_data.columns if c.endswith("_fdr")]

    if not fdr_cols:
        raise ValueError("No _fdr columns found in merged_data")

    metadata_cols = [
        perturbation_name_col,
        "cell_count",
        "num_constructs",
        "total_cells",
        "num_sims",
    ]
    bootstrap_suffixes = ("_pval", "_log10", "_fdr")
    feature_cols = [
        c
        for c in merged_data.columns
        if c not in metadata_cols and not c.endswith(bootstrap_suffixes)
    ]

    is_control = pd.Series(False, index=merged_data.index)
    for pattern in control_patterns:
        is_control |= merged_data[perturbation_name_col].str.match(pattern, na=False)

    zscore_mask = pd.Series(False, index=merged_data.index)
    for col in feature_cols:
        if zscore_direction == "both":
            zscore_mask |= merged_data[col].abs() >= zscore_threshold
        elif zscore_direction == "positive":
            zscore_mask |= merged_data[col] >= zscore_threshold
        else:
            zscore_mask |= merged_data[col] <= -zscore_threshold

    fdr_mask = pd.Series(False, index=merged_data.index)
    for col in fdr_cols:
        fdr_mask |= merged_data[col] < fdr_threshold

    combined_mask = pd.Series(False, index=merged_data.index)
    for col in feature_cols:
        fdr_col = f"{col}_fdr"
        if fdr_col in merged_data.columns:
            if zscore_direction == "both":
                feat_zscore_pass = merged_data[col].abs() >= zscore_threshold
            elif zscore_direction == "positive":
                feat_zscore_pass = merged_data[col] >= zscore_threshold
            else:
                feat_zscore_pass = merged_data[col] <= -zscore_threshold
            feat_fdr_pass = merged_data[fdr_col] < fdr_threshold
            combined_mask |= feat_zscore_pass & feat_fdr_pass

    if filter_mode == "zscore":
        passes_filter = zscore_mask
    elif filter_mode == "fdr":
        passes_filter = fdr_mask
    else:
        passes_filter = combined_mask

    final_mask = passes_filter | is_control

    filter_stats = {
        "total_genes": len(merged_data),
        "pass_zscore": int(zscore_mask.sum()),
        "pass_fdr": int(fdr_mask.sum()),
        "pass_combined": int(passes_filter.sum()),
        "num_controls": int(is_control.sum()),
        "final_filtered": int(final_mask.sum()),
        "num_features_tested": len(feature_cols),
    }

    return merged_data[final_mask].copy(), filter_stats


def get_filtered_gene_list(
    merged_data,
    perturbation_name_col,
    control_patterns,
    zscore_threshold,
    zscore_direction,
    fdr_threshold,
    filter_mode,
):
    """Get list of gene names that pass bootstrap filtering.

    Args:
        merged_data: DataFrame with bootstrap results merged.
        perturbation_name_col: Column name for gene identifiers.
        control_patterns: List of regex patterns for control genes.
        zscore_threshold: Z-score threshold.
        zscore_direction: "positive", "negative", or "both".
        fdr_threshold: FDR cutoff.
        filter_mode: "zscore", "fdr", or "both".

    Returns:
        list: Gene names that pass the filter.
    """
    filtered_df, _ = filter_genes_by_bootstrap(
        merged_data,
        perturbation_name_col,
        control_patterns,
        zscore_threshold,
        zscore_direction,
        fdr_threshold,
        filter_mode,
    )
    return filtered_df[perturbation_name_col].tolist()


def plot_cell_histogram(
    gene_cell_counts,
    cutoff,
    perturbation_name_col,
    count_col_name="cell_count",
    bins=50,
    figsize=(12, 6),
):
    """Plot a histogram of cell numbers with a vertical cutoff line and return genes below the cutoff.

    Args:
        gene_cell_counts (pandas.DataFrame): DataFrame containing cell count data per gene.
        cutoff (float): Vertical line position and threshold for identifying genes.
        perturbation_name_col (str): Column name for gene/perturbation identifiers.
        count_col_name (str, optional): Column name containing cell counts. Defaults to "cell_count".
        bins (int, optional): Number of bins for histogram. Defaults to 50.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (12, 6).

    Returns:
        matplotlib.figure.Figure: The figure object of the generated plot.
    """
    # Create the figure
    fig, ax = plt.subplots(figsize=figsize)

    # Plot histogram using seaborn for better styling
    sns.histplot(
        data=gene_cell_counts,
        x=count_col_name,
        bins=bins,
        color="skyblue",
        alpha=0.6,
        ax=ax,
    )

    # Add vertical line at cutoff
    ax.axvline(x=cutoff, color="red", linestyle="--", label=f"Cutoff: {cutoff}")

    # Customize the plot
    ax.set_title("Distribution of Cell Numbers", fontsize=12, pad=15)
    ax.set_xlabel("Cell Number", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.legend()

    # Add grid for better readability
    ax.grid(True, alpha=0.3)

    # Get genes below cutoff
    genes_below_cutoff = gene_cell_counts[gene_cell_counts[count_col_name] <= cutoff][
        perturbation_name_col
    ].tolist()

    # Print genes below cutoff
    print(f"Number of genes below cutoff: {len(genes_below_cutoff)}")
    print(genes_below_cutoff)

    # Return the figure object
    return fig


def plot_cluster_sizes(phate_leiden_clustering):
    """Creates a histogram of cluster sizes from clustering data.

    Visualizes the distribution of cluster sizes to evaluate clustering granularity
    and identify potential imbalances in cluster assignments.

    Args:
        phate_leiden_clustering (pandas.DataFrame): DataFrame containing a 'cluster' column
            with cluster IDs assigned to each entity.

    Returns:
        matplotlib.figure.Figure: Figure object that can be saved or displayed.
    """
    fig = plt.figure(figsize=(10, 6))

    # Create histogram with bin count equal to max cluster number
    max_cluster = phate_leiden_clustering["cluster"].max()
    sns.histplot(
        data=phate_leiden_clustering, x="cluster", bins=max_cluster, discrete=True
    )

    # Labels
    plt.title("Cluster Sizes")
    plt.xlabel("Cluster Number")
    plt.ylabel("Cluster Size")

    return fig
