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
    metric="balanced",
    resolutions=None,
    ideal_size_range=(15, 25),
    size_metric="mean",
):
    """Find the optimal Leiden resolution by reading benchmark outputs.

    Scans benchmark result files across resolutions and ranks them by the
    specified metric. Includes cluster count and size statistics.

    Args:
        root_fp (Path): Root output directory (config["all"]["root_fp"]).
        channel_combo (str): Channel combination name.
        cell_class (str): Cell class name.
        use_filtered (bool): Whether to look in filtered/ subdirectory.
        metric (str): Metric to optimize. Options:
            - "corum_enrichment": Proportion of clusters enriched for CORUM complexes
            - "kegg_enrichment": Proportion of clusters enriched for KEGG pathways
            - "string_f1": STRING pair F1 score
            - "combined": Average of all normalized enrichment metrics
            - "balanced": Enrichment + cluster granularity + size (recommended)
        resolutions (list, optional): List of resolutions to check.
            If None, auto-discovers from directory structure.
        ideal_size_range (tuple): (min, max) ideal genes per cluster.
            Default (15, 25) targets mean cluster size of ~20 genes.
            Based on typical pathway/complex sizes.
            Scoring favors the midpoint with smooth decay.
            Cluster count range is auto-derived from total genes / ideal_size_range.
        size_metric (str): Which cluster size metric to optimize. Options:
            - "mean": Mean genes per cluster (default, better for skewed distributions)
            - "median": Median genes per cluster (less sensitive to outliers)

    Returns:
        dict: Dictionary with:
            - optimal_resolution: Best resolution for the metric
            - all_results: DataFrame with metrics for all resolutions
            - metric_used: Which metric was used for ranking
            - size_metric_used: Which size metric was used ("mean" or "median")
            - ideal_size_range: Target cluster size range used for scoring
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
    total_genes = None  # Will be set from first clustering file

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

        # Load clustering file to get cluster statistics
        clustering_file = res_path / "phate_leiden_clustering.tsv"
        num_clusters = 0
        median_cluster_size = 0
        mean_cluster_size = 0
        min_cluster_size = 0
        max_cluster_size = 0

        if clustering_file.exists():
            clustering_df = (
                pd.read_parquet(clustering_file)
                if clustering_file.suffix == ".parquet"
                else pd.read_csv(clustering_file, sep="\t")
            )

            # Get total genes from first clustering file
            if total_genes is None:
                total_genes = len(clustering_df)

            cluster_sizes = clustering_df["cluster"].value_counts().sort_index()
            num_clusters = len(cluster_sizes)
            median_cluster_size = int(cluster_sizes.median())
            mean_cluster_size = round(cluster_sizes.mean(), 1)
            min_cluster_size = int(cluster_sizes.min())
            max_cluster_size = int(cluster_sizes.max())

        result = {
            "resolution": res,
            "num_clusters": num_clusters,
            "median_cluster_size": median_cluster_size,
            "mean_cluster_size": mean_cluster_size,
            "min_cluster_size": min_cluster_size,
            "max_cluster_size": max_cluster_size,
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

    if total_genes is None:
        raise ValueError("Could not determine total gene count from clustering files")

    results_df = pd.DataFrame(results)

    # Calculate ideal cluster count range based on total genes and ideal cluster size
    # If we want clusters of size 20-50, and have 3000 genes:
    # - ideal_max_clusters = 3000 / 20 = 150 clusters
    # - ideal_min_clusters = 3000 / 50 = 60 clusters
    ideal_max_clusters = int(total_genes / ideal_size_range[0])
    ideal_min_clusters = int(total_genes / ideal_size_range[1])
    ideal_cluster_range = (ideal_min_clusters, ideal_max_clusters)

    # Calculate combined score (normalized average of enrichment metrics)
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

    def cluster_size_utility(size, ideal_min, ideal_max):
        """Score cluster size - strongly prefer smaller clusters in ideal range.

        Calculates balanced score combining enrichment and size utility.
        Primary constraint is cluster size (biologically meaningful).
        Cluster count is indirectly constrained through size preference.

        Heavily penalizes clusters larger than ideal range.
        More forgiving for clusters slightly below ideal range.

        Args:
            size: The cluster size to score.
            ideal_min: Minimum of ideal cluster size range.
            ideal_max: Maximum of ideal cluster size range.

        Returns:
            float: Score between 0 and 1.
        """
        target = (
            ideal_min + ideal_max
        ) / 2  # Target is midpoint (e.g., 25 for range 20-30)

        if size < ideal_min:
            # Below minimum: gentle penalty if close (e.g., 18 when min is 20)
            # Use quadratic to be more forgiving near the boundary
            ratio = size / ideal_min
            return max(0, ratio**0.7)  # 18/20 = 0.9 -> 0.93, 15/20 = 0.75 -> 0.81
        elif size > ideal_max:
            # Above maximum: STEEP penalty - we really don't want large clusters
            # Linear decay that drops quickly
            excess = size - ideal_max
            penalty_rate = 2.0  # Much steeper than before (was 0.3)
            return max(0, 1.0 - penalty_rate * excess / ideal_max)
            # e.g., size=38, max=30: 1.0 - 2.0 * 8/30 = 1.0 - 0.53 = 0.47
            # e.g., size=46, max=30: 1.0 - 2.0 * 16/30 = 1.0 - 1.07 = 0.0
        else:
            # Within range: Prefer smaller end (closer to ideal_min than ideal_max)
            # Peak score at target, but favor being below target over above target
            if size <= target:
                # Below or at target: full score to gentle decay
                normalized_distance = (target - size) / (target - ideal_min)
                return 1.0 - 0.1 * normalized_distance  # Small penalty
            else:
                # Above target: steeper decay towards ideal_max
                normalized_distance = (size - target) / (ideal_max - target)
                return 1.0 - 0.4 * normalized_distance  # Larger penalty

    # Choose which size metric to use for scoring
    size_column = (
        "mean_cluster_size" if size_metric == "mean" else "median_cluster_size"
    )
    results_df["cluster_size_utility"] = results_df[size_column].apply(
        lambda x: cluster_size_utility(x, ideal_size_range[0], ideal_size_range[1])
    )

    # Balanced score: 40% enrichment quality, 60% cluster size utility
    # STRONGLY favor proper cluster granularity over enrichment metrics
    results_df["balanced"] = (
        0.4 * results_df["combined"] + 0.6 * results_df["cluster_size_utility"]
    )

    # Find optimal based on metric
    if metric == "balanced":
        optimal_idx = results_df["balanced"].idxmax()
    elif metric == "combined":
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
        "size_metric_used": size_metric,
        "ideal_size_range": ideal_size_range,
    }


def plot_resolution_comparison(results_df, metric="balanced", figsize=(15, 10)):
    """Plot benchmark metrics across resolutions including cluster statistics.

    Args:
        results_df (pandas.DataFrame): DataFrame from find_optimal_resolution.
        metric (str): Primary metric to highlight. Default "balanced".
        figsize (tuple): Figure size.

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    axes = axes.flatten()

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

    # Number of clusters
    axes[3].plot(
        results_df["resolution"],
        results_df["num_clusters"],
        "o-",
        color="purple",
    )
    axes[3].set_xlabel("Leiden Resolution")
    axes[3].set_ylabel("Number of Clusters")
    axes[3].set_title("Cluster Granularity")
    axes[3].grid(True, alpha=0.3)

    # Median cluster size
    axes[4].plot(
        results_df["resolution"],
        results_df["median_cluster_size"],
        "o-",
        color="darkred",
    )
    axes[4].set_xlabel("Leiden Resolution")
    axes[4].set_ylabel("Median Genes per Cluster")
    axes[4].set_title("Cluster Size Distribution")
    axes[4].grid(True, alpha=0.3)

    # Balanced score (or selected metric)
    score_col = metric if metric in results_df.columns else "balanced"
    axes[5].plot(
        results_df["resolution"],
        results_df[score_col],
        "o-",
        color="black",
        linewidth=2,
    )
    # Highlight optimal
    optimal_idx = results_df[score_col].idxmax()
    axes[5].plot(
        results_df.loc[optimal_idx, "resolution"],
        results_df.loc[optimal_idx, score_col],
        "o",
        color="gold",
        markersize=12,
        markeredgecolor="black",
        markeredgewidth=2,
    )
    axes[5].set_xlabel("Leiden Resolution")
    axes[5].set_ylabel("Score")
    axes[5].set_title(f"Composite Score: {score_col.replace('_', ' ').title()}")
    axes[5].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def format_resolution_table(results_df, top_n=None):
    """Format results table for easy decision making.

    Args:
        results_df (pandas.DataFrame): DataFrame from find_optimal_resolution.
        top_n (int, optional): Only show top N resolutions by balanced score.

    Returns:
        pandas.DataFrame: Formatted table with key metrics.
    """
    # Select key columns
    display_cols = [
        "resolution",
        "num_clusters",
        "median_cluster_size",
        "mean_cluster_size",
        "corum_enrichment",
        "kegg_enrichment",
        "string_f1",
        "corum_num_enriched",
    ]

    # Add score columns if they exist
    if "balanced" in results_df.columns:
        display_cols.append("balanced")
    if "combined" in results_df.columns:
        display_cols.append("combined")

    table = results_df[display_cols].copy()

    # Round for readability
    table["corum_enrichment"] = (table["corum_enrichment"] * 100).round(1)
    table["kegg_enrichment"] = (table["kegg_enrichment"] * 100).round(1)
    table["string_f1"] = (table["string_f1"] * 100).round(1)
    if "balanced" in table.columns:
        table["balanced"] = (table["balanced"] * 100).round(1)
    if "combined" in table.columns:
        table["combined"] = (table["combined"] * 100).round(1)

    # Rename for clarity
    table = table.rename(
        columns={
            "resolution": "Resolution",
            "num_clusters": "# Clusters",
            "median_cluster_size": "Median Size",
            "mean_cluster_size": "Mean Size",
            "corum_enrichment": "CORUM %",
            "kegg_enrichment": "KEGG %",
            "string_f1": "STRING F1 %",
            "corum_num_enriched": "# CORUM",
            "balanced": "Balanced Score",
            "combined": "Enrichment Score",
        }
    )

    # Sort by balanced score if available
    if "Balanced Score" in table.columns:
        table = table.sort_values("Balanced Score", ascending=False)
    elif "Enrichment Score" in table.columns:
        table = table.sort_values("Enrichment Score", ascending=False)

    if top_n:
        table = table.head(top_n)

    return table.reset_index(drop=True)


def analyze_all_resolutions(
    root_fp,
    cell_classes,
    channel_combos,
    use_filtered=False,
    metric="balanced",
    ideal_size_range=(15, 25),
    size_metric="mean",
    show_plots=True,
    top_n_table=10,
    verbose=True,
):
    """Analyze optimal resolutions for all cell class/channel combinations.

    Convenience wrapper that finds optimal resolutions, displays results,
    and returns a summary for all combinations.

    Args:
        root_fp (Path): Root output directory (config["all"]["root_fp"]).
        cell_classes (list): List of cell class names (e.g., ["Interphase", "Mitotic"]).
        channel_combos (list): List of channel combinations.
        use_filtered (bool): Whether to look in filtered/ subdirectory.
        metric (str): Metric to optimize ("balanced", "combined", etc.).
        ideal_size_range (tuple): (min, max) target cluster size.
            Default (15, 25) targets mean size of ~20 genes.
        size_metric (str): "mean" (default) or "median" cluster size.
        show_plots (bool): Whether to display comparison plots (default True).
        top_n_table (int): Number of top resolutions to show in table (default 10).
        verbose (bool): Whether to print detailed output (default True).

    Returns:
        dict: Dictionary mapping "cellclass_channel" to result dict from find_optimal_resolution.
            Also includes "summary_df" key with pandas DataFrame of all optimal resolutions.
    """
    import matplotlib.pyplot as plt
    from IPython.display import display

    optimal_resolutions = {}

    for cell_class in cell_classes:
        for channel_combo in channel_combos:
            try:
                result = find_optimal_resolution(
                    root_fp=root_fp,
                    channel_combo=channel_combo,
                    cell_class=cell_class,
                    use_filtered=use_filtered,
                    metric=metric,
                    ideal_size_range=ideal_size_range,
                    size_metric=size_metric,
                )
                key = f"{cell_class}_{channel_combo}"
                optimal_resolutions[key] = result

                if verbose:
                    print(f"\n{'=' * 80}")
                    print(f"{cell_class} / {channel_combo}")
                    print(f"{'=' * 80}")
                    print(f"  Optimal resolution: {result['optimal_resolution']}")
                    print(f"  Optimization metric: {result['metric_used']}")
                    print(f"  Size metric used: {result['size_metric_used']}")
                    print(f"  Target size range: {result['ideal_size_range']}")

                    # Show formatted decision table
                    print(
                        f"\nTop {top_n_table} resolutions (sorted by {metric} score):"
                    )
                    table = format_resolution_table(
                        result["all_results"], top_n=top_n_table
                    )
                    display(table)

                if show_plots:
                    # Show metrics comparison plot
                    fig = plot_resolution_comparison(
                        result["all_results"], metric=metric
                    )
                    plt.suptitle(
                        f"{cell_class} / {channel_combo}",
                        fontsize=14,
                        fontweight="bold",
                    )
                    plt.tight_layout()
                    plt.show()

            except Exception as e:
                if verbose:
                    print(
                        f"\n{cell_class} / {channel_combo}: No benchmark results found"
                    )
                    print(f"  Error: {e}")

    # Generate summary table
    if verbose:
        print("\n" + "=" * 80)
        print("SUMMARY: Optimal Resolutions")
        print("=" * 80)

    summary_rows = []
    for key, result in optimal_resolutions.items():
        opt_res = result["optimal_resolution"]
        opt_row = result["all_results"][
            result["all_results"]["resolution"] == opt_res
        ].iloc[0]
        summary_rows.append(
            {
                "cell_class_channel": key,
                "resolution": opt_res,
                "num_clusters": int(opt_row["num_clusters"]),
                "median_size": int(opt_row["median_cluster_size"]),
                "mean_size": opt_row["mean_cluster_size"],
                "corum_enrich": f"{opt_row['corum_enrichment']:.1%}",
                "kegg_enrich": f"{opt_row['kegg_enrichment']:.1%}",
                "string_f1": f"{opt_row['string_f1']:.1%}",
                "balanced_score": f"{opt_row['balanced']:.3f}",
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    if verbose:
        display(summary_df)

    # Add summary to results dict
    optimal_resolutions["summary_df"] = summary_df

    return optimal_resolutions


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
