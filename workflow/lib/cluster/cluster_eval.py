"""Evaluation utilities for clustering quality assessment.

This module provides visualization and evaluation functions for assessing
the quality of gene clustering results, including cell distribution analysis,
cluster size visualization, and bootstrap filtering visualization.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from lib.shared.file_utils import parse_filename


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
            Expected columns: gene (or bootstrap_gene_col), *_pval, *_log10, *_fdr columns.
        genes_df (pandas.DataFrame): Gene features table (e.g., features_genes.tsv).
            Expected columns: perturbation_name_col + feature columns.
        perturbation_name_col (str): Column name for gene identifiers in genes_df.
        bootstrap_gene_col (str): Column name for gene identifiers in bootstrap_df.
            Defaults to "gene".

    Returns:
        pandas.DataFrame: Merged dataframe with genes_df columns + bootstrap stats.
    """
    # Make a copy to avoid modifying original
    bootstrap_copy = bootstrap_df.copy()

    # Rename bootstrap gene column to match genes_df if needed
    if bootstrap_gene_col in bootstrap_copy.columns and perturbation_name_col not in bootstrap_copy.columns:
        bootstrap_copy = bootstrap_copy.rename(columns={bootstrap_gene_col: perturbation_name_col})

    # Identify which features were bootstrapped by looking at _pval columns
    bootstrapped_features = [
        col.replace("_pval", "")
        for col in bootstrap_copy.columns
        if col.endswith("_pval")
    ]

    # Subset genes_df to only include bootstrapped features + metadata
    metadata_cols = [perturbation_name_col, "cell_count"]
    genes_cols_to_keep = [
        col for col in genes_df.columns
        if col in metadata_cols or col in bootstrapped_features
    ]
    genes_subset = genes_df[genes_cols_to_keep]

    # Merge on gene column
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

    Identifies genes with at least one feature passing the significance threshold,
    plus all nontargeting controls matching the specified patterns.

    Args:
        merged_data (pandas.DataFrame): DataFrame with bootstrap results merged with
            gene features. Must contain columns ending in '_log10' and '_fdr'.
        perturbation_name_col (str): Column name for gene/perturbation identifiers.
        control_patterns (list): List of regex patterns to identify control genes
            (e.g., ["^nontargeting_intergenic_", "^nontargeting_or_"]).
        zscore_threshold (float): Z-score threshold for raw feature values.
        zscore_direction (str): Filter direction - "positive", "negative", or "both".
        fdr_threshold (float): FDR cutoff for multiple testing correction.
        filter_mode (str): How to combine filters - "zscore", "fdr", or "both".

    Returns:
        tuple: (filtered_df, filter_stats) where:
            - filtered_df: DataFrame with only genes passing filters + controls
            - filter_stats: dict with filtering statistics
    """
    # Find fdr columns
    fdr_cols = [c for c in merged_data.columns if c.endswith("_fdr")]

    if not fdr_cols:
        raise ValueError("No _fdr columns found in merged_data")

    # Find raw feature columns (exclude metadata and bootstrap result columns)
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

    # Identify controls using patterns
    is_control = pd.Series(False, index=merged_data.index)
    for pattern in control_patterns:
        is_control |= merged_data[perturbation_name_col].str.match(pattern, na=False)

    # Apply zscore filtering on raw feature values (gene passes if ANY feature passes threshold)
    zscore_mask = pd.Series(False, index=merged_data.index)
    for col in feature_cols:
        if zscore_direction == "both":
            zscore_mask |= merged_data[col].abs() >= zscore_threshold
        elif zscore_direction == "positive":
            zscore_mask |= merged_data[col] >= zscore_threshold
        else:
            zscore_mask |= merged_data[col] <= -zscore_threshold

    # Apply FDR filtering (gene passes if ANY feature passes threshold)
    fdr_mask = pd.Series(False, index=merged_data.index)
    for col in fdr_cols:
        fdr_mask |= merged_data[col] < fdr_threshold

    # Apply combined filtering (gene passes if ANY feature passes BOTH thresholds)
    combined_mask = pd.Series(False, index=merged_data.index)
    for col in feature_cols:
        fdr_col = f"{col}_fdr"
        if fdr_col in merged_data.columns:
            # Check if this feature passes zscore threshold
            if zscore_direction == "both":
                feat_zscore_pass = merged_data[col].abs() >= zscore_threshold
            elif zscore_direction == "positive":
                feat_zscore_pass = merged_data[col] >= zscore_threshold
            else:
                feat_zscore_pass = merged_data[col] <= -zscore_threshold
            # Check if this feature passes FDR threshold
            feat_fdr_pass = merged_data[fdr_col] < fdr_threshold
            # Gene passes if this feature passes BOTH
            combined_mask |= feat_zscore_pass & feat_fdr_pass

    # Combine filters based on mode
    if filter_mode == "zscore":
        passes_filter = zscore_mask
    elif filter_mode == "fdr":
        passes_filter = fdr_mask
    else:  # "both"
        passes_filter = combined_mask

    # Final mask includes passing genes + controls
    final_mask = passes_filter | is_control

    # Build statistics
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

    Convenience function that returns just the gene names (for subsetting other tables).

    Args:
        merged_data (pandas.DataFrame): DataFrame with bootstrap results merged with
            gene features. Must contain columns ending in '_log10' and '_fdr'.
        perturbation_name_col (str): Column name for gene/perturbation identifiers.
        control_patterns (list): List of regex patterns to identify control genes.
        zscore_threshold (float): Z-score threshold for raw feature values.
        zscore_direction (str): Filter direction - "positive", "negative", or "both".
        fdr_threshold (float): FDR cutoff for multiple testing correction.
        filter_mode (str): How to combine filters - "zscore", "fdr", or "both".

    Returns:
        list: List of gene names that pass the filter (including controls).
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


def plot_bootstrap_filter_scatter(
    merged_data,
    perturbation_name_col,
    control_patterns,
    zscore_threshold,
    zscore_direction,
    fdr_threshold,
    filter_mode,
    figsize=(10, 8),
):
    """Create a scatter plot showing which genes pass bootstrap filtering thresholds.

    Visualizes bootstrap statistics with threshold lines, highlighting genes that
    pass filters, fail filters, and control genes that are always kept.

    Args:
        merged_data (pandas.DataFrame): DataFrame with bootstrap results merged with
            aggregated data. Must contain columns ending in '_log10' and '_fdr'.
        perturbation_name_col (str): Column name for gene/perturbation identifiers.
        control_patterns (list): List of regex patterns to identify control genes.
        zscore_threshold (float): -log10(pval) threshold for significance.
        zscore_direction (str): Filter direction - "positive", "negative", or "both".
        fdr_threshold (float): FDR cutoff for multiple testing correction.
        filter_mode (str): How to combine filters - "zscore", "fdr", or "both".
        figsize (tuple, optional): Figure size. Defaults to (10, 8).

    Returns:
        tuple: (fig, filter_summary) where fig is the matplotlib figure and
            filter_summary is a dict with filtering statistics.
    """
    # Find log10 and fdr columns
    log10_cols = [c for c in merged_data.columns if c.endswith("_log10")]
    fdr_cols = [c for c in merged_data.columns if c.endswith("_fdr")]

    if not log10_cols or not fdr_cols:
        raise ValueError("No _log10 or _fdr columns found in merged_data")

    plot_log10_col = log10_cols[0]
    plot_fdr_col = fdr_cols[0]

    # Identify controls using patterns
    is_control = pd.Series(False, index=merged_data.index)
    for pattern in control_patterns:
        is_control |= merged_data[perturbation_name_col].str.match(pattern, na=False)

    # Apply filtering logic
    zscore_mask = pd.Series(False, index=merged_data.index)
    for col in log10_cols:
        if zscore_direction == "both":
            zscore_mask |= merged_data[col].abs() >= zscore_threshold
        elif zscore_direction == "positive":
            zscore_mask |= merged_data[col] >= zscore_threshold
        else:
            zscore_mask |= merged_data[col] <= -zscore_threshold

    fdr_mask = pd.Series(False, index=merged_data.index)
    for col in fdr_cols:
        fdr_mask |= merged_data[col] < fdr_threshold

    if filter_mode == "zscore":
        passes_filter = zscore_mask
    elif filter_mode == "fdr":
        passes_filter = fdr_mask
    else:
        passes_filter = zscore_mask & fdr_mask

    # Create scatter plot
    fig, ax = plt.subplots(figsize=figsize)

    # Plot genes that don't pass filter (gray)
    fails = ~passes_filter & ~is_control
    ax.scatter(
        merged_data.loc[fails, plot_log10_col],
        -np.log10(merged_data.loc[fails, plot_fdr_col] + 1e-10),
        c="lightgray",
        alpha=0.5,
        s=10,
        label=f"Filtered out ({fails.sum()})",
    )

    # Plot genes that pass filter (blue)
    passes = passes_filter & ~is_control
    ax.scatter(
        merged_data.loc[passes, plot_log10_col],
        -np.log10(merged_data.loc[passes, plot_fdr_col] + 1e-10),
        c="steelblue",
        alpha=0.7,
        s=15,
        label=f"Pass filter ({passes.sum()})",
    )

    # Plot controls (red)
    ax.scatter(
        merged_data.loc[is_control, plot_log10_col],
        -np.log10(merged_data.loc[is_control, plot_fdr_col] + 1e-10),
        c="red",
        alpha=0.8,
        s=20,
        marker="^",
        label=f"Controls ({is_control.sum()})",
    )

    # Add threshold lines
    ax.axhline(
        -np.log10(fdr_threshold),
        color="orange",
        linestyle="--",
        label=f"FDR threshold ({fdr_threshold})",
    )
    if zscore_direction in ["both", "positive"]:
        ax.axvline(
            zscore_threshold,
            color="green",
            linestyle="--",
            label=f"Z-score threshold ({zscore_threshold})",
        )
    if zscore_direction in ["both", "negative"]:
        ax.axvline(-zscore_threshold, color="green", linestyle="--")

    ax.set_xlabel(f"{plot_log10_col} (-log10 p-value)")
    ax.set_ylabel(f"-log10({plot_fdr_col})")
    ax.set_title(
        f"Bootstrap Filtering: {passes.sum()} genes + {is_control.sum()} controls pass\n"
        f"(Filter mode: {filter_mode})"
    )
    ax.legend(loc="upper right")
    plt.tight_layout()

    # Build summary dict
    filter_summary = {
        "total_genes": len(merged_data),
        "pass_zscore": zscore_mask.sum(),
        "pass_fdr": fdr_mask.sum(),
        "pass_combined": passes_filter.sum(),
        "controls": is_control.sum(),
        "final_filtered": (passes_filter | is_control).sum(),
    }

    return fig, filter_summary
