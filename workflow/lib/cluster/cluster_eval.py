"""Evaluation utilities for clustering quality assessment.

This module provides visualization and evaluation functions for assessing
the quality of gene clustering results, including cell distribution analysis
and cluster size visualization.
"""

from pathlib import Path

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns

from lib.shared.file_utils import parse_filename


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
