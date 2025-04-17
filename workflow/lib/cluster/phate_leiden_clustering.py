"""Implementation of PHATE dimensionality reduction with Leiden clustering.

This module provides functions to perform dimensionality reduction using PHATE
(Potential of Heat-diffusion for Affinity-based Trajectory Embedding) followed by
Leiden community detection for clustering. It also includes visualization utilities
for the resulting low-dimensional embeddings.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from igraph import Graph
import leidenalg
import phate


def phate_leiden_pipeline(
    aggregated_data,
    resolution,
    phate_distance_metric,
    first_feature_name="PC_0",
):
    """Run complete PHATE dimensionality reduction and Leiden clustering pipeline.

    Args:
        aggregated_data (pd.DataFrame): Input data with metadata and feature columns.
        resolution (float): Resolution parameter for Leiden clustering.
        phate_distance_metric (str): Distance metric for PHATE algorithm (e.g., 'euclidean', 'cosine').
        first_feature_name (str, optional): Name of first feature column. Defaults to "PC_0".

    Returns:
        pd.DataFrame: DataFrame with original metadata, PHATE coordinates and cluster assignments.
    """
    # Identify feature columns - first_feature_name and everything after it
    all_cols = aggregated_data.columns.tolist()
    feature_start_idx = all_cols.index(first_feature_name)
    feature_cols = all_cols[feature_start_idx:]
    feature_selected_data = aggregated_data[feature_cols]

    # Get metadata columns (everything before first_feature_name)
    metadata_cols = all_cols[:feature_start_idx]

    # Run PHATE
    df_phate, p = run_phate(feature_selected_data, metric=phate_distance_metric)

    # Get weights from PHATE
    weights = np.asarray(p.graph.diff_op.todense())

    # Run Leiden clustering
    clusters = run_leiden_clustering(weights, resolution=resolution)

    # Add clusters to results
    df_phate["cluster"] = clusters

    # Combine metadata with PHATE results
    result_df = pd.concat([aggregated_data[metadata_cols], df_phate], axis=1)

    # sort by cluster
    result_df = result_df.sort_values(by=["cluster"])

    return result_df.reset_index(drop=True)


def run_phate(
    feature_selected_data,
    random_state=42,
    knn=10,
    metric="euclidean",
    **kwargs,
):
    """Run PHATE dimensionality reduction.

    Performs dimensionality reduction using the PHATE algorithm to generate
    a low-dimensional representation of the input data.

    Args:
        feature_selected_data (pd.DataFrame): Input data matrix with features as columns.
        random_state (int, optional): Random seed for reproducibility. Defaults to 42.
        knn (int, optional): Number of nearest neighbors to use. Defaults to 10.
        metric (str, optional): Distance metric for KNN calculations. Defaults to 'euclidean'.
        **kwargs: Additional parameters to pass to the PHATE constructor.

    Returns:
        tuple:
            pd.DataFrame: DataFrame with PHATE coordinates.
            phate.PHATE: Fitted PHATE object with graph and other attributes.
    """
    # Initialize and run PHATE
    p = phate.PHATE(
        random_state=random_state,
        n_jobs=-1,
        knn=knn,
        knn_dist=metric,
        verbose=False,
    )

    # Transform data
    X_phate = p.fit_transform(feature_selected_data.values)

    # Create output DataFrame
    df_phate = pd.DataFrame(
        X_phate, index=feature_selected_data.index, columns=["PHATE_0", "PHATE_1"]
    )

    return df_phate, p


def run_leiden_clustering(weights, resolution=1.0, seed=42):
    """Run Leiden community detection algorithm on a weighted adjacency matrix.

    Performs clustering using the Leiden algorithm, which is an improved version
    of the Louvain method for community detection in networks.

    Args:
        weights (np.ndarray): Weighted adjacency matrix representing the graph.
        resolution (float, optional): Resolution parameter controlling cluster granularity.
            Higher values yield more clusters. Defaults to 1.0.
        seed (int, optional): Random seed for reproducibility. Defaults to 42.

    Returns:
        list: Cluster assignments for each node in the graph.
    """
    # Force symmetry by averaging with transpose
    weights_symmetric = (weights + weights.T) / 2

    # Create graph from symmetrized weights
    g = Graph().Weighted_Adjacency(matrix=weights_symmetric.tolist(), mode="undirected")

    # Run Leiden clustering
    partition = leidenalg.find_partition(
        g,
        partition_type=leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        n_iterations=-1,
        seed=seed,
        resolution_parameter=resolution,
    )

    return partition.membership


def plot_phate_leiden_clusters(
    phate_leiden_clustering, perturbation_name_col, control_key, figsize=(8, 8)
):
    """Create a scatter plot visualization of PHATE embedding colored by Leiden clusters.

    Generates a visualization showing the 2D PHATE embedding with points colored by
    cluster assignment, with control samples highlighted in gray.

    Args:
        phate_leiden_clustering (pd.DataFrame): Output from phate_leiden_pipeline with
            'PHATE_0', 'PHATE_1', and 'cluster' columns.
        perturbation_name_col (str): Column name containing perturbation identifiers.
        control_key (str): Prefix or value in perturbation_name_col that identifies controls.
        figsize (tuple, optional): Figure dimensions (width, height). Defaults to (8, 8).

    Returns:
        matplotlib.figure.Figure: The figure object for further customization or saving.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Split data into experimental and control groups
    control_mask = phate_leiden_clustering[perturbation_name_col].str.startswith(
        control_key
    )
    control_data = phate_leiden_clustering[control_mask]
    exp_data = phate_leiden_clustering[~control_mask]

    # Plot experimental data colored by cluster
    sns.scatterplot(
        data=exp_data,
        x="PHATE_0",
        y="PHATE_1",
        hue="cluster",
        palette="husl",
        alpha=0.7,
        legend=False,
        ax=ax,
    )

    # Plot control data in gray
    sns.scatterplot(
        data=control_data,
        x="PHATE_0",
        y="PHATE_1",
        color="gray",
        alpha=0.5,
        label="control",
        ax=ax,
    )

    # Format plot
    plt.legend(loc="upper right")
    plt.tight_layout()

    return fig
