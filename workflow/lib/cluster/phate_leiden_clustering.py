"""Module for PHATE and Leiden clustering pipeline.

This module contains functions used to perform dimensionality reduction, clustering,
and data integration for the PHATE and Leiden clustering pipeline within the cluster
processing workflow. The main tasks include feature selection, normalization, PCA
analysis, PHATE dimensionality reduction, Leiden clustering, visualization, and
merging of clustering results with external data.

Functions:
    - select_features: Select features based on correlation, variance, and unique values.
    - normalize_to_controls: Normalize data using StandardScaler fit to control samples.
    - perform_pca_analysis: Perform PCA analysis and generate an explained variance plot.
    - phate_leiden_pipeline: Execute the full PHATE and Leiden clustering pipeline.
    - run_phate: Apply PHATE dimensionality reduction to the input data.
    - run_leiden_clustering: Run Leiden clustering on a weighted adjacency matrix.
    - dimensionality_reduction: Create a scatter plot for dimensionality reduction results.
    - merge_phate_uniprot: Merge PHATE clustering results with UniProt data.
"""

import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from igraph import Graph
import leidenalg
import phate


def phate_leiden_pipeline(
    aggregated_data, resolution=1.0, first_feature_name="PC_0", phate_kwargs=None
):
    """Run complete PHATE and Leiden clustering pipeline.

    Args:
        aggregated_data (pd.DataFrame): Input data with metadata and feature columns.
        resolution (float): Resolution parameter for Leiden clustering. Defaults to 1.0.
        first_feature_name (str): Name of first feature column. Defaults to "PC_0".
        phate_kwargs (dict, optional): Additional arguments for PHATE.

    Returns:
        pd.DataFrame: DataFrame with original metadata, PHATE coordinates and cluster assignments.
    """
    if phate_kwargs is None:
        phate_kwargs = {}

    # Identify feature columns - first_feature_name and everything after it
    all_cols = aggregated_data.columns.tolist()
    feature_start_idx = all_cols.index(first_feature_name)
    feature_cols = all_cols[feature_start_idx:]
    feature_selected_data = aggregated_data[feature_cols]

    # Get metadata columns (everything before first_feature_name)
    metadata_cols = all_cols[:feature_start_idx]

    # Run PHATE
    df_phate, p = run_phate(feature_selected_data, **phate_kwargs)

    # Get weights from PHATE
    weights = np.asarray(p.graph.diff_op.todense())

    # Run Leiden clustering
    clusters = run_leiden_clustering(weights, resolution=resolution)

    # Add clusters to results
    df_phate["cluster"] = clusters

    # Combine metadata with PHATE results
    result_df = pd.concat([aggregated_data[metadata_cols], df_phate], axis=1)

    print(f"Number of clusters: {result_df['cluster'].nunique()}")
    print(f"Average cluster size: {result_df['cluster'].value_counts().mean():.2f}")

    return result_df


def run_phate(
    feature_selected_data,
    random_state=42,
    n_jobs=4,
    knn=10,
    metric="euclidean",
    **kwargs,
):
    """Run PHATE dimensionality reduction.

    Parameters:
    -----------
    feature_selected_data : pandas.DataFrame
        Input data matrix
    random_state : int, default=42
        Random seed for reproducibility
    n_jobs : int, default=4
        Number of parallel jobs
    knn : int, default=10
        Number of nearest neighbors
    metric : str, default='euclidean'
        Distance metric for KNN
    **kwargs : dict
        Additional arguments passed to PHATE

    Returns:
    --------
    tuple
        (DataFrame with PHATE coordinates, PHATE object)
    """
    # Initialize and run PHATE
    p = phate.PHATE(
        random_state=random_state, n_jobs=n_jobs, knn=knn, knn_dist=metric, **kwargs
    )

    # Transform data
    X_phate = p.fit_transform(feature_selected_data.values)

    # Create output DataFrame
    df_phate = pd.DataFrame(
        X_phate, index=feature_selected_data.index, columns=["PHATE_0", "PHATE_1"]
    )

    return df_phate, p


def run_leiden_clustering(weights, resolution=1.0, seed=42):
    """Run Leiden clustering on a weighted adjacency matrix.

    Args:
        weights (np.ndarray): Weighted adjacency matrix.
        resolution (float): Resolution parameter for Leiden clustering. Defaults to 1.0.
        seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        list: Cluster assignments.
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
    phate_leiden_clustering, perturbation_name_col, control_key, figsize=(15, 15)
):
    """Create a simple scatter plot for PHATE leiden clustering results.

    Args:
        phate_leiden_clustering (pd.DataFrame): Output from phate_leiden_pipeline.
        perturbation_name_col (str): Column name with perturbation identifiers.
        control_key (str): Value in perturbation_name_col that indicates controls.
        figsize (tuple): Figure size. Defaults to (15, 15).

    Returns:
        matplotlib.figure.Figure: The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Split data into control and experimental
    control_data = phate_leiden_clustering[
        phate_leiden_clustering[perturbation_name_col] == control_key
    ]
    exp_data = phate_leiden_clustering[
        phate_leiden_clustering[perturbation_name_col] != control_key
    ]

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
