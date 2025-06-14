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
    return_potential=False,
):
    """Run complete PHATE dimensionality reduction and Leiden clustering pipeline.

    Args:
        aggregated_data (pd.DataFrame): Input data with metadata and feature columns.
        resolution (float): Resolution parameter for Leiden clustering.
        phate_distance_metric (str): Distance metric for PHATE algorithm (e.g., 'euclidean', 'cosine').
        first_feature_name (str, optional): Name of first feature column. Defaults to "PC_0".
        return_potential (bool, optional): Whether to return the reshaped potential array. Defaults to False.

    Returns:
        pd.DataFrame or tuple: DataFrame with original metadata, PHATE coordinates and cluster assignments.
                              If return_potential=True, also returns the reshaped potential array.
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

    # Create a DataFrame from the potential matrix
    potential = p.diff_potential
    potential_df = pd.DataFrame(
        potential, columns=[f"potential_{i}" for i in range(potential.shape[1])]
    )

    # Get weights from PHATE
    weights = np.asarray(p.graph.diff_op.todense())

    # Run Leiden clustering
    clusters = run_leiden_clustering(weights, resolution=resolution)

    # Add clusters to results
    df_phate["cluster"] = clusters

    # Combine metadata with PHATE results
    result_df = pd.concat([aggregated_data[metadata_cols], df_phate], axis=1)

    # Add the first column of df_phate to potential_df in the first position
    gene_col = result_df.columns[0]
    potential_df.insert(0, gene_col, result_df[gene_col])

    # sort by cluster
    result_df = result_df.sort_values(by=["cluster"])

    if return_potential:
        return result_df, potential_df
    else:
        return result_df


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
    phate_leiden_clustering,
    perturbation_name_col,
    control_key,
    figsize=(8, 8),
    clusters_of_interest=None,
    highlight_palette="Set2",
    non_highlight_color="gray",
    control_color_highlight="#4575b4",
):
    """Create a scatter plot visualization of PHATE embedding colored by Leiden clusters.

    Generates a visualization showing the 2D PHATE embedding with points colored by
    cluster assignment, with control samples highlighted. Can optionally highlight
    specific clusters of interest while dimming others.

    Args:
        phate_leiden_clustering (pd.DataFrame): Output from phate_leiden_pipeline with
            'PHATE_0', 'PHATE_1', and 'cluster' columns.
        perturbation_name_col (str): Column name containing perturbation identifiers.
        control_key (str): Prefix or value in perturbation_name_col that identifies controls.
        figsize (tuple, optional): Figure dimensions (width, height). Defaults to (8, 8).
        clusters_of_interest (list or int, optional): Cluster ID(s) to highlight. If None,
            all clusters are colored. Defaults to None.
        highlight_palette (str or list, optional): Color palette for highlighted clusters.
            Defaults to "Set2".
        non_highlight_color (str, optional): Color for non-highlighted clusters.
            Defaults to "gray".
        control_color_highlight (str, optional): Color for controls when highlighting clusters.
            Defaults to "#4575b4".

    Returns:
        matplotlib.figure.Figure: The figure object for further customization or saving.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Ensure clusters_of_interest is a list
    if clusters_of_interest is not None:
        if not isinstance(clusters_of_interest, list):
            clusters_of_interest = [clusters_of_interest]

    # Split data into experimental and control groups
    control_mask = phate_leiden_clustering[perturbation_name_col].str.startswith(
        control_key
    )
    control_data = phate_leiden_clustering[control_mask]
    exp_data = phate_leiden_clustering[~control_mask]

    if clusters_of_interest is None:
        # Original behavior - plot all experimental data colored by cluster
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

        # Plot control data in gray (original behavior)
        sns.scatterplot(
            data=control_data,
            x="PHATE_0",
            y="PHATE_1",
            color="gray",
            alpha=0.5,
            label="control",
            ax=ax,
        )
    else:
        # New behavior - highlight specific clusters
        # Plot non-highlighted clusters first in gray
        non_highlight_mask = ~exp_data["cluster"].isin(clusters_of_interest)
        non_highlight_data = exp_data[non_highlight_mask]

        if len(non_highlight_data) > 0:
            sns.scatterplot(
                data=non_highlight_data,
                x="PHATE_0",
                y="PHATE_1",
                color=non_highlight_color,
                alpha=0.3,
                label="Other clusters",
                ax=ax,
                legend=False,
            )

        # Plot highlighted clusters with colors
        highlight_data = exp_data[exp_data["cluster"].isin(clusters_of_interest)]

        if len(highlight_data) > 0:
            # Use sns.scatterplot with the hue parameter for consistency
            sns.scatterplot(
                data=highlight_data,
                x="PHATE_0",
                y="PHATE_1",
                hue="cluster",
                palette=highlight_palette,
                alpha=0.7,
                ax=ax,
                legend="brief",
            )

            # Rename legend entries to include "Cluster" prefix
            handles, labels = ax.get_legend_handles_labels()
            new_labels = [
                f"Cluster {label}"
                if label.isdigit() or (isinstance(label, (int, float)))
                else label
                for label in labels
            ]
            ax.legend(handles, new_labels, loc="upper right")

        # Plot control data in blue when highlighting
        if len(control_data) > 0:
            sns.scatterplot(
                data=control_data,
                x="PHATE_0",
                y="PHATE_1",
                color=control_color_highlight,
                alpha=0.5,
                label="Control",
                ax=ax,
            )

            # Update legend to include controls if not already present
            if len(highlight_data) > 0:
                handles, labels = ax.get_legend_handles_labels()
                # Remove duplicate "Control" entries if any
                seen = set()
                new_handles, new_labels = [], []
                for h, l in zip(handles, labels):
                    if l not in seen:
                        seen.add(l)
                        new_handles.append(h)
                        new_labels.append(l)
                ax.legend(new_handles, new_labels, loc="upper right")

    # Format plot
    if clusters_of_interest is None or len(highlight_data) > 0 or len(control_data) > 0:
        plt.legend(loc="upper right")

    return fig


def calculate_potential_to_nontargeting(
    potential_df, control_key, distance_metric="euclidean", normalize=True
):
    """Calculate the average distance from each row to nontargeting controls.

    Args:
        potential_df (pd.DataFrame): DataFrame with gene_symbol_0 and potential columns
        control_key (str): String pattern used to identify control rows
        distance_metric (str): Distance metric to use (default: 'euclidean')
        normalize (bool): Whether to min-max normalize the distances (default: True)

    Returns:
        pd.DataFrame: DataFrame with gene_symbol_0, mean_potential_to_nontargeting,
                      and normalized_potential_to_nontargeting (if normalize=True)
    """
    import numpy as np
    from scipy.spatial.distance import pdist, squareform

    # Extract potential columns (all columns except gene_symbol_0)
    potential_cols = [
        col for col in potential_df.columns if col.startswith("potential_")
    ]

    # Identify nontargeting control rows
    nontargeting_mask = potential_df["gene_symbol_0"].str.contains(
        control_key, na=False
    )
    nontargeting_indices = potential_df.index[nontargeting_mask].tolist()

    # Extract only the potential values for calculation
    potential_values = potential_df[potential_cols].values

    # Calculate pairwise distances between all rows
    distances = squareform(pdist(potential_values, metric=distance_metric))

    # Convert to DataFrame for easier indexing
    distance_df = pd.DataFrame(
        distances, index=potential_df.index, columns=potential_df.index
    )

    # For each row, calculate average distance to nontargeting controls
    average_distance = []
    for idx in potential_df.index:
        gene_symbol = potential_df.loc[idx, "gene_symbol_0"]

        # Get distances from this row to all nontargeting controls
        distances_to_nontargeting = [
            distance_df.loc[idx, control_idx] for control_idx in nontargeting_indices
        ]

        # Calculate average distance
        avg_distance = np.mean(distances_to_nontargeting)

        average_distance.append(
            {
                "gene_symbol_0": gene_symbol,
                "mean_potential_to_nontargeting": avg_distance,
            }
        )

    # Create result DataFrame
    average_distance_df = pd.DataFrame(average_distance)

    # Apply min-max normalization if requested
    if normalize:
        min_val = average_distance_df["mean_potential_to_nontargeting"].min()
        max_val = average_distance_df["mean_potential_to_nontargeting"].max()

        # Add the normalized values as a new column
        average_distance_df["normalized_potential_to_nontargeting"] = (
            average_distance_df["mean_potential_to_nontargeting"] - min_val
        ) / (max_val - min_val)

    return average_distance_df
