import random

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from igraph import Graph
import leidenalg
import phate


def select_features(
    df, correlation_threshold=0.9, variance_threshold=0.01, min_unique_values=5
):
    """
    Select features based on correlation, variance, and unique values.

    Parameters:
    -----------
    df : pd.DataFrame
        Input DataFrame with features to be selected
    correlation_threshold : float, default=0.9
        Threshold for removing highly correlated features
    variance_threshold : float, default=0.01
        Threshold for removing low variance features
    min_unique_values : int, default=5
        Minimum unique values required for a feature to be kept

    Returns:
    --------
    tuple
        (DataFrame with selected features, dictionary of removed features)

    """
    import numpy as np
    import pandas as pd

    # Make a copy and handle initial column filtering
    df = df.copy()
    if "cell_number" in df.columns:
        df = df.drop(columns=["cell_number"])

    # Store information about removed features
    removed_features = {"correlated": [], "low_variance": [], "few_unique_values": []}

    # Get numeric columns only, excluding gene_symbol_0
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    feature_cols = [col for col in numeric_cols if col != "gene_symbol_0"]
    df_numeric = df[feature_cols]

    # Calculate correlation matrix once
    correlation_matrix = df_numeric.corr().abs()

    # Create a mask to get upper triangle of correlation matrix
    upper_tri = np.triu(np.ones(correlation_matrix.shape), k=1)
    high_corr_pairs = []

    # Get all highly correlated pairs at once
    pairs_idx = np.where(
        (correlation_matrix.values * upper_tri) > correlation_threshold
    )
    for i, j in zip(*pairs_idx):
        high_corr_pairs.append(
            (
                correlation_matrix.index[i],
                correlation_matrix.columns[j],
                correlation_matrix.iloc[i, j],
            )
        )

    # Process all correlated features at once
    if high_corr_pairs:
        # Calculate mean correlation for each feature
        mean_correlations = correlation_matrix.mean()

        # Track features to remove
        features_to_remove = set()

        # For each correlated pair, remove the feature with higher mean correlation
        for col1, col2, corr_value in high_corr_pairs:
            if col1 not in features_to_remove and col2 not in features_to_remove:
                feature_to_remove = (
                    col1 if mean_correlations[col1] > mean_correlations[col2] else col2
                )
                features_to_remove.add(feature_to_remove)

                removed_features["correlated"].append(
                    {
                        "feature": feature_to_remove,
                        "correlated_with": col2 if feature_to_remove == col1 else col1,
                        "correlation": corr_value,
                    }
                )

        df_numeric = df_numeric.drop(columns=list(features_to_remove))

    # Step 2: Remove low variance features (unchanged but done in one step)
    variances = df_numeric.var()
    low_variance_features = variances[variances < variance_threshold].index
    removed_features["low_variance"] = [
        {"feature": feat, "variance": variances[feat]} for feat in low_variance_features
    ]
    df_numeric = df_numeric.drop(columns=low_variance_features)

    # Step 3: Remove features with few unique values (unchanged but done in one step)
    unique_counts = df_numeric.nunique()
    few_unique_features = unique_counts[unique_counts < min_unique_values].index
    removed_features["few_unique_values"] = [
        {"feature": feat, "unique_values": unique_counts[feat]}
        for feat in few_unique_features
    ]
    df_numeric = df_numeric.drop(columns=few_unique_features)

    # Print summary
    print("\nFeature Selection Summary:")
    print(f"Original features: {len(numeric_cols)}")
    print(f"Features removed due to correlation: {len(removed_features['correlated'])}")
    print(
        f"Features removed due to low variance: {len(removed_features['low_variance'])}"
    )
    print(
        f"Features removed due to few unique values: {len(removed_features['few_unique_values'])}"
    )
    print(f"Final features: {len(df_numeric.columns)}")

    # Create final DataFrame with remaining numeric columns AND gene_symbol_0
    final_columns = ["gene_symbol_0"] + df_numeric.columns.tolist()

    return df[final_columns], removed_features


def normalize_to_controls(df, control_prefix="sg_nt"):
    """
    Normalize data using StandardScaler fit to control samples.
    Sets gene_symbol_0 as index if it isn't already.

    Args:
        df (pd.DataFrame): DataFrame to normalize
        control_prefix (str): Prefix identifying control samples in index or gene_symbol_0 column

    Returns:
        pd.DataFrame: Normalized DataFrame with gene symbols as index
    """
    df_copy = df.copy()

    # Handle cases where gene_symbol_0 might be a column or already the index
    if "gene_symbol_0" in df_copy.columns:
        df_copy = df_copy.set_index("gene_symbol_0")

    # Fit scaler on control samples
    scaler = StandardScaler()
    control_mask = df_copy.index.str.startswith(control_prefix)
    scaler.fit(df_copy[control_mask].values)

    # Transform all data
    df_norm = pd.DataFrame(
        scaler.transform(df_copy.values), index=df_copy.index, columns=df_copy.columns
    )

    return df_norm


def perform_pca_analysis(df, variance_threshold=0.95, random_state=42):
    """
    Perform PCA analysis and create explained variance plot.
    Expects gene_symbol_0 to be the index.

    Args:
        df (pd.DataFrame): Data with gene symbols as index
        variance_threshold (float): Cumulative variance threshold (default 0.95)
        random_state (int): Random seed for reproducibility

    Returns:
        tuple: (pca_df, n_components, pca_object, fig)
            - pca_df: DataFrame with PCA transformed data (gene symbols as index)
            - n_components: Number of components needed to reach variance threshold
            - pca_object: Fitted PCA object
            - fig: Figure object for explained variance plot
    """
    # Initialize and fit PCA
    pca = PCA(random_state=random_state)
    pca_transformed = pca.fit_transform(df)

    # Create DataFrame with PCA results
    n_components_total = pca_transformed.shape[1]
    pca_df = pd.DataFrame(
        pca_transformed,
        columns=[f"pca_{n}" for n in range(n_components_total)],
        index=df.index,
    )

    # Find number of components needed for threshold
    cumsum = pca.explained_variance_ratio_.cumsum()
    n_components = np.argwhere(cumsum >= variance_threshold)[0][0] + 1

    # Create variance plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(cumsum, "-")
    ax.axhline(
        variance_threshold,
        linestyle="--",
        color="red",
        label=f"{variance_threshold*100}% Threshold",
    )
    ax.axvline(n_components, linestyle="--", color="blue", label=f"n={n_components}")
    ax.set_ylabel("Cumulative fraction of variance explained")
    ax.set_xlabel("Number of principal components included")
    ax.set_title("PCA Explained Variance Ratio")
    ax.grid(True)
    ax.legend()

    print(
        f"Number of components needed for {variance_threshold*100}% variance: {n_components}"
    )
    print(f"Shape of input data: {df.shape}")

    # Create threshold-limited version
    pca_df_threshold = pca_df[[f"pca_{i}" for i in range(n_components)]]

    print(f"Shape of PCA transformed and reduced data: {pca_df_threshold.shape}")

    return pca_df_threshold, n_components, pca, fig


def phate_leiden_pipeline(df, resolution=1.0, phate_kwargs=None):
    """
    Run complete PHATE and Leiden clustering pipeline.

    Parameters:
    -----------
    df : pandas.DataFrame
        Input data matrix
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering
    phate_kwargs : dict, optional
        Additional arguments for PHATE

    Returns:
    --------
    pandas.DataFrame
        DataFrame with PHATE coordinates and cluster assignments
    """
    # Default PHATE parameters
    if phate_kwargs is None:
        phate_kwargs = {}

    # Run PHATE
    df_phate, p = run_phate(df, **phate_kwargs)

    # Get weights from PHATE
    weights = np.asarray(p.graph.diff_op.todense())

    # Run Leiden clustering
    clusters = run_leiden_clustering(weights, resolution=resolution)

    # Add clusters to results
    df_phate["cluster"] = clusters

    # Sort by cluster
    df_phate = df_phate.sort_values("cluster")

    # Print number of clusters and average cluster size
    print(f"Number of clusters: {df_phate['cluster'].nunique()}")
    print(f"Average cluster size: {df_phate['cluster'].value_counts().mean():.2f}")

    return df_phate


def run_phate(df, random_state=42, n_jobs=4, knn=10, metric="euclidean", **kwargs):
    """
    Run PHATE dimensionality reduction.

    Parameters:
    -----------
    df : pandas.DataFrame
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
    X_phate = p.fit_transform(df.values)

    # Create output DataFrame
    df_phate = pd.DataFrame(X_phate, index=df.index, columns=["PHATE_0", "PHATE_1"])

    return df_phate, p


def run_leiden_clustering(weights, resolution=1.0, seed=42):
    """
    Run Leiden clustering on a weighted adjacency matrix.

    Parameters:
    -----------
    weights : numpy.ndarray
        Weighted adjacency matrix
    resolution : float, default=1.0
        Resolution parameter for Leiden clustering
    seed : int, default=42
        Random seed for reproducibility

    Returns:
    --------
    list
        Cluster assignments
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


def dimensionality_reduction(
    df,
    x="X",
    y="Y",
    default_kwargs={"color": "lightgray", "alpha": 0.5},
    control_query='gene_id=="-1"',
    control_color="black",
    control_legend=True,
    control_kwargs=dict(),
    label_query=None,
    label_hue="cluster",
    label_as_cmap=False,
    label_palette="rainbow",
    label_kwargs=dict(),
    randomize_palette=False,
    label_legend=False,
    legend_kwargs=dict(),
    hide_axes=False,
    ax=None,
    rasterized=True,
    save_plot_path=None,
    **kwargs,
):
    """
    Create a scatter plot for dimensionality reduction results.

    Parameters:
        df (pd.DataFrame): DataFrame with the data to plot.
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data.
        default_kwargs (dict): Default arguments for the scatter plot.
        control_query (str): Query to subset control data.
        control_color (str): Color for control points.
        control_legend (bool or str): If True, include legend for control data.
        control_kwargs (dict): Additional arguments for control points.
        label_query (str, optional): Query to subset data for labels.
        label_hue (str): Column name for label color grouping.
        label_as_cmap (bool): If True, use a color map for labels.
        label_palette (str or list): Seaborn palette for label colors.
        label_kwargs (dict): Additional arguments for labeled points.
        randomize_palette (bool or int): Randomize color palette if True.
        label_legend (bool): If True, include legend for labels.
        legend_kwargs (dict): Additional arguments for the legend.
        hide_axes (bool): If True, hide the axes.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        rasterized (bool): If True, use rasterized rendering.
        save_plot_path (str, optional): Path to save the plot as an image.

    Returns:
        matplotlib.axes.Axes: The Axes object with the plot.
    """
    df_ = df.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if control_query is not None:
        df_control = df_.query(control_query)
        df_ = df_[~(df_.index.isin(df_control.index))]

    if label_query is not None:
        df_label = df_.query(label_query)
        df_ = df_[~(df_.index.isin(df_label.index))]

    sns.scatterplot(
        data=df_, x=x, y=y, **default_kwargs, **kwargs, ax=ax, rasterized=rasterized
    )

    if control_query is not None:
        if "legend" not in control_kwargs:
            if isinstance(control_legend, str):
                control_kwargs["label"] = control_legend
            elif control_legend:
                control_kwargs["label"] = "control"
            else:
                control_kwargs["legend"] = False

        kwargs_ = kwargs.copy()
        kwargs_.update(control_kwargs)

        sns.scatterplot(
            data=df_control,
            x=x,
            y=y,
            color=control_color,
            alpha=0.75,
            **kwargs_,
            ax=ax,
            rasterized=rasterized,
        )

    if label_query is not None:
        n_colors = 1
        if label_hue is not None:
            n_colors = df_label[label_hue].nunique()

            palette = sns.color_palette(
                label_palette, n_colors=n_colors, as_cmap=label_as_cmap
            )

            if randomize_palette:
                random.seed(int(randomize_palette))
                random.shuffle(palette)
        else:
            palette = None

        kwargs_ = kwargs.copy()
        kwargs_.update(label_kwargs)

        sns.scatterplot(
            data=df_label,
            x=x,
            y=y,
            hue=label_hue,
            palette=palette,
            legend=label_legend,
            **kwargs_,
            ax=ax,
            rasterized=rasterized,
        )

        if label_legend:
            loc = legend_kwargs.pop("loc", (1.05, 0.33))
            if label_as_cmap:
                hue_norm = kwargs.get(
                    "hue_norm",
                    (
                        df_label[label_hue].astype(float).min(),
                        df_label[label_hue].astype(float).max(),
                    ),
                )
                s = kwargs.get("s", 10)
                hdl, _ = ax.get_legend_handles_labels()
                legend_colors = sns.color_palette(label_palette, as_cmap=True)(
                    np.linspace(0, 255, 5, dtype=int)
                )
                legend_color_vals = np.linspace(*hue_norm, 5)
                legend_color_header = hdl[0]
                legend_elements = [
                    plt.scatter(
                        [],
                        [],
                        marker="o",
                        s=s,
                        color=c,
                        linewidth=0.5,
                        edgecolor="k",
                        label=str(cl),
                    )
                    for c, cl in zip(legend_colors, legend_color_vals)
                ]
                ax.legend(handles=legend_elements, loc=loc, ncol=1, **legend_kwargs)
            else:
                n_cols = max(1, (n_colors // 20))
                ax.legend(loc=loc, ncol=n_cols, **legend_kwargs)

    if hide_axes:
        ax.axis("off")

    if save_plot_path:
        ax.figure.savefig(save_plot_path, dpi=300, bbox_inches="tight")

    return ax


def merge_phate_uniprot(df_phate, uniprot_data_fp):
    """
    Merge PHATE clustering results with UniProt data

    Parameters:
    -----------
    df_phate : pandas.DataFrame
        DataFrame with PHATE coordinates and cluster assignments
    uniprot_data_fp : str
        Path to UniProt data file

    Returns:
    --------
    pandas.DataFrame
        Merged DataFrame with UniProt data

    """
    # Make a copy to avoid modifying the original
    df_phate = df_phate.copy()

    # If gene_symbol_0 is in the index, reset it to become a column
    if df_phate.index.name == "gene_symbol_0":
        df_phate = df_phate.reset_index()
    # If we still don't have gene_symbol_0 as a column, create it from the index
    elif "gene_symbol_0" not in df_phate.columns:
        df_phate["gene_symbol_0"] = df_phate.index
        df_phate = df_phate.reset_index(drop=True)

    # Load UniProt data
    uniprot_df = pd.read_csv(uniprot_data_fp, sep="\t")

    # Split gene names and explode
    uniprot_df["gene_names"] = uniprot_df["Gene Names"].str.split()
    uniprot_df = uniprot_df.explode("gene_names")
    uniprot_df.rename(columns={"Function [CC]": "Function"}, inplace=True)

    # Merge with PHATE data
    result = pd.merge(
        df_phate,
        uniprot_df.rename(columns={"gene_names": "gene_symbol_0"}),
        on="gene_symbol_0",
        how="left",
    )

    # Remove duplicate columns
    for col in result.columns:
        if result[col].dtype == "object":
            result[col] = result[col].str.replace(";", "")

    return result
