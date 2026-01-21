"""Cluster analysis and visualization utilities for omics data.

This module provides functions for analyzing and visualizing feature differences
between clusters of perturbations (e.g., genes) in experimental data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.stats import mannwhitneyu, ttest_ind
from adjustText import adjust_text


def median_analysis(
    feature_df,
    cluster_df,
    cluster_id,
    perturbation_name_col="gene_symbol_0",
    control_type="other_clusters",
    control_label=None,
    top_n=10,
    min_diff_threshold=0.1,
    min_ratio_threshold=1.1,
):
    """Compare median feature values between genes in a specific cluster versus control genes.

    Analyzes differences in feature medians between a target cluster and control groups
    (either other clusters or non-targeting controls). Uses threshold-based significance
    determination for robust identification of meaningful differences.

    Args:
        feature_df (pandas.DataFrame): DataFrame containing feature measurements with perturbation_name_col column.
        cluster_df (pandas.DataFrame): DataFrame containing cluster assignments with perturbation_name_col column.
        cluster_id (int or str): ID of the cluster to analyze.
        perturbation_name_col (str, optional): Column name containing perturbation identifiers. Defaults to "gene_symbol_0".
        control_type (str, optional): Either "other_clusters" or "nontargeting". Defaults to "other_clusters".
        control_label (str, optional): Label in perturbation_name_col that identifies non-targeting controls.
            Required if control_type is "nontargeting".
        top_n (int, optional): Number of top higher/lower features to return. Defaults to 10.
        min_diff_threshold (float, optional): Minimum absolute difference threshold for significance. Defaults to 0.1.
        min_ratio_threshold (float, optional): Minimum ratio threshold for significance (must be > 1). Defaults to 1.1.

    Returns:
        dict: Dictionary containing:
            - 'results_df': Complete results DataFrame with all features
            - 'higher_in_cluster': DataFrame with top features higher in cluster
            - 'lower_in_cluster': DataFrame with top features lower in cluster
            - 'significant': DataFrame with all significant features
            - 'test_genes': List of genes in test cluster
            - 'control_genes': List of genes in control group
    """
    # Create output structure
    results = {}

    # Get genes in the test cluster
    test_genes = cluster_df[cluster_df["cluster"] == cluster_id][
        perturbation_name_col
    ].tolist()
    results["test_genes"] = test_genes

    # Determine control genes based on control_type
    if control_type == "other_clusters":
        control_genes = cluster_df[cluster_df["cluster"] != cluster_id][
            perturbation_name_col
        ].tolist()
    elif control_type == "nontargeting":
        if control_label is None:
            raise ValueError(
                "control_label must be provided when control_type is 'nontargeting'"
            )
        control_genes = feature_df[
            feature_df[perturbation_name_col].str.contains(control_label)
        ][perturbation_name_col].tolist()
    else:
        raise ValueError(
            "control_type must be either 'other_clusters' or 'nontargeting'"
        )

    results["control_genes"] = control_genes

    # Filter feature_df to only include numeric features
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != "cluster"]

    # Initialize results
    all_results = []

    # Perform analysis for each feature
    for feature in feature_cols:
        # Get test and control values
        test_values = feature_df[feature_df[perturbation_name_col].isin(test_genes)][
            feature
        ].dropna()
        control_values = feature_df[
            feature_df[perturbation_name_col].isin(control_genes)
        ][feature].dropna()

        # Skip if not enough data
        if len(test_values) < 3 or len(control_values) < 3:
            continue

        # Calculate statistics - use median for robustness
        median_test = test_values.median()
        median_control = control_values.median()

        # Calculate absolute difference
        diff = median_test - median_control

        # Calculate ratio/fold change, handling edge cases
        if median_control == 0 and median_test == 0:
            ratio = 1.0
        elif median_control == 0:
            ratio = float("inf") if median_test > 0 else float("-inf")
        else:
            ratio = median_test / median_control

        # Determine if feature is higher or lower in cluster
        direction = "higher" if diff > 0 else "lower" if diff < 0 else "unchanged"

        # Calculate interquartile ranges to understand variance
        q1_test, q3_test = test_values.quantile([0.25, 0.75])
        q1_control, q3_control = control_values.quantile([0.25, 0.75])
        iqr_test = q3_test - q1_test
        iqr_control = q3_control - q1_control

        # Determine if difference is significant based on simple thresholds
        # A feature is significant if:
        # 1. The absolute difference exceeds the threshold, AND
        # 2. The ratio (or its reciprocal) exceeds the ratio threshold
        if abs(diff) >= min_diff_threshold and (
            ratio >= min_ratio_threshold or ratio <= 1 / min_ratio_threshold
        ):
            is_significant = True
        else:
            is_significant = False

        # Store results
        all_results.append(
            {
                "feature": feature,
                "median_cluster": median_test,
                "median_control": median_control,
                "difference": diff,
                "ratio": ratio,
                "abs_diff": abs(diff),
                "direction": direction,
                "iqr_cluster": iqr_test,
                "iqr_control": iqr_control,
                "cluster_n": len(test_values),
                "control_n": len(control_values),
                "significant": is_significant,
            }
        )

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Sort by absolute difference (largest first)
    if not results_df.empty:
        results_df = results_df.sort_values(by="abs_diff", ascending=False)

        # Get top features higher in cluster
        higher_features = results_df[results_df["direction"] == "higher"].head(top_n)

        # Get top features lower in cluster
        lower_features = results_df[results_df["direction"] == "lower"].head(top_n)

        # Get all significant features
        significant = results_df[results_df["significant"]]

        # Store in results dictionary
        results["results_df"] = results_df
        results["higher_in_cluster"] = higher_features
        results["lower_in_cluster"] = lower_features
        results["significant"] = significant
    else:
        # Handle empty results
        results["results_df"] = pd.DataFrame()
        results["higher_in_cluster"] = pd.DataFrame()
        results["lower_in_cluster"] = pd.DataFrame()
        results["significant"] = pd.DataFrame()

    return results


def differential_analysis(
    feature_df,
    cluster_df,
    cluster_id,
    perturbation_name_col="gene_symbol_0",
    control_type="other_clusters",
    control_label=None,
    top_n=10,
    pval_threshold=0.05,
    log2fc_threshold=0.5,
    use_nonparametric=True,
    normalize_method="robust_zscore",
):
    """Perform differential analysis comparing features for genes in a specific cluster versus control genes.

    Conducts statistical testing to identify features that differ significantly between
    a target cluster and control groups. Supports multiple statistical approaches including
    parametric and non-parametric tests with various effect size calculations.

    Args:
        feature_df (pandas.DataFrame): DataFrame containing feature measurements with perturbation_name_col column.
        cluster_df (pandas.DataFrame): DataFrame containing cluster assignments with perturbation_name_col column.
        cluster_id (int or str): ID of the cluster to analyze.
        perturbation_name_col (str, optional): Column name containing perturbation identifiers. Defaults to "gene_symbol_0".
        control_type (str, optional): Either "other_clusters" or "nontargeting". Defaults to "other_clusters".
        control_label (str, optional): Label in perturbation_name_col that identifies non-targeting controls.
            Required if control_type is "nontargeting".
        top_n (int, optional): Number of top up/down-regulated features to return. Defaults to 10.
        pval_threshold (float, optional): p-value threshold for significance. Defaults to 0.05.
        log2fc_threshold (float, optional): Log2 fold change threshold for significance. Defaults to 0.5.
        use_nonparametric (bool, optional): If True, use Mann-Whitney U test instead of t-test. Defaults to True.
        normalize_method (str, optional): Method for calculating effect size:
            "robust_zscore" or "zscore". Defaults to "robust_zscore".

    Returns:
        dict: Dictionary containing:
            - 'results_df': Complete results DataFrame with all features
            - 'top_up': DataFrame with top upregulated features
            - 'top_down': DataFrame with top downregulated features
            - 'significant': DataFrame with all significant features
            - 'test_genes': List of genes in test cluster
            - 'control_genes': List of genes in control group
    """
    # Create output structure
    results = {}

    # Get genes in the test cluster
    test_genes = cluster_df[cluster_df["cluster"] == cluster_id][
        perturbation_name_col
    ].tolist()
    results["test_genes"] = test_genes

    # Determine control genes based on control_type
    if control_type == "other_clusters":
        control_genes = cluster_df[cluster_df["cluster"] != cluster_id][
            perturbation_name_col
        ].tolist()
    elif control_type == "nontargeting":
        if control_label is None:
            raise ValueError(
                "control_label must be provided when control_type is 'nontargeting'"
            )
        control_genes = feature_df[
            feature_df[perturbation_name_col].str.contains(control_label)
        ][perturbation_name_col].tolist()
    else:
        raise ValueError(
            "control_type must be either 'other_clusters' or 'nontargeting'"
        )

    results["control_genes"] = control_genes

    # Filter feature_df to only include numeric features
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [col for col in numeric_cols if col != "cluster"]

    # Initialize results dataframe
    all_results = []

    # Function to calculate robust z-score using median and MAD
    def robust_zscore(x, y):
        # Calculate median absolute deviation (MAD)
        median_x = np.median(x)
        median_y = np.median(y)
        mad_x = stats.median_abs_deviation(x, scale=1)
        mad_y = stats.median_abs_deviation(y, scale=1)

        # Handle edge cases
        if mad_x == 0 and mad_y == 0:
            return 0
        elif mad_x == 0:
            return float("inf") if median_x > median_y else float("-inf")
        elif mad_y == 0:
            return float("inf") if median_x > median_y else float("-inf")

        # Calculate pooled MAD
        n_x, n_y = len(x), len(y)
        pooled_mad = np.sqrt(
            ((n_x - 1) * mad_x**2 + (n_y - 1) * mad_y**2) / (n_x + n_y - 2)
        )

        # Calculate robust z-score (similar to Cohen's d but with medians and MADs)
        return (median_x - median_y) / pooled_mad

    # Perform analysis for each feature
    for feature in feature_cols:
        # Get test and control values
        test_values = feature_df[feature_df[perturbation_name_col].isin(test_genes)][
            feature
        ].dropna()
        control_values = feature_df[
            feature_df[perturbation_name_col].isin(control_genes)
        ][feature].dropna()

        # Skip if not enough data
        if len(test_values) < 3 or len(control_values) < 3:
            continue

        # Calculate statistics based on selected method
        mean_test = test_values.mean()
        mean_control = control_values.mean()
        median_test = test_values.median()
        median_control = control_values.median()

        if normalize_method == "robust_zscore":
            # Calculate robust z-score using median and MAD
            effect_size = robust_zscore(test_values, control_values)
            effect_size_name = "robust_zscore"

        elif normalize_method == "zscore":
            # Calculate Cohen's d (standardized mean difference)
            # Get standard deviations
            std_test = test_values.std()
            std_control = control_values.std()

            # Calculate pooled standard deviation
            n_test, n_control = len(test_values), len(control_values)
            pooled_std = np.sqrt(
                ((n_test - 1) * std_test**2 + (n_control - 1) * std_control**2)
                / (n_test + n_control - 2)
            )

            # Calculate Cohen's d
            if pooled_std == 0:
                effect_size = 0
            else:
                effect_size = (mean_test - mean_control) / pooled_std
            effect_size_name = "cohen_d"

        else:
            raise ValueError(f"Unknown normalize_method: {normalize_method}")

        # Calculate p-value using appropriate test
        if use_nonparametric:
            # Use Mann-Whitney U test (non-parametric)
            try:
                u_stat, p_val = mannwhitneyu(
                    test_values, control_values, alternative="two-sided"
                )
                test_name = "Mann-Whitney U"
            except ValueError:
                # Fall back to t-test if Mann-Whitney fails (e.g., if all values are identical)
                t_stat, p_val = ttest_ind(test_values, control_values, equal_var=False)
                test_name = "Welch's t-test (fallback)"
        else:
            # Use Welch's t-test (handles unequal variances)
            t_stat, p_val = ttest_ind(test_values, control_values, equal_var=False)
            test_name = "Welch's t-test"

        # Determine significance
        if normalize_method == "log2fc":
            significant = (p_val < pval_threshold) and (
                abs(effect_size) > log2fc_threshold
            )
        else:
            # For z-score based methods, use a different threshold approach
            effect_threshold = 0.8  # Large effect size in Cohen's d scale
            significant = (p_val < pval_threshold) and (
                abs(effect_size) > effect_threshold
            )

        # Store results
        all_results.append(
            {
                "feature": feature,
                "mean_test": mean_test,
                "mean_control": mean_control,
                "median_test": median_test,
                "median_control": median_control,
                effect_size_name: effect_size,
                "p_value": p_val,
                "test_used": test_name,
                "test_n": len(test_values),
                "control_n": len(control_values),
                "significant": significant,
            }
        )

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)

    # Sort by p-value and effect size
    if not results_df.empty:
        effect_col = effect_size_name  # Use the appropriate effect size column

        # For sorting, take the absolute value of the effect size but preserve direction for filtering
        results_df["abs_effect"] = results_df[effect_col].abs()

        # Sort by p-value and absolute effect size
        results_df = results_df.sort_values(
            by=["p_value", "abs_effect"], ascending=[True, False]
        )

        # Get top upregulated features (positive effect size)
        top_up = results_df[results_df[effect_col] > 0].head(top_n)

        # Get top downregulated features (negative effect size)
        top_down = results_df[results_df[effect_col] < 0].head(top_n)

        # Get all significant features
        significant = results_df[results_df["significant"]]

        # Drop the temporary column
        results_df = results_df.drop(columns=["abs_effect"])

        # Store in results dictionary
        results["results_df"] = results_df
        results["top_up"] = top_up
        results["top_down"] = top_down
        results["significant"] = significant
    else:
        # Handle empty results
        results["results_df"] = pd.DataFrame()
        results["top_up"] = pd.DataFrame()
        results["top_down"] = pd.DataFrame()
        results["significant"] = pd.DataFrame()

    return results


def waterfall_plot(
    feature_df,
    feature,
    cluster_df=None,
    cluster_id=None,
    perturbation_name_col="gene_symbol_0",
    nontargeting_pattern=None,
    figsize=(12, 7),
    title=None,
    label_genes=None,
):
    """Create a waterfall plot for a single feature with cluster gene highlighting.

    Generates a ranked visualization showing all genes ordered by their feature values,
    with special highlighting for genes from a specific cluster. Uses intelligent
    label placement to avoid overlaps.

    Args:
        feature_df (pandas.DataFrame): DataFrame containing the feature data with perturbation_name_col column.
        feature (str): Column name of the feature to plot.
        cluster_df (pandas.DataFrame, optional): DataFrame containing cluster assignments with perturbation_name_col column.
        cluster_id (int or str, optional): ID of the cluster to highlight.
        perturbation_name_col (str, optional): Column name containing perturbation identifiers. Defaults to "gene_symbol_0".
        nontargeting_pattern (str, optional): Pattern to identify non-targeting controls in perturbation_name_col.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).
        add_trendline (bool, optional): Whether to add a trendline to the plot. Defaults to True.
        title (str, optional): Custom title for the plot.
        label_genes (int, float, list, or None, optional): Controls which genes receive text labels:
            - If int: Label the top N genes with most extreme values (distance from origin)
            - If float: Label genes with combined absolute values (sqrt(x²+y²)) above this threshold
            - If list: Label only genes in this list
            - If None: Label all cluster genes (default)

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot. perturbation_name_col.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (12, 7).
        title (str, optional): Custom title for the plot.
        label_genes (int, float, list, or None, optional): Controls which genes receive text labels:
            - If int: Label the top N genes with most extreme values
            - If float: Label genes with absolute values above this threshold
            - If list: Label only genes in this list
            - If None: Label all cluster genes (default)

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """
    # Create a copy of the dataframe
    df_plot = feature_df.copy()

    # Get cluster genes
    cluster_genes = []
    if cluster_df is not None and cluster_id is not None:
        cluster_genes = cluster_df[cluster_df["cluster"] == cluster_id][
            perturbation_name_col
        ].tolist()

    # Create figure and axis with a clean style
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Sort dataframe by feature value
    df_plot = df_plot.sort_values(by=feature)

    # Create category column
    df_plot["category"] = "background"

    # Mark non-targeting controls
    if nontargeting_pattern is not None:
        nontargeting_mask = df_plot[perturbation_name_col].str.contains(
            nontargeting_pattern, case=False, na=False
        )
        df_plot.loc[nontargeting_mask, "category"] = "nontargeting"

    # Mark cluster genes
    if cluster_genes:
        cluster_mask = df_plot[perturbation_name_col].isin(cluster_genes)
        df_plot.loc[cluster_mask, "category"] = "cluster"

    # Assign x-coordinates based on position in sorted dataframe
    df_plot["x_pos"] = np.arange(len(df_plot))

    # Plot all background points first
    background_mask = df_plot["category"] == "background"
    background_scatter = ax.scatter(
        df_plot.loc[background_mask, "x_pos"],
        df_plot.loc[background_mask, feature],
        s=15,
        color="lightgray",
        alpha=0.4,
        edgecolors="none",
        marker="o",
        label="All genes",
    )

    # Plot non-targeting controls
    nt_scatter = None
    nt_mask = df_plot["category"] == "nontargeting"
    if nt_mask.any():
        nt_scatter = ax.scatter(
            df_plot.loc[nt_mask, "x_pos"],
            df_plot.loc[nt_mask, feature],
            s=15,
            color="#4575b4",
            alpha=0.8,
            edgecolors="none",
            linewidths=0.5,
            marker="o",
            label="Non-targeting controls",
        )

    # Plot cluster genes
    cluster_mask = df_plot["category"] == "cluster"
    if cluster_mask.any():
        cluster_scatter = ax.scatter(
            df_plot.loc[cluster_mask, "x_pos"],
            df_plot.loc[cluster_mask, feature],
            s=30,
            color="#d73027",
            alpha=1.0,
            edgecolors="black",
            linewidths=0.8,
            marker="o",
            label=f"Cluster {cluster_id} genes",
        )

        # Get cluster genes subset for labeling
        cluster_df_subset = df_plot[cluster_mask].copy()

        # Filter genes to label based on label_genes parameter
        genes_to_label = cluster_df_subset.copy()
        if label_genes is not None:
            if isinstance(label_genes, int):
                # Label top N most extreme values
                genes_to_label = cluster_df_subset.reindex(
                    cluster_df_subset[feature]
                    .abs()
                    .sort_values(ascending=False)
                    .head(label_genes)
                    .index
                )
            elif isinstance(label_genes, float):
                # Label values above threshold
                genes_to_label = cluster_df_subset[
                    cluster_df_subset[feature].abs() > label_genes
                ]
            elif isinstance(label_genes, list):
                # Label specific genes
                genes_to_label = cluster_df_subset[
                    cluster_df_subset[perturbation_name_col].isin(label_genes)
                ]

        # Create a list to hold all text objects for adjustText
        texts = []

        # Store original point coordinates for later drawing of arrows
        original_positions = {}

        # Create initial text objects for each gene we want to label
        # Place them with an initial offset to help the algorithm
        for _, row in genes_to_label.iterrows():
            # Initial vertical offset to give adjust_text a better starting point
            y_offset = 0.2 * (df_plot[feature].max() - df_plot[feature].min())
            initial_y = row[feature] + (
                y_offset if row[feature] < df_plot[feature].mean() else -y_offset
            )

            # Increased font size from 9 to 11
            text = ax.text(
                row["x_pos"],
                initial_y,
                row[perturbation_name_col],
                ha="center",
                va="center",
                fontsize=11,  # Increased font size
                fontweight="normal",
                color="#333333",
                zorder=5,
            )
            texts.append(text)

            # Store original point positions
            original_positions[text] = (row["x_pos"], row[feature])

        # Use adjustText to optimize label placement
        if texts:
            # Calculate plot bounds for constraint
            x_min, x_max = ax.get_xlim()
            y_min, y_max = ax.get_ylim()

            # Get ALL data points from ALL scatter plots for collision avoidance
            all_x_points = []
            all_y_points = []

            # Add background points
            all_x_points.extend(df_plot.loc[background_mask, "x_pos"].tolist())
            all_y_points.extend(df_plot.loc[background_mask, feature].tolist())

            # Add non-targeting points if any
            if nt_mask.any():
                all_x_points.extend(df_plot.loc[nt_mask, "x_pos"].tolist())
                all_y_points.extend(df_plot.loc[nt_mask, feature].tolist())

            # Add cluster points
            all_x_points.extend(df_plot.loc[cluster_mask, "x_pos"].tolist())
            all_y_points.extend(df_plot.loc[cluster_mask, feature].tolist())

            # Get all scatter objects
            scatter_points = [background_scatter]
            if nt_scatter is not None:
                scatter_points.append(nt_scatter)
            scatter_points.append(cluster_scatter)

            # Optimize text positions with better parameters - NO arrows yet
            adjust_text(
                texts,
                add_objects=scatter_points,  # Avoid all scatter points
                x=all_x_points,  # All x coordinates to avoid
                y=all_y_points,  # All y coordinates to avoid
                force_points=(0.2, 5.0),  # Much stronger force to avoid all points
                force_text=(0.0, 3.0),  # Stronger force between texts
                expand_points=(0.2, 5.0),  # Significantly expanded radius around points
                expand_text=(1.8, 1.8),  # Expanded text bounding box for larger text
                autoalign="xy",  # Auto-align on xy axis
                lim=1000,  # More iterations for better placement
                ax=ax,  # Reference to the axis
                # Allow text to move in x and y directions
                only_move={"points": "xy", "texts": "xy"},
                avoid_self=True,  # Avoid overlapping with each other
                arrowprops=None,  # Explicitly disable arrows in adjust_text
            )

            # Manually add the connecting lines after positioning is optimized
            for text in texts:
                # Get the original point position
                x_orig, y_orig = original_positions[text]

                # Get the text position
                x_text = text.get_position()[0]
                y_text = text.get_position()[1]

                # Create annotation with arrow manually
                ax.annotate(
                    "",  # Empty text since we already have the text object
                    xy=(x_orig, y_orig),  # Starting point (data point)
                    xytext=(x_text, y_text),  # Ending point (label position)
                    arrowprops=dict(
                        arrowstyle="-",
                        color="gray",
                        lw=0.7,
                        alpha=0.7,
                        connectionstyle="arc3,rad=0.0",
                    ),
                    zorder=4,  # Below text but above most other elements
                )

    # Add horizontal line at y=0 if data includes negative values
    if min(df_plot[feature]) < 0 and max(df_plot[feature]) > 0:
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    # Clean up the feature name for display
    feature_display = feature.replace("_", " ").title()

    # Set labels and title with professional styling
    ax.set_xlabel("Genes (Ranked)", fontsize=12)
    ax.set_ylabel(feature_display, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, pad=20, fontweight="bold", color="#333333")
    else:
        ax.set_title(
            f"{feature_display} Values Across Genes",
            fontsize=14,
            pad=20,
            fontweight="bold",
            color="#333333",
        )

    # Set x-axis limits with some padding
    ax.set_xlim(-len(df_plot) * 0.01, len(df_plot) * 1.01)

    # Add legend with clean styling
    legend = ax.legend(
        loc="upper right",
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        edgecolor="lightgray",
        fancybox=True,
    )
    legend.get_frame().set_linewidth(0.5)

    # Improve tick parameters
    ax.tick_params(axis="both", which="major", labelsize=10, colors="#333333")

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    plt.tight_layout()

    return ax


def two_feature_plot(
    feature_df,
    x,
    y,
    cluster_df=None,
    cluster_id=None,
    perturbation_name_col="gene_symbol_0",
    nontargeting_pattern=None,
    figsize=(10, 8),
    add_trendline=True,
    title=None,
    label_genes=None,
):
    """Create a scatter plot comparing two features with cluster gene highlighting.

    Visualizes the relationship between two features across all genes, with special
    highlighting for genes from a specific cluster. Includes optional trendline
    and intelligent label placement for selected genes using the adjustText library.

    Args:
        feature_df (pandas.DataFrame): DataFrame with the data to plot with perturbation_name_col column.
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data.
        cluster_df (pandas.DataFrame, optional): DataFrame containing cluster assignments with perturbation_name_col column.
        cluster_id (int or str, optional): ID of the cluster to highlight.
        perturbation_name_col (str, optional): Column name containing perturbation identifiers. Defaults to "gene_symbol_0".
        nontargeting_pattern (str, optional): Pattern to identify non-targeting controls in perturbation_name_col.
        figsize (tuple, optional): Figure size as (width, height). Defaults to (10, 8).
        add_trendline (bool, optional): Whether to add a trendline to the plot. Defaults to True.
        title (str, optional): Custom title for the plot.
        label_genes (int, float, list, or None, optional): Controls which genes receive text labels:
            - If int: Label the top N genes with most extreme values (distance from origin)
            - If float: Label genes with combined absolute values (sqrt(x²+y²)) above this threshold
            - If list: Label only genes in this list
            - If None: Label all cluster genes (default).

    Returns:
        matplotlib.axes.Axes: The axes object containing the plot.
    """
    # Create a copy of the dataframe
    df_plot = feature_df.copy()

    # Get cluster genes
    cluster_genes = []
    if cluster_df is not None and cluster_id is not None:
        cluster_genes = cluster_df[cluster_df["cluster"] == cluster_id][
            perturbation_name_col
        ].tolist()

    # Create figure and axis with a clean style
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Create category column
    df_plot["category"] = "background"

    # Mark non-targeting controls
    if nontargeting_pattern is not None:
        nontargeting_mask = df_plot[perturbation_name_col].str.contains(
            nontargeting_pattern, case=False, na=False
        )
        df_plot.loc[nontargeting_mask, "category"] = "nontargeting"

    # Mark cluster genes
    if cluster_genes:
        cluster_mask = df_plot[perturbation_name_col].isin(cluster_genes)
        df_plot.loc[cluster_mask, "category"] = "cluster"

    # Plot all background points first with consistent styling
    background_mask = df_plot["category"] == "background"
    background_scatter = ax.scatter(
        df_plot.loc[background_mask, x],
        df_plot.loc[background_mask, y],
        s=15,
        color="lightgray",
        alpha=0.4,
        edgecolors="none",
        marker="o",
        label="All genes",
    )

    # Plot non-targeting controls with consistent styling
    nt_scatter = None
    nt_mask = df_plot["category"] == "nontargeting"
    if nt_mask.any():
        nt_scatter = ax.scatter(
            df_plot.loc[nt_mask, x],
            df_plot.loc[nt_mask, y],
            s=15,
            color="#4575b4",
            alpha=0.8,
            edgecolors="none",
            linewidths=0.5,
            marker="o",
            label="Non-targeting controls",
        )

    # Plot cluster genes with consistent styling
    cluster_mask = df_plot["category"] == "cluster"
    if cluster_mask.any():
        cluster_scatter = ax.scatter(
            df_plot.loc[cluster_mask, x],
            df_plot.loc[cluster_mask, y],
            s=30,
            color="#d73027",
            alpha=1.0,
            edgecolors="black",
            linewidths=0.8,
            marker="o",
            label=f"Cluster {cluster_id} genes",
        )

        # Get cluster genes subset for labeling
        cluster_subset = df_plot[cluster_mask].copy()

        # Calculate distance from origin for threshold filtering
        cluster_subset["distance"] = np.sqrt(
            cluster_subset[x] ** 2 + cluster_subset[y] ** 2
        )

        # Filter genes to label based on label_genes parameter
        genes_to_label = cluster_subset.copy()
        if label_genes is not None:
            if isinstance(label_genes, int):
                # Label top N most extreme values (by distance from origin)
                genes_to_label = cluster_subset.reindex(
                    cluster_subset["distance"]
                    .sort_values(ascending=False)
                    .head(label_genes)
                    .index
                )
            elif isinstance(label_genes, float):
                # Label values above threshold distance
                genes_to_label = cluster_subset[
                    cluster_subset["distance"] > label_genes
                ]
            elif isinstance(label_genes, list):
                # Label specific genes
                genes_to_label = cluster_subset[
                    cluster_subset[perturbation_name_col].isin(label_genes)
                ]

        # Create a list to hold all text objects for adjustText
        texts = []

        # Store original point coordinates for later drawing of arrows
        original_positions = {}

        # Calculate means for intelligent initial placement
        x_mean = df_plot[x].mean()
        y_mean = df_plot[y].mean()

        # Calculate ranges for smart initial offsets
        x_range = df_plot[x].max() - df_plot[x].min()
        y_range = df_plot[y].max() - df_plot[y].min()

        # Create initial text objects for each gene we want to label
        # With smart initial placement based on quadrants
        for _, row in genes_to_label.iterrows():
            # Determine which quadrant the point is in
            x_pos = row[x]
            y_pos = row[y]

            # Enhanced initial placement with larger offset to start away from ALL points
            offset_x = 0.2 * x_range * (1 if x_pos < x_mean else -1)
            offset_y = 0.2 * y_range * (1 if y_pos < y_mean else -1)

            # Increased font size from 9 to 11
            text = ax.text(
                x_pos + offset_x,
                y_pos + offset_y,
                row[perturbation_name_col],
                ha="center",
                va="center",
                fontsize=11,
                fontweight="normal",
                color="#333333",
                zorder=5,
            )
            texts.append(text)

            # Store original point positions
            original_positions[text] = (x_pos, y_pos)

        # Use adjustText to optimize label placement
        if texts:
            # Create point avoidance coordinates with specific buffers
            # This makes sure labels stay away from ALL points
            all_scatter_xy = {}

            # Collect all point coordinates with specific collection logic
            for category, scatter_obj in [
                ("background", background_scatter),
                ("nontargeting", nt_scatter if nt_mask.any() else None),
                ("cluster", cluster_scatter),
            ]:
                if scatter_obj is not None:
                    all_scatter_xy[category] = (
                        scatter_obj.get_offsets()[:, 0],
                        scatter_obj.get_offsets()[:, 1],
                    )

            # Combine all points for avoidance
            all_x_points = np.concatenate([xy[0] for xy in all_scatter_xy.values()])
            all_y_points = np.concatenate([xy[1] for xy in all_scatter_xy.values()])

            # Optimize text positions with much stronger parameters
            adjust_text(
                texts,
                add_objects=[background_scatter, nt_scatter, cluster_scatter]
                if nt_scatter is not None
                else [background_scatter, cluster_scatter],
                x=all_x_points,
                y=all_y_points,
                force_points=(
                    8,
                    10,
                ),  # Much stronger force to ensure points are avoided
                force_text=(3, 3),  # Stronger force between texts
                expand_points=(
                    9,
                    9,
                ),  # Significantly larger radius around points to avoid
                expand_text=(4, 4),  # Expanded text bounding box for better spacing
                autoalign="xy",  # Auto-align on xy axis
                lim=2000,  # Many more iterations for better placement
                ax=ax,
                only_move={"points": "xy", "texts": "xy"},
                avoid_self=True,  # Avoid text overlapping
                arrowprops=None,  # No arrows in adjustText (we'll add manually)
            )

            # Manually add connecting lines after positioning is optimized
            for text in texts:
                # Get the original point position
                x_orig, y_orig = original_positions[text]

                # Get the text position
                x_text, y_text = text.get_position()

                # Create annotation with arrow manually
                ax.annotate(
                    "",  # Empty text since we already have the text object
                    xy=(x_orig, y_orig),  # Starting point (data point)
                    xytext=(x_text, y_text),  # Ending point (label position)
                    arrowprops=dict(
                        arrowstyle="-",
                        color="gray",
                        lw=0.7,
                        alpha=0.7,
                        connectionstyle="arc3,rad=0.0",
                    ),
                    zorder=4,  # Below text but above most other elements
                )

    # Add trendline if requested with professional styling
    if add_trendline:
        try:
            # Calculate correlation
            corr = df_plot[[x, y]].corr().iloc[0, 1]

            # Add linear regression line
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                df_plot[x].dropna(), df_plot[y].dropna()
            )

            x_range_vals = np.linspace(df_plot[x].min(), df_plot[x].max(), 100)
            ax.plot(
                x_range_vals,
                intercept + slope * x_range_vals,
                color="#555555",
                linestyle="--",
                linewidth=1.5,
                label=f"Trend (r={corr:.2f})",
            )
        except Exception as e:
            print(f"Warning: Could not add trendline: {e}")

    # Add horizontal and vertical lines at y=0 and x=0 if data includes negative values
    if min(df_plot[x]) < 0 and max(df_plot[x]) > 0:
        ax.axvline(x=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)
    if min(df_plot[y]) < 0 and max(df_plot[y]) > 0:
        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5, alpha=0.5)

    # Clean up feature names for display
    x_display = x.replace("_", " ").title()
    y_display = y.replace("_", " ").title()

    # Set labels and title with professional styling
    ax.set_xlabel(x_display, fontsize=12)
    ax.set_ylabel(y_display, fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, pad=20, fontweight="bold", color="#333333")
    else:
        ax.set_title(
            f"{x_display} vs {y_display}",
            fontsize=14,
            pad=20,
            fontweight="bold",
            color="#333333",
        )

    # Add legend with clean styling
    legend = ax.legend(
        loc="best",
        frameon=True,
        framealpha=0.95,
        fontsize=10,
        edgecolor="lightgray",
        fancybox=True,
    )
    legend.get_frame().set_linewidth(0.5)

    # Improve tick parameters
    ax.tick_params(axis="both", which="major", labelsize=10, colors="#333333")

    # Remove top and right spines for cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#333333")
    ax.spines["bottom"].set_color("#333333")

    plt.tight_layout()

    return ax


def cluster_heatmap(
    feature_df,
    cluster_df,
    cluster_ids,
    features,
    perturbation_name_col="gene_symbol_0",
    figsize=(18, 10),
    cmap="seismic",
    title=None,
    transpose=True,
    robust=True,
    center=0,
    z_score="global",
    label_size=20,
    max_genes_displayed=75,
    cluster_method="average",
    group_by_cluster=False,
):
    """Create a hierarchically clustered heatmap for visualizing feature patterns across cluster genes.

    Generates a heatmap showing normalized feature values for genes in specified clusters,
    with hierarchical clustering applied to both genes and features for pattern discovery.
    Supports multiple normalization methods and customizable display options.

    Args:
        feature_df (pandas.DataFrame): DataFrame containing feature measurements with perturbation_name_col column.
        cluster_df (pandas.DataFrame): DataFrame containing cluster assignments with perturbation_name_col column.
        cluster_ids (int, str, or list): ID(s) of the cluster(s) to visualize.
        features (list): List of feature columns to include in the heatmap.
        perturbation_name_col (str, optional): Column name containing perturbation identifiers. Defaults to "gene_symbol_0".
        figsize (tuple, optional): Figure size as (width, height). Defaults to (18, 10).
        cmap (str, optional): Colormap to use for the heatmap. Defaults to "seismic".
        title (str, optional): Custom title for the plot.
        transpose (bool, optional): If True, features are on y-axis and genes on x-axis. Defaults to True.
        robust (bool, optional): If True, compute robust min and max for colormap. Defaults to True.
        center (float, optional): Value at which to center the colormap. Defaults to 0.
        z_score (str or None, optional): Normalization method:
            - "global": Normalize using mean and std of entire feature_df
            - "local": Normalize using mean and std of only the subset being visualized
            - "rows": Normalize each row
            - "columns": Normalize each column
            - None: No normalization.
            Defaults to "global".
        label_size (int, optional): Font size for axis labels. Defaults to 20.
        max_genes_displayed (int, optional): Maximum number of gene labels to display. Defaults to 75.
        cluster_method (str, optional): Linkage method for hierarchical clustering. Defaults to 'average'.
        group_by_cluster (bool, optional): If True, group genes by cluster instead of hierarchical clustering. Defaults to False.

    Returns:
        tuple: (clustermap_obj, heatmap_data) - ClusterGrid object and processed DataFrame.
    """
    # Ensure cluster_ids is a list
    if not isinstance(cluster_ids, list):
        cluster_ids = [cluster_ids]

    # Get genes for each cluster
    cluster_genes_dict = {}
    for cluster_id in cluster_ids:
        genes = cluster_df[cluster_df["cluster"] == cluster_id][
            perturbation_name_col
        ].tolist()
        if genes:
            cluster_genes_dict[cluster_id] = genes
        else:
            print(f"No genes found in cluster {cluster_id}")

    if not cluster_genes_dict:
        print("No genes found in any of the specified clusters")
        return None

    # Combine all genes from all clusters
    all_cluster_genes = []
    for genes in cluster_genes_dict.values():
        all_cluster_genes.extend(genes)

    # Filter data to only include genes in the clusters
    df_heatmap = feature_df[
        feature_df[perturbation_name_col].isin(all_cluster_genes)
    ].copy()

    # Add a column for cluster assignment
    df_heatmap["cluster"] = np.nan
    for cluster_id, genes in cluster_genes_dict.items():
        df_heatmap.loc[df_heatmap[perturbation_name_col].isin(genes), "cluster"] = (
            cluster_id
        )

    # Set gene column as index
    df_heatmap = df_heatmap.set_index(perturbation_name_col)

    # Filter to include only the specified features
    valid_features = [f for f in features if f in df_heatmap.columns]

    if not valid_features:
        print("No valid features found")
        return None

    # Create display name mapping for features
    feature_display_names = {
        feat: feat.replace("_", " ").title() for feat in valid_features
    }

    # Extract just the features for the heatmap
    heatmap_data = df_heatmap[valid_features].copy()

    # Apply z-score normalization based on selected method
    if z_score == "global":
        # Global normalization: Use the mean and std from the entire feature_df
        global_means = feature_df[valid_features].mean()
        global_stds = feature_df[valid_features].std()

        # Normalize each feature using global statistics
        for feature in valid_features:
            if global_stds[feature] > 0:  # Avoid division by zero
                heatmap_data[feature] = (
                    heatmap_data[feature] - global_means[feature]
                ) / global_stds[feature]
            else:
                heatmap_data[feature] = 0  # If std is zero, set to zero

    elif z_score == "local":
        # Local normalization: Only normalize based on the subset of data
        heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

    elif z_score == "rows":
        # Normalize each row (gene)
        heatmap_data = heatmap_data.sub(heatmap_data.mean(axis=1), axis=0).div(
            heatmap_data.std(axis=1), axis=0
        )

    elif z_score == "columns":
        # Normalize each column (feature)
        heatmap_data = (heatmap_data - heatmap_data.mean()) / heatmap_data.std()

    # Rename features to their display names
    heatmap_data.columns = [
        feature_display_names[feat] for feat in heatmap_data.columns
    ]

    # Create cluster annotation for genes
    cluster_annotations = df_heatmap.loc[heatmap_data.index, "cluster"]

    # Create color mapping for clusters
    unique_clusters = sorted(cluster_annotations.unique())
    cluster_colors = sns.color_palette("Set2", len(unique_clusters))
    cluster_color_map = dict(zip(unique_clusters, cluster_colors))

    # Create annotation Series for genes (not DataFrame)
    gene_colors = cluster_annotations.map(cluster_color_map)

    # Transpose data if requested (genes on x-axis, features on y-axis)
    if transpose:
        heatmap_data = heatmap_data.T
        # No need to transpose gene_colors since it's now a Series

    # Set appropriate color bar title based on normalization method
    if z_score is None:
        cbar_label = "Value"
    elif z_score == "global":
        cbar_label = "Global Z-score"
    elif z_score == "local":
        cbar_label = "Local Z-score"
    elif z_score == "rows":
        cbar_label = "Row Z-score"
    elif z_score == "columns":
        cbar_label = "Column Z-score"

    # Handle gene label display limits
    if transpose:
        # Genes are now columns
        if len(all_cluster_genes) > max_genes_displayed:
            # Create sparse labels
            n_genes = heatmap_data.shape[1]
            display_indices = np.linspace(
                0, n_genes - 1, min(max_genes_displayed, n_genes), dtype=int
            )
            col_labels = [
                heatmap_data.columns[i] if i in display_indices else ""
                for i in range(n_genes)
            ]
            xticklabels = col_labels
        else:
            xticklabels = True
        yticklabels = True  # Always show feature labels
    else:
        # Genes are rows
        if len(all_cluster_genes) > max_genes_displayed:
            n_genes = heatmap_data.shape[0]
            display_indices = np.linspace(
                0, n_genes - 1, min(max_genes_displayed, n_genes), dtype=int
            )
            row_labels = [
                heatmap_data.index[i] if i in display_indices else ""
                for i in range(n_genes)
            ]
            yticklabels = row_labels
        else:
            yticklabels = True
        xticklabels = True  # Always show feature labels

    # Determine clustering parameters
    if group_by_cluster:
        # Group by cluster with hierarchical clustering within each cluster
        from scipy.cluster.hierarchy import linkage, dendrogram
        from scipy.spatial.distance import pdist

        # Create ordered index with hierarchical clustering within each cluster
        ordered_genes = []

        for cluster_id in sorted(unique_clusters):
            # Get genes in this cluster
            cluster_mask = cluster_annotations == cluster_id
            cluster_genes = cluster_annotations[cluster_mask].index.tolist()

            if len(cluster_genes) > 1:
                # Get data for this cluster
                if transpose:
                    cluster_data = heatmap_data[cluster_genes].T
                else:
                    cluster_data = heatmap_data.loc[cluster_genes]

                # Perform hierarchical clustering within this cluster
                try:
                    # Calculate distance matrix
                    dist_matrix = pdist(cluster_data, metric="euclidean")
                    # Perform hierarchical clustering
                    linkage_matrix = linkage(dist_matrix, method=cluster_method)
                    # Get the order of genes from dendrogram
                    dendro = dendrogram(linkage_matrix, no_plot=True)
                    # Reorder genes based on clustering
                    reordered_genes = [cluster_genes[i] for i in dendro["leaves"]]
                    ordered_genes.extend(reordered_genes)
                except:
                    # If clustering fails, just use original order
                    ordered_genes.extend(cluster_genes)
            else:
                # Single gene, no clustering needed
                ordered_genes.extend(cluster_genes)

        # Reorder the data based on clustered order
        if transpose:
            heatmap_data = heatmap_data[ordered_genes]
            gene_colors = gene_colors[ordered_genes]
            # Still cluster rows (features)
            row_cluster = True
            col_cluster = False
        else:
            heatmap_data = heatmap_data.loc[ordered_genes]
            gene_colors = gene_colors[ordered_genes]
            # Still cluster columns (features)
            row_cluster = False
            col_cluster = True
    else:
        # Full hierarchical clustering
        row_cluster = True
        col_cluster = True

    # Create the clustermap
    g = sns.clustermap(
        heatmap_data,
        method=cluster_method,
        cmap=cmap,
        center=center,
        robust=robust,
        figsize=figsize,
        # Annotation parameters
        row_colors=gene_colors if not transpose else None,
        col_colors=gene_colors if transpose else None,
        # Clustering parameters
        row_cluster=row_cluster,
        col_cluster=col_cluster,
        # Dendrogram parameters - use minimal ratio instead of 0
        dendrogram_ratio=0.01,  # Very small but not zero to avoid layout issues
        colors_ratio=0.04,  # Small ratio for color bar
        # Label parameters
        xticklabels=xticklabels,
        yticklabels=yticklabels,
        # Colorbar parameters
        cbar_kws={"label": cbar_label, "shrink": 0.5},
    )

    # Hide the dendrogram axes after creation
    if hasattr(g, "ax_row_dendrogram") and g.ax_row_dendrogram is not None:
        g.ax_row_dendrogram.set_visible(False)
    if hasattr(g, "ax_col_dendrogram") and g.ax_col_dendrogram is not None:
        g.ax_col_dendrogram.set_visible(False)

    # Remove labels from cluster color bars
    if hasattr(g, "ax_row_colors") and g.ax_row_colors is not None:
        g.ax_row_colors.set_ylabel("")
        g.ax_row_colors.set_xlabel("")
        # Remove all tick labels
        g.ax_row_colors.set_xticklabels([])
        g.ax_row_colors.set_yticklabels([])
        # Turn off ticks
        g.ax_row_colors.tick_params(left=False, right=False, top=False, bottom=False)
    if hasattr(g, "ax_col_colors") and g.ax_col_colors is not None:
        g.ax_col_colors.set_ylabel("")
        g.ax_col_colors.set_xlabel("")
        # Remove all tick labels
        g.ax_col_colors.set_xticklabels([])
        g.ax_col_colors.set_yticklabels([])
        # Turn off ticks
        g.ax_col_colors.tick_params(left=False, right=False, top=False, bottom=False)

    # Add cluster labels directly on the color bar
    if transpose and hasattr(g, "ax_col_colors"):
        # Get the positions and add text labels
        ax = g.ax_col_colors
        # Group genes by cluster for labeling
        cluster_positions = {}
        for i, (gene, cluster) in enumerate(
            zip(heatmap_data.columns, gene_colors.index)
        ):
            cluster_val = cluster_annotations[gene]
            if cluster_val not in cluster_positions:
                cluster_positions[cluster_val] = []
            cluster_positions[cluster_val].append(i)

        # Add text labels for each cluster
        for cluster_id, positions in cluster_positions.items():
            if positions:
                # Calculate center position for this cluster
                center_pos = np.mean(positions)
                ax.text(
                    center_pos,
                    0.5,
                    f"{int(cluster_id)}",
                    ha="center",
                    va="center",
                    fontsize=label_size,
                    fontweight="bold",
                    transform=ax.get_xaxis_transform(),
                )

    elif not transpose and hasattr(g, "ax_row_colors"):
        # Get the positions and add text labels
        ax = g.ax_row_colors
        # Group genes by cluster for labeling
        cluster_positions = {}
        for i, (gene, cluster) in enumerate(zip(heatmap_data.index, gene_colors.index)):
            cluster_val = cluster_annotations[gene]
            if cluster_val not in cluster_positions:
                cluster_positions[cluster_val] = []
            cluster_positions[cluster_val].append(i)

        # Add text labels for each cluster
        for cluster_id, positions in cluster_positions.items():
            if positions:
                # Calculate center position for this cluster
                center_pos = np.mean(positions)
                ax.text(
                    0.5,
                    center_pos,
                    f"{int(cluster_id)}",
                    ha="center",
                    va="center",
                    fontsize=label_size - 4,
                    fontweight="bold",
                    transform=ax.get_yaxis_transform(),
                )

    # Customize the appearance
    # Set font sizes
    g.ax_heatmap.tick_params(labelsize=label_size)

    # Remove the bottom axis label (gene_symbol_0)
    g.ax_heatmap.set_xlabel("")

    if transpose:
        # Genes on x-axis
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=90, ha="center")
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
    else:
        # Features on x-axis
        plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
        plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)

    # Add title with better positioning
    title = title or (
        f"Cluster {cluster_ids[0]}" if len(cluster_ids) == 1 else "Multiple Clusters"
    )
    # Adjust the subplot to make room for title
    g.fig.subplots_adjust(top=0.92)
    g.fig.suptitle(title, fontsize=25, y=0.96)

    # Move the colorbar to bottom right as horizontal with reduced width
    cbar = g.ax_cbar
    if cbar is not None:
        # Remove the original colorbar
        cbar.remove()

        # Create new horizontal colorbar at bottom right with reduced width
        cbar_ax = g.fig.add_axes(
            [0.95, 0.02, 0.10, 0.025]
        )  # [left, bottom, width, height] - reduced width
        cbar_new = plt.colorbar(
            g.ax_heatmap.collections[0],
            cax=cbar_ax,
            orientation="horizontal",
            label=cbar_label,
        )
        cbar_new.ax.tick_params(labelsize=label_size - 6)
        cbar_new.ax.set_xlabel(cbar_label, fontsize=label_size - 4)

    return g


def volcano_plot(
    merged_df,
    feature,
    perturbation_name_col="gene_symbol_0",
    cluster_df=None,
    cluster_id=None,
    fdr_threshold=0.05,
    zscore_threshold=2.0,
    figsize=(10, 8),
    title=None,
    label_genes=None,
    show_thresholds=True,
):
    """Create a volcano plot showing effect size vs significance.

    Visualizes the relationship between feature effect sizes (z-scores) and
    statistical significance (-log10 p-values) from bootstrap analysis.
    Optionally highlights genes from a specific cluster.

    Args:
        merged_df (pandas.DataFrame): DataFrame with feature z-scores and bootstrap
            statistics (must contain {feature} and {feature}_log10 columns).
        feature (str): Feature name to plot (z-score column).
        perturbation_name_col (str, optional): Column name for gene identifiers.
            Defaults to "gene_symbol_0".
        cluster_df (pandas.DataFrame, optional): DataFrame with cluster assignments.
        cluster_id (int or str, optional): Cluster ID to highlight.
        fdr_threshold (float, optional): FDR threshold for significance line.
            Defaults to 0.05.
        zscore_threshold (float, optional): Z-score threshold for effect size lines.
            Defaults to 2.0.
        figsize (tuple, optional): Figure size. Defaults to (10, 8).
        title (str, optional): Plot title.
        label_genes (int, list, or None, optional): Controls gene labeling:
            - If int: Label top N most significant genes
            - If list: Label only these specific genes
            - If None: Label significant cluster genes (default)
        show_thresholds (bool, optional): Whether to show threshold lines.
            Defaults to True.

    Returns:
        tuple: (fig, ax) matplotlib figure and axes objects.
    """
    # Check required columns
    log10_col = f"{feature}_log10"
    fdr_col = f"{feature}_fdr"

    if feature not in merged_df.columns:
        raise ValueError(f"Feature column '{feature}' not found in DataFrame")
    if log10_col not in merged_df.columns:
        raise ValueError(f"Log10 p-value column '{log10_col}' not found in DataFrame")

    # Create working copy
    df_plot = merged_df.copy()
    df_plot = df_plot.dropna(subset=[feature, log10_col])

    # Get cluster genes if provided
    cluster_genes = []
    if cluster_df is not None and cluster_id is not None:
        cluster_genes = cluster_df[cluster_df["cluster"] == cluster_id][
            perturbation_name_col
        ].tolist()

    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=300)

    # Assign categories
    df_plot["category"] = "background"
    if cluster_genes:
        cluster_mask = df_plot[perturbation_name_col].isin(cluster_genes)
        df_plot.loc[cluster_mask, "category"] = "cluster"

    # Mark significant genes (pass both thresholds)
    if fdr_col in df_plot.columns:
        sig_mask = (df_plot[fdr_col] < fdr_threshold) & (
            df_plot[feature].abs() >= zscore_threshold
        )
        df_plot.loc[sig_mask & (df_plot["category"] == "background"), "category"] = (
            "significant"
        )

    # Plot background points
    bg_mask = df_plot["category"] == "background"
    ax.scatter(
        df_plot.loc[bg_mask, feature],
        df_plot.loc[bg_mask, log10_col],
        s=15,
        color="lightgray",
        alpha=0.5,
        edgecolors="none",
        label="Background",
    )

    # Plot significant points
    sig_mask = df_plot["category"] == "significant"
    if sig_mask.any():
        ax.scatter(
            df_plot.loc[sig_mask, feature],
            df_plot.loc[sig_mask, log10_col],
            s=25,
            color="#756bb1",
            alpha=0.7,
            edgecolors="none",
            label="Significant",
        )

    # Plot cluster points
    cluster_mask = df_plot["category"] == "cluster"
    if cluster_mask.any():
        ax.scatter(
            df_plot.loc[cluster_mask, feature],
            df_plot.loc[cluster_mask, log10_col],
            s=40,
            color="#d73027",
            alpha=1.0,
            edgecolors="black",
            linewidths=0.8,
            label=f"Cluster {cluster_id}",
            zorder=10,
        )

    # Add threshold lines
    if show_thresholds:
        # Z-score thresholds
        ax.axvline(
            x=zscore_threshold,
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )
        ax.axvline(
            x=-zscore_threshold,
            color="gray",
            linestyle="--",
            alpha=0.5,
            linewidth=1,
        )

        # FDR threshold (convert to -log10)
        if fdr_col in df_plot.columns:
            # Find approximate -log10(p) for FDR threshold
            # Use genes near the FDR threshold to estimate
            near_threshold = df_plot[
                (df_plot[fdr_col] > fdr_threshold * 0.5)
                & (df_plot[fdr_col] < fdr_threshold * 2)
            ]
            if len(near_threshold) > 0:
                approx_log10 = near_threshold[log10_col].median()
                ax.axhline(
                    y=approx_log10,
                    color="gray",
                    linestyle="--",
                    alpha=0.5,
                    linewidth=1,
                )

    # Determine genes to label
    genes_to_label = pd.DataFrame()
    if label_genes is not None:
        if isinstance(label_genes, int):
            # Label top N most significant
            genes_to_label = df_plot.nlargest(label_genes, log10_col)
        elif isinstance(label_genes, list):
            # Label specific genes
            genes_to_label = df_plot[df_plot[perturbation_name_col].isin(label_genes)]
    elif cluster_genes:
        # Label significant cluster genes by default
        cluster_data = df_plot[cluster_mask]
        if fdr_col in df_plot.columns:
            genes_to_label = cluster_data[
                (cluster_data[fdr_col] < fdr_threshold)
                | (cluster_data[feature].abs() >= zscore_threshold)
            ]
        else:
            genes_to_label = cluster_data[
                cluster_data[feature].abs() >= zscore_threshold
            ]

    # Add labels with adjustText
    if len(genes_to_label) > 0:
        texts = []
        for _, row in genes_to_label.iterrows():
            texts.append(
                ax.text(
                    row[feature],
                    row[log10_col],
                    row[perturbation_name_col],
                    fontsize=8,
                    ha="center",
                    va="bottom",
                )
            )
        if texts:
            adjust_text(
                texts,
                ax=ax,
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5),
            )

    # Labels and title
    ax.set_xlabel(f"{feature} (z-score)", fontsize=12)
    ax.set_ylabel(r"$-\log_{10}$(p-value)", fontsize=12)

    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f"Volcano Plot: {feature}", fontsize=14)

    ax.legend(loc="upper right", framealpha=0.9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig, ax
