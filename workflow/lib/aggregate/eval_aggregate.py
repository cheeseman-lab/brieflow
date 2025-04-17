"""Helper functions for evaluating the results of the evaluation step in the aggregate module.

This module includes functions for generating visualizations and testing data integrity.
It provides the following functionalities:
- Suggesting parameters for feature analysis based on input data.
- Visualization of feature distributions across datasets with violin plots.
- Detection and reporting of missing values, including NA, null, blank, and infinite values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def nas_summary(cell_data, vis_subsample=None):
    """Creates a visualization matrix highlighting NA values and returns summary statistics.

    Args:
        cell_data (pandas.DataFrame): The DataFrame to analyze for NA values.
        vis_subsample (int, optional): Number of samples to visualize. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - pandas.DataFrame: A DataFrame with columns: 'column', 'num_nas', and 'percent_na'.
            - matplotlib.figure.Figure or None: The figure object if NAs are found, None otherwise.
    """
    cols_with_na = cell_data.columns[cell_data.isna().any()].tolist()

    if not cols_with_na:
        print("No columns with NA values found in the DataFrame.")
        return pd.DataFrame(columns=["column", "num_nas", "percent_na"]), None

    na_counts = cell_data[cols_with_na].isna().sum()
    na_percent = na_counts / len(cell_data)

    na_summary_df = pd.DataFrame(
        {
            "column": cols_with_na,
            "percent_na": na_percent.values,
        }
    )

    if vis_subsample is not None:
        if vis_subsample > len(cell_data):
            vis_subsample = len(cell_data)
        cell_data = cell_data.sample(vis_subsample)

    plt.figure(figsize=(15, 7))
    plt.title(f"NA Values Matrix ({len(cols_with_na)} columns with missing values)")

    ax = sns.heatmap(
        cell_data[cols_with_na].isna(), cmap="viridis", cbar=False, yticklabels=False
    )

    plt.xticks(rotation=90)
    plt.tight_layout()

    return na_summary_df, plt.gcf()


def summarize_cell_data(
    cell_data: pd.DataFrame, classes: list, collapse_cols: list
) -> pd.DataFrame:
    """Summarizes cell data by counting total cells, class-specific cells, and unique metric values.

    Args:
        cell_data (pd.DataFrame): DataFrame containing cell metadata.
        classes (list): List of class names to filter.
        collapse_cols (list): List of column names to count unique values.

    Returns:
        pd.DataFrame: Summary table with stage names and corresponding counts/percentages.
    """
    counts = [("Raw Data", len(cell_data))]

    for class_name in classes:
        class_subset = cell_data[cell_data["class"] == class_name]
        counts.append((f"{class_name} cells", len(class_subset)))

        for col in collapse_cols:
            counts.append((f"{class_name} {col}", class_subset[col].nunique()))

    summary_df = pd.DataFrame(counts, columns=["Stage", "Count"])
    total = summary_df.loc[summary_df["Stage"] == "Raw Data", "Count"].values[0]
    summary_df["Percent"] = (summary_df["Count"] / total * 100).round(2)
    return summary_df


def plot_feature_distributions(
    original_feature_cols,
    original_cell_data,
    aligned_feature_cols,
    aligned_cell_data,
):
    """Plot violin plots comparing original and aligned feature distributions.

    Generates a subplot grid where each row corresponds to a feature pair
    (original vs aligned). Each subplot shows value distributions across
    plate_well combinations with outliers clipped to the 1st and 99th percentiles.

    Args:
        original_feature_cols: List of column names for original features.
        original_cell_data: DataFrame containing original features and 'plate', 'well' columns.
        aligned_feature_cols: List of column names for aligned features (e.g., PCs).
        aligned_cell_data: DataFrame containing aligned features and 'plate', 'well' columns.

    Returns:
        matplotlib.figure.Figure: Figure containing the violin plots.
    """
    # Melt original features
    df_orig = original_cell_data[["plate", "well"] + original_feature_cols].melt(
        id_vars=["plate", "well"], var_name="Feature", value_name="Value"
    )
    df_orig["plate_well"] = (
        df_orig["plate"].astype(int).astype(str) + "_" + df_orig["well"]
    )
    df_orig["Type"] = "Original"

    # Melt aligned features
    df_aligned = aligned_cell_data[["plate", "well"] + aligned_feature_cols].melt(
        id_vars=["plate", "well"], var_name="Feature", value_name="Value"
    )
    df_aligned["plate_well"] = (
        df_aligned["plate"].astype(int).astype(str) + "_" + df_aligned["well"]
    )
    df_aligned["Type"] = "Aligned"

    # Clip outliers to 1st–99th percentiles
    df_orig = df_orig.groupby(["Feature", "plate_well"], group_keys=False).apply(
        lambda g: g[
            (g["Value"] >= g["Value"].quantile(0.01))
            & (g["Value"] <= g["Value"].quantile(0.99))
        ]
    )
    df_aligned = df_aligned.groupby(["Feature", "plate_well"], group_keys=False).apply(
        lambda g: g[
            (g["Value"] >= g["Value"].quantile(0.01))
            & (g["Value"] <= g["Value"].quantile(0.99))
        ]
    )

    # Create plot
    fig, axes = plt.subplots(
        len(original_feature_cols),
        2,
        figsize=(16, 3 * len(original_feature_cols)),
        sharey=False,
    )
    fig.suptitle("Original vs Aligned Feature Distributions")

    for i, (orig_col, aligned_col) in enumerate(
        zip(original_feature_cols, aligned_feature_cols)
    ):
        df_o = df_orig[df_orig["Feature"] == orig_col]
        df_a = df_aligned[df_aligned["Feature"] == aligned_col]

        sns.violinplot(
            data=df_o,
            x="plate_well",
            y="Value",
            ax=axes[i, 0],
            cut=0,
            density_norm="width",
            inner="quartile",
        )
        axes[i, 0].set_title(f"{orig_col}")
        axes[i, 0].set_ylabel("Original Value")
        axes[i, 0].tick_params(axis="x", rotation=45)
        if i < len(original_feature_cols) - 1:
            axes[i, 0].set_xlabel("")

        sns.violinplot(
            data=df_a,
            x="plate_well",
            y="Value",
            ax=axes[i, 1],
            cut=0,
            density_norm="width",
            inner="quartile",
        )
        axes[i, 1].set_title(f"{aligned_col}")
        axes[i, 1].yaxis.set_label_position("right")
        axes[i, 1].yaxis.tick_right()
        axes[i, 1].set_ylabel("Aligned Value")
        axes[i, 1].tick_params(axis="x", rotation=45)
        if i < len(original_feature_cols) - 1:
            axes[i, 1].set_xlabel("")

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig
