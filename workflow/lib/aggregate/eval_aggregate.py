"""Helper functions for evaluating the results of the evaluation step in the aggregate module.

This module includes functions for generating visualizations and testing data integrity.
It provides the following functionalities:
- Visualization of feature distributions across datasets with violin plots.
- Detection and reporting of missing values, including NA, null, blank, and infinite values.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_feature_distributions(combined_cell_data, features, remove_clean=False):
    """Create violin plots with jittered points for feature distributions across different datasets.

    Uses a log scale for clean and transformed data and a linear scale for standardized data.
    Points are colored based on control vs. non-control status.

    Args:
        combined_cell_data (dict): Dictionary of dataframes with keys as dataset names (e.g., "clean", "transformed").
        features (list): List of feature names to plot.
        remove_clean (bool): Whether to remove clean data from the plot.

    Returns:
        matplotlib.figure.Figure: The generated figure containing the violin plots.
    """
    # Prepare data for plotting
    plot_data = []

    # Remove clean data if requested
    if remove_clean:
        combined_cell_data = {
            key: combined_cell_data
            for key, combined_cell_data in combined_cell_data.items()
            if key != "clean"
        }

    for dataset_name, cell_data in combined_cell_data.items():
        for feature in features:
            feature_data = pd.DataFrame(
                {
                    "Dataset": dataset_name,
                    "Well": cell_data["well"],
                    "Value": cell_data[feature],
                    "Feature": feature,
                }
            )
            plot_data.append(feature_data)

    # Combine all data
    plot_df = pd.concat(plot_data, ignore_index=True)

    # Create subplot for each feature
    n_features = len(features)
    fig, axes = plt.subplots(1, n_features, figsize=(7 * n_features, 8))

    if n_features == 1:
        axes = [axes]

    # Plot violins for each feature
    for ax, feature in zip(axes, features):
        # Create a second axis that shares the same x-axis
        ax2 = ax.twinx()

        # Plot log-scale data (clean and transformed) on main axis
        log_data = plot_df[
            (plot_df["Feature"] == feature)
            & (plot_df["Dataset"].isin(["clean", "transformed"]))
        ]
        if not log_data.empty:
            # Violin plot with high transparency
            sns.violinplot(
                data=log_data,
                x="Dataset",
                y="Value",
                hue="Well",
                ax=ax,
                inner=None,
                cut=0,
                scale="width",
                split=False,
                alpha=0.5,
            )

            ax.set_yscale("log")

        # Plot linear-scale data (standardized) on second axis
        linear_data = plot_df[
            (plot_df["Feature"] == feature) & (plot_df["Dataset"] == "standardized")
        ]
        if not linear_data.empty:
            # Violin plot with high transparency
            sns.violinplot(
                data=linear_data,
                x="Dataset",
                y="Value",
                hue="Well",
                ax=ax2,
                inner=None,
                cut=0,
                scale="width",
                split=False,
                alpha=0.5,
            )

        # Customize plot
        ax.set_title(feature.replace("_", " ").title())
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45)

        # Set y-axis labels
        ax.set_ylabel("Value (log scale)")
        ax2.set_ylabel("Value (linear scale)")

        # Remove redundant legends from violin plots
        if ax.get_legend():
            ax.get_legend().remove()

    # Adjust layout to prevent label cutoff
    plt.tight_layout()

    return fig


def test_missing_values(gene_data, name):
    """Test for missing values in a dataframe, including NA, null, blank, and infinite values.

    Returns results in a format suitable for CSV export.

    Args:
        gene_data (pandas.DataFrame): DataFrame to test.
        name (str): Name of the dataset for printing.

    Returns:
        pandas.DataFrame: DataFrame containing detailed missing value information.
    """
    # Check for various types of missing or problematic values
    missing = pd.DataFrame(
        {
            "null_na": gene_data.isna().sum(),  # Catches np.nan, None, pd.NA
            "empty_string": gene_data.astype(str).eq("").sum(),  # Empty strings
            "infinite": gene_data.isin([np.inf, -np.inf]).sum(),  # Infinite values
        }
    )

    # Sum up all types of problematic values
    missing["total_issues"] = missing.sum(axis=1)

    # Calculate percentages
    missing["percentage"] = (missing["total_issues"] / len(gene_data) * 100).round(2)

    # Add dataset name
    missing["dataset"] = name

    # Add column names as a separate column
    missing["column"] = missing.index

    # Reorder columns
    missing = missing[
        [
            "dataset",
            "column",
            "total_issues",
            "percentage",
            "null_na",
            "empty_string",
            "infinite",
        ]
    ]

    # Filter for columns that have any issues
    results = missing[missing["total_issues"] > 0].reset_index(drop=True)

    # Print summary
    if not results.empty:
        print(f"\nMissing values in {name} dataset:")
        for _, row in results.iterrows():
            details = []
            if row["null_na"] > 0:
                details.append(f"{row['null_na']} NA/null")
            if row["empty_string"] > 0:
                details.append(f"{row['empty_string']} empty")
            if row["infinite"] > 0:
                details.append(f"{row['infinite']} infinite")

            print(
                f"{row['column']}: {row['total_issues']} total issues "
                f"({row['percentage']:.2f}%) - {', '.join(details)}"
            )
    else:
        print(f"\nNo missing values found in {name} dataset")

    return results
