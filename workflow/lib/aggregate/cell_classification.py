"""This module provides functions for visualizing and analyzing cell populations.

Functions include:
- Plotting the distribution of mitotic cells using histograms and scatter plots.
- Splitting cell populations into mitotic and interphase groups based on feature thresholds.

Functions:
    - plot_mitotic_distribution_hist: Plot histogram of a feature and calculate the percentage of mitotic cells.
    - plot_mitotic_distribution_scatter: Plot a scatter plot of two features with threshold cutoffs.
    - split_mitotic_simple: Split cells into mitotic and interphase populations based on thresholds.

"""

import matplotlib.pyplot as plt


def plot_mitotic_distribution_hist(
    cell_data, threshold_variable, threshold_value, bins=100
):
    """Plot distribution of the threshold variable and calculate percent of mitotic cells.

    Args:
        cell_data (pd.DataFrame): Input dataframe containing cell measurements.
        threshold_variable (str): Column name for mitotic cell identification.
        threshold_value (float): Threshold value for separating mitotic cells.
        bins (int): Number of bins for histogram.

    Returns:
        float: Percentage of cells classified as mitotic.
    """
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.hist(cell_data[threshold_variable], bins=bins)
    plt.title(f"Histogram of {threshold_variable}")
    plt.xlabel(threshold_variable)
    plt.ylabel("Frequency")
    plt.axvline(
        x=threshold_value,
        color="r",
        linestyle="--",
        label=f"Mitotic threshold ({threshold_value})",
    )
    plt.legend()
    plt.show()

    # Calculate percent mitotic
    mitotic_mask = cell_data[threshold_variable] > threshold_value
    percent_mitotic = (mitotic_mask.sum() / len(cell_data)) * 100

    print(f"Number of mitotic cells: {mitotic_mask.sum():,}")
    print(f"Total cells: {len(cell_data):,}")
    print(f"Percent mitotic: {percent_mitotic:.2f}%")


def plot_mitotic_distribution_scatter(
    cell_data,
    threshold_variable_x,
    threshold_variable_y,
    threshold_x,
    threshold_y,
    alpha=0.5,
):
    """Plot scatter plot of two variables with two threshold cutoffs.

    Args:
        cell_data (pd.DataFrame): Input dataframe containing cell measurements.
        threshold_variable_x (str): Column name for x-axis variable.
        threshold_variable_y (str): Column name for y-axis variable.
        threshold_x (float): Threshold value for x-axis.
        threshold_y (float): Threshold value for y-axis.
        alpha (float): Transparency of points.

    Returns:
        None
    """
    # Create plot
    plt.figure(figsize=(10, 6))
    plt.scatter(
        cell_data[threshold_variable_x], cell_data[threshold_variable_y], alpha=alpha
    )
    plt.title(f"Scatter plot of {threshold_variable_x} vs {threshold_variable_y}")
    plt.xlabel(threshold_variable_x)
    plt.ylabel(threshold_variable_y)
    plt.axvline(
        x=threshold_x,
        color="r",
        linestyle="--",
        label=f"Mitotic threshold ({threshold_x})",
    )
    plt.axhline(
        y=threshold_y,
        color="g",
        linestyle="--",
        label=f"Mitotic threshold ({threshold_y})",
    )
    plt.legend()
    plt.show()

    # Calculate percent mitotic
    mitotic_mask_x = cell_data[threshold_variable_x] > threshold_x
    mitotic_mask_y = cell_data[threshold_variable_y] > threshold_y
    percent_mitotic = (mitotic_mask_x & mitotic_mask_y).sum() / len(cell_data) * 100

    print(f"Number of mitotic cells: {sum(mitotic_mask_x & mitotic_mask_y):,}")
    print(f"Total cells: {len(cell_data):,}")
    print(f"Percent mitotic: {percent_mitotic:.2f}%")


def split_mitotic_simple(cell_data, conditions):
    """Split cells into mitotic and interphase populations based on feature thresholds.

    Args:
        cell_data (pd.DataFrame): Input dataframe.
        conditions (dict): Dictionary mapping feature names to (threshold, direction) tuples
            where direction is 'greater' or 'less'.

    Returns:
        tuple: (mitotic_df, interphase_df) pair of DataFrames.
    """
    mitotic_df = cell_data.copy()

    for feature, (cutoff, direction) in conditions.items():
        if direction == "greater":
            mitotic_df = mitotic_df[mitotic_df[feature] > cutoff]
        elif direction == "less":
            mitotic_df = mitotic_df[mitotic_df[feature] < cutoff]
        else:
            raise ValueError("Direction must be 'greater' or 'less'")

    interphase_df = cell_data.drop(mitotic_df.index)

    return mitotic_df, interphase_df
