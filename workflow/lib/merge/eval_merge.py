"""Helper functions for evaluating results of merge process."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy.spatial.distance import cdist
import yaml

from lib.shared.eval import plot_plate_heatmap


def plot_sbs_ph_matching_heatmap(
    df_merge,
    df_info,
    target="sbs",
    metadata=None,
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plots the rate of matching segmented cells between phenotype and SBS datasets by well and tile in a convenient plate layout.

    Args:
        df_merge: DataFrame of all matched cells, e.g., concatenated outputs for all tiles and wells
            of merge_triangle_hash. Expects 'tile' and 'cell_0' columns to correspond to phenotype data and
            'site', 'cell_1' columns to correspond to SBS data.
        df_info: DataFrame of all cells segmented from either phenotype or SBS images, e.g., concatenated outputs for all tiles
            and wells of extract_phenotype_minimal(data_phenotype=nulcei, nuclei=nuclei), often used as `sbs_cell_info`
            rule in Snakemake.
        metadata: Optional metadata DataFrame with x_pos/y_pos for spatial plotting.
        target: Which dataset to use as the target, e.g., if target='sbs', plots the fraction of cells in each SBS tile
            that match to a phenotype cell. Should match the information stored in df_info; if df_info is a table of all
            segmented cells from SBS tiles, then target should be set as 'sbs'.
        return_plot: If True, returns figure.
        return_summary: If True, returns `df_summary`.
        **kwargs: Additional keyword arguments passed to `plot_plate_heatmap()`.

    Returns:
        df_summary: DataFrame used for plotting, returned if `return_summary=True`.
        matplotlib.figure.Figure: The figure object.
    """
    # Determine the merge columns and source based on the target
    if target == "sbs":
        merge_cols = ["site", "cell_1"]
        source = "phenotype"
    elif target == "phenotype":
        merge_cols = ["tile", "cell_0"]
        source = "sbs"
    else:
        raise ValueError("target = {} not implemented".format(target))

    # Calculate the summary dataframe
    df_summary = (
        df_info.rename(columns={"tile": merge_cols[0], "cell": merge_cols[1]})[
            ["well"] + merge_cols
        ]
        .merge(
            df_merge[["well"] + merge_cols + ["distance"]],
            how="left",
            on=["well"] + merge_cols,
        )
        .assign(matched=lambda x: x["distance"].notna())
        .groupby(["well"] + merge_cols[:1])["matched"]
        .value_counts(normalize=True)
        .rename("fraction of {} cells matched to {} cells".format(target, source))
        .to_frame()
        .reset_index()
        .query("matched==True")
        .drop(columns="matched")
        .rename(columns={merge_cols[0]: "tile"})
    )

    if return_summary and return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, metadata=metadata, **kwargs)
        return df_summary, axes[0]
    elif return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, metadata=metadata, **kwargs)
        return axes[0]
    elif return_summary:
        return df_summary
    else:
        return None


def plot_channel_histogram(df_before, df_after, channel_min_cutoff=0):
    """Generates a histogram of channel values with raw counts and consistent bin edges.

    Args:
        df_before: DataFrame containing channel values before cleaning.
        df_after: DataFrame containing channel values after cleaning.
        channel_min_cutoff: Threshold value to mark with a red vertical line. Defaults to 0.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(10, 6))

    # Calculate bin edges based on the full range of data
    min_val = min(df_before["channels_min"].min(), df_after["channels_min"].min())
    max_val = max(df_before["channels_min"].max(), df_after["channels_min"].max())
    bins = np.linspace(min_val, max_val, 201)  # 201 edges make 200 bins

    # Plot histograms with raw counts instead of density
    plt.hist(
        df_before["channels_min"].dropna(),
        bins=bins,
        color="blue",
        alpha=0.5,
        label="Before clean",
    )
    plt.hist(
        df_after["channels_min"].dropna(),
        bins=bins,
        color="orange",
        alpha=0.5,
        label="After clean",
    )

    # Add vertical line for channel_min_cutoff
    plt.axvline(channel_min_cutoff, color="red", linestyle="--", label="Cutoff")

    plt.title("Histogram of channels_min Values")
    plt.xlabel("channels_min")
    plt.ylabel("Count")
    plt.legend()
    return fig


def plot_cell_positions(df_merge, title, color=None, hue="channels_min"):
    """Generates a scatter plot of cell positions, preferring global coords when present.

    Args:
        df_merge: DataFrame containing cell position data. Uses global_i_0/global_j_0
            if present, otherwise falls back to i_0/j_0.
        title: Plot title.
        color: Fixed color for all points. If specified, overrides hue.
        hue: Column name for color variation. Defaults to 'channels_min'.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(20, 20))

    i_col = "global_i_0" if "global_i_0" in df_merge.columns else "i_0"
    j_col = "global_j_0" if "global_j_0" in df_merge.columns else "j_0"

    if color is not None:
        sns.scatterplot(data=df_merge, x=i_col, y=j_col, color=color, alpha=0.5)
    else:
        # seaborn's numeric-hue legend locator cannot interpret pandas nullable
        # extension dtypes (e.g. Int64); cast the hue to plain float64 for plotting.
        plot_df = df_merge
        if pd.api.types.is_extension_array_dtype(df_merge[hue].dtype):
            plot_df = df_merge.assign(**{hue: df_merge[hue].astype("float64")})
        sns.scatterplot(data=plot_df, x=i_col, y=j_col, hue=hue, alpha=0.5)

    plt.title(title)
    plt.xlabel(i_col)
    plt.ylabel(j_col)
    return fig
