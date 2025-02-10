"""Helper functions for evaluating results of merge process."""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from lib.shared.eval import plot_plate_heatmap


def plot_sbs_ph_matching_heatmap(
    df_merge,
    df_info,
    target="sbs",
    shape="square",
    plate="6W",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plots the rate of matching segmented cells between phenotype and SBS datasets by well and tile in a convenient plate layout.

    Args:
        df_merge: DataFrame of all matched cells, e.g., concatenated outputs for all tiles and wells
            of `Snake.merge_triangle_hash()`. Expects 'tile' and 'cell_0' columns to correspond to phenotype data and
            'site', 'cell_1' columns to correspond to SBS data.
        df_info: DataFrame of all cells segmented from either phenotype or SBS images, e.g., concatenated outputs for all tiles
            and wells of `Snake.extract_phenotype_minimal(data_phenotype=nulcei, nuclei=nuclei)`, often used as `sbs_cell_info`
            rule in Snakemake.
        target: Which dataset to use as the target, e.g., if target='sbs', plots the fraction of cells in each SBS tile
            that match to a phenotype cell. Should match the information stored in df_info; if df_info is a table of all
            segmented cells from SBS tiles, then target should be set as 'sbs'.
        shape: Shape of subplot for each well used in `plot_plate_heatmap`. Defaults to 'square' and infers shape based on
            the value of `target`.
        plate: Plate type for `plot_plate_heatmap`, options are {'6W', '24W', '96W'}.
        return_plot: If True, returns `df_summary`.
        return_summary: If True, returns `df_summary`.
        **kwargs: Additional keyword arguments passed to `plot_plate_heatmap()`.

    Returns:
        df_summary: DataFrame used for plotting, returned if `return_summary=True`.
        axes: Numpy array of matplotlib Axes objects.
    """
    # Determine the merge columns and source based on the target
    if target == "sbs":
        merge_cols = ["site", "cell_1"]
        source = "phenotype"
        # Determine the default shape if not provided
        if not shape:
            shape = "6W_sbs"
    elif target == "phenotype":
        merge_cols = ["tile", "cell_0"]
        source = "sbs"
        # Determine the default shape if not provided
        if not shape:
            shape = "6W_ph"
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
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return df_summary, axes[0]
    elif return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
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


def plot_cell_positions(df, title, color=None, hue="channels_min"):
    """Generates a scatter plot of cell positions in the i_0, j_0 coordinate space.

    Args:
        df: DataFrame containing cell position data with i_0, j_0 columns.
        title: Plot title.
        color: Fixed color for all points. If specified, overrides hue.
        hue: Column name for color variation. Defaults to 'channels_min'.

    Returns:
        The generated matplotlib figure object.
    """
    fig = plt.figure(figsize=(20, 20))

    # Plot scatter with either fixed color or hue-based coloring
    if color is not None:
        sns.scatterplot(data=df, x="i_0", y="j_0", color=color, alpha=0.5)
    else:
        sns.scatterplot(data=df, x="i_0", y="j_0", hue=hue, alpha=0.5)

    plt.title(title)
    plt.xlabel("i_0")
    plt.ylabel("j_0")
    return fig