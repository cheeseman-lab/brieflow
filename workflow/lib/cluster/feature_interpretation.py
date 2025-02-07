"""Module for interpreting features during clustering module processing.

This module provides functions for visualization, feature ranking, and axis transformations to help
interpret and analyze features in clustering workflows. The functions generate scatter plots, heatmaps,
and apply transformations to better visualize complex datasets.

Functions:
    - rank_transform: Transform features in a DataFrame to their rank values.
    - two_feature: Create a scatter plot comparing two features with optional annotations and controls.
    - heatmap: Generate a heatmap with optional clustering and color annotations.
    - symlog_axis: Apply a symmetrical log scale to an axis for better visualization of large dynamic ranges.
"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.ticker import SymmetricalLogLocator, FixedLocator
import seaborn as sns
from adjustText import adjust_text


def rank_transform(df, non_feature_cols=["gene_symbol_0"]):
    """Transform features in a dataframe to their rank values, where highest value gets rank 1.

    Args:
        df (pd.DataFrame): Input dataframe with features to be ranked.
        non_feature_cols (list): List of column names that should not be ranked (e.g., identifiers, counts).

    Returns:
        pd.DataFrame: New dataframe with same structure but feature values replaced with ranks.
    """
    # Get feature columns (all columns not in non_feature_cols)
    feature_cols = [col for col in df.columns if col not in non_feature_cols]

    # Create ranks for all feature columns at once
    ranked_features = df[feature_cols].rank(ascending=False).astype(int)

    # Combine non-feature columns with ranked features
    ranked = pd.concat([df[non_feature_cols], ranked_features], axis=1)

    return ranked


def two_feature(
    df,
    x,
    y,
    annotate_query=None,
    annotate_labels=False,
    annotate_kwargs=dict(edgecolor="black"),
    xscale=None,
    yscale=None,
    control_query=None,
    control_kwargs=dict(),
    ax=None,
    rasterized=True,
    adjust_labels=True,
    save_plot_path=None,
    **kwargs,
):
    """Create a scatter plot comparing two features.

    Args:
        df (pd.DataFrame): DataFrame with the data to plot.
        x (str): Column name for x-axis data.
        y (str): Column name for y-axis data.
        annotate_query (str, optional): Query to subset data for annotation.
        annotate_labels (bool or str): Column name for annotation labels.
        annotate_kwargs (dict): Additional arguments for annotated points.
        xscale (str, optional): Scale for x-axis ("symlog" or None).
        yscale (str, optional): Scale for y-axis ("symlog" or None).
        control_query (str, optional): Query to subset control data.
        control_kwargs (dict): Additional arguments for control points.
        ax (matplotlib.axes.Axes, optional): Axes to plot on.
        rasterized (bool): If True, use rasterized rendering.
        adjust_labels (bool): If True, adjust label positions to avoid overlap.
        save_plot_path (str, optional): Path to save the plot as an image.
        **kwargs: Additional arguments for seaborn's `scatterplot`.

    Returns:
        matplotlib.axes.Axes: The Axes object with the plot.
    """
    df_ = df.copy()

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(7, 7))

    if annotate_query is not None:
        df_annotate = df_.query(annotate_query)

    if control_query is not None:
        df_control = df_.query(control_query)
        df_ = df_[~(df_.index.isin(df_control.index))]

    sns.scatterplot(data=df_, x=x, y=y, ax=ax, rasterized=rasterized, **kwargs)

    if control_query is not None:
        _ = control_kwargs.setdefault("color", sns.color_palette()[1])
        kwargs_ = kwargs.copy()
        kwargs_.update(control_kwargs)
        sns.scatterplot(
            data=df_control,
            x=x,
            y=y,
            ax=ax,
            rasterized=rasterized,
            **kwargs_,
        )

    if annotate_query is not None:
        kwargs_ = kwargs.copy()
        _ = annotate_kwargs.setdefault("edgecolor", "black")
        _ = annotate_kwargs.setdefault("alpha", 1)
        kwargs_.update(annotate_kwargs)
        sns.scatterplot(
            data=df_annotate,
            x=x,
            y=y,
            ax=ax,
            rasterized=rasterized,
            **kwargs_,
        )
        if annotate_labels:
            labels = []
            for _, entry in df_annotate.iterrows():
                labels.append(
                    ax.annotate(
                        entry[annotate_labels],
                        (entry[x], entry[y]),
                        arrowprops=dict(
                            arrowstyle="-", relpos=(0, 0), shrinkA=0, shrinkB=0
                        ),
                    )
                )

    # Apply symlog scale if specified
    if xscale == "symlog":
        ax = symlog_axis(df_[x], ax, "x")

    if yscale == "symlog":
        ax = symlog_axis(df_[y], ax, "y")

    ax.set_xlabel(" ".join(x.split("_")))
    ax.set_ylabel(" ".join(y.split("_")))

    if adjust_labels:
        try:
            adjust_text(
                labels,
                df_[x].values,
                df_[y].values,
                ax=ax,
                force_text=(0.1, 0.05),
                force_points=(0.01, 0.025),
            )
        except:
            pass

    if save_plot_path:
        plt.savefig(save_plot_path, dpi=300, bbox_inches="tight")

    return ax


def heatmap(
    df,
    figsize=None,
    row_colors=None,
    col_colors=None,
    row_palette="Set2",
    col_palette="Set2",
    label_fontsize=5,
    rasterized=True,
    colors_ratio=0.1,
    spinewidth=0.25,
    alternate_ticks=(True, True),
    alternate_tick_length=(30, 30),
    label_every=(1, 1),
    xticklabel_kwargs=dict(),
    yticklabel_kwargs=dict(),
    xticks_emphasis=[],
    yticks_emphasis=[],
    save_plot_path=None,
    **kwargs,
):
    """Generates a heatmap with optional clustering and color annotations.

    Note:
    Weird things happen if you make the heatmap aspect ratio too big/small
    (e.g., figsize=(1.3,6) looks about the same as (2.7,6)).

    Args:
        df (pd.DataFrame): Data for the heatmap.
        figsize (tuple, optional): Figure size (width, height). If None, size is auto-calculated.
        row_colors (str or pd.Series, optional): Column name or Series for row color annotations. If str, it's a column name; if pd.Series, it should contain color values.
        col_colors (str or pd.Series, optional): Index name or Series for column color annotations. If str, it's an index name; if pd.Series, it should contain color values.
        row_palette (str, optional): Palette name for row colors. Defaults to 'Set2'.
        col_palette (str, optional): Palette name for column colors. Defaults to 'Set2'.
        label_fontsize (int, optional): Font size for axis labels. Defaults to 5.
        rasterized (bool, optional): Whether to rasterize the heatmap. Defaults to True.
        colors_ratio (float, optional): Ratio for the color bar size. Defaults to 0.1.
        spinewidth (float, optional): Width of the spines around the heatmap. Defaults to 0.25.
        alternate_ticks (tuple, optional): Whether to alternate major and minor ticks for x and y axes. Defaults to (True, True).
        alternate_tick_length (tuple, optional): Length of major and minor ticks. Defaults to (30, 30).
        label_every (tuple, optional): Frequency of label display for x and y axes. Defaults to (1, 1).
        xticklabel_kwargs (dict, optional): Additional keyword arguments for x-axis tick labels.
        yticklabel_kwargs (dict, optional): Additional keyword arguments for y-axis tick labels.
        xticks_emphasis (list, optional): List of x-axis tick labels to emphasize.
        yticks_emphasis (list, optional): List of y-axis tick labels to emphasize.
        save_plot_path (str, optional): Path to save the plot as an image.
        **kwargs: Additional arguments for seaborn’s `clustermap`.

    Returns:
        sns.matrix.ClusterGrid: The clustermap object with the heatmap.
    """
    # Extract parameters from kwargs
    vmin = kwargs.pop("vmin", -5)
    vmax = kwargs.pop("vmax", 5)
    col_cluster = kwargs.pop("col_cluster", False)
    row_cluster = kwargs.pop("row_cluster", False)
    cmap = kwargs.pop("cmap", "vlag")
    cbar_pos = kwargs.pop("cbar_pos", None)

    print(cbar_pos)

    # Check and set dendrogram ratio
    if col_cluster or row_cluster:
        dendrogram_ratio = kwargs.pop("dendrogram_ratio", 0.1)
        if dendrogram_ratio == 0:
            raise ValueError(
                "dendrogram_ratio must be greater than zero if clustering rows or columns."
            )
    else:
        dendrogram_ratio = kwargs.pop("dendrogram_ratio", 0)

    # Process row colors if provided
    if isinstance(row_colors, str):
        # Map row color column to colors using specified palette
        row_color_map = {
            group: color
            for group, color in zip(
                df[row_colors].unique(),
                sns.color_palette(row_palette, n_colors=df[row_colors].nunique()),
            )
        }
        row_colors = df[row_colors].map(row_color_map)
        row_divisions = (
            row_colors.value_counts()[row_colors.unique()].cumsum().values[:-1]
        )

    # Process column colors if provided
    if isinstance(col_colors, str):
        # Map column color index to colors using specified palette
        col_color_map = {
            group: color
            for group, color in zip(
                df.loc[col_colors].unique(),
                sns.color_palette(col_palette, n_colors=df.loc[col_colors].nunique()),
            )
        }
        col_colors = df.loc[col_colors].map(col_color_map)
        col_divisions = (
            col_colors.value_counts()[col_colors.unique()].cumsum().values[:-1]
        )

    # Remove color columns from dataframe
    if row_colors is not None:
        df = df[[col for col in df.columns if col != row_colors.name]]
    if col_colors is not None:
        df = df.loc[[index != col_colors.name for index in df.index]]

    # Calculate figure size if not provided
    y_len, x_len = df.shape
    if figsize is None:
        figsize = np.array([x_len, y_len]) * 0.08

    # Create clustermap
    cg = sns.clustermap(
        df,
        figsize=figsize,
        row_colors=row_colors,
        col_colors=col_colors,
        vmin=vmin,
        vmax=vmax,
        col_cluster=col_cluster,
        row_cluster=row_cluster,
        cmap=cmap,
        dendrogram_ratio=dendrogram_ratio,
        colors_ratio=colors_ratio,
        rasterized=rasterized,
        cbar_pos=cbar_pos,
        **kwargs,
    )

    # Update layout to remove space between subplots
    cg.gs.update(hspace=0, wspace=0)

    # Remove color axis ticks and add dividers if clustering is not applied
    if row_colors is not None:
        cg.ax_row_colors.set_xticks([])
        if not row_cluster:
            [
                cg.ax_heatmap.axhline(d, color="black", linestyle="--", linewidth=0.25)
                for d in row_divisions
            ]
    if col_colors is not None:
        cg.ax_col_colors.set_yticks([])
        if not col_cluster:
            [
                cg.ax_heatmap.axvline(d, color="black", linestyle="--", linewidth=0.25)
                for d in col_divisions
            ]

    # Remove axis labels
    cg.ax_heatmap.set_ylabel(None)
    cg.ax_heatmap.set_xlabel(None)

    # Add spines with specified width
    for _, spine in cg.ax_heatmap.spines.items():
        spine.set_visible(True)
        spine.set_lw(spinewidth)

    # Set up tick labels and ticks with alternating offsets
    x_le, y_le = label_every
    x_at, y_at = alternate_ticks
    x_atl, y_atl = alternate_tick_length
    ytickrotation = yticklabel_kwargs.pop("rotation", "horizontal")
    xtickrotation = xticklabel_kwargs.pop("rotation", "vertical")

    if col_cluster:
        x_labels = df.columns.get_level_values(0)[cg.dendrogram_col.reordered_ind]
    else:
        x_labels = df.columns.get_level_values(0)
    if row_cluster:
        y_labels = df.index.get_level_values(0)[cg.dendrogram_row.reordered_ind]
    else:
        y_labels = df.index.get_level_values(0)

    # Set major and minor ticks with alternating labels
    cg.ax_heatmap.xaxis.set_major_locator(
        FixedLocator(
            np.linspace(
                0.5,
                x_len - (x_len - 0.5) % (x_le * 2),
                int(np.ceil((x_len - 0.5) / (2 * x_le))),
            )
        )
    )
    cg.ax_heatmap.xaxis.set_minor_locator(
        FixedLocator(
            np.linspace(
                0.5 + x_le,
                x_len - (x_len - 0.5 - x_le) % (x_le * 2),
                int(np.ceil((x_len - 0.5 - x_le) / (2 * x_le))),
            )
        )
    )
    cg.ax_heatmap.yaxis.set_major_locator(
        FixedLocator(
            np.linspace(
                0.5,
                y_len - (y_len - 0.5) % (y_le * 2),
                int(np.ceil((y_len - 0.5) / (2 * y_le))),
            )
        )
    )
    cg.ax_heatmap.yaxis.set_minor_locator(
        FixedLocator(
            np.linspace(
                0.5 + y_le,
                y_len - (y_len - 0.5 - y_le) % (y_le * 2),
                int(np.ceil((y_len - 0.5 - y_le) / (2 * y_le))),
            )
        )
    )
    _ = cg.ax_heatmap.set_yticklabels(
        y_labels[:: 2 * y_le],
        fontsize=label_fontsize,
        rotation=ytickrotation,
        rotation_mode="anchor",
        **yticklabel_kwargs,
    )
    _ = cg.ax_heatmap.set_yticklabels(
        y_labels[y_le :: 2 * y_le],
        minor=True,
        fontsize=label_fontsize,
        rotation=ytickrotation,
        rotation_mode="anchor",
        **yticklabel_kwargs,
    )
    _ = cg.ax_heatmap.set_xticklabels(
        x_labels[:: 2 * x_le],
        fontsize=label_fontsize,
        rotation=xtickrotation,
        rotation_mode="anchor",
        **xticklabel_kwargs,
    )
    _ = cg.ax_heatmap.set_xticklabels(
        x_labels[x_le :: 2 * x_le],
        minor=True,
        fontsize=label_fontsize,
        rotation=xtickrotation,
        rotation_mode="anchor",
        **xticklabel_kwargs,
    )

    # Set tick parameters for both major and minor ticks
    if y_at:
        cg.ax_heatmap.tick_params(axis="y", which="major", pad=2, length=2)
        cg.ax_heatmap.tick_params(axis="y", which="minor", pad=2, length=y_atl)
    else:
        cg.ax_heatmap.tick_params(axis="y", which="both", pad=2, length=2)
    if x_at:
        cg.ax_heatmap.tick_params(axis="x", which="major", pad=-2, length=2)
        cg.ax_heatmap.tick_params(axis="x", which="minor", pad=-2, length=x_atl)
    else:
        cg.ax_heatmap.tick_params(axis="x", which="both", pad=-2, length=2)

    # Emphasize specific ticks if provided
    if len(xticks_emphasis) > 0:
        [
            tick.set_visible(False)
            for tick in cg.ax_heatmap.get_xticklabels(which="both")
            if tick.get_text() in xticks_emphasis
        ]
    if len(yticks_emphasis) > 0:
        [
            tick.set_color("red")
            for tick in cg.ax_heatmap.get_yticklabels(which="both")
            if tick.get_text() in yticks_emphasis
        ]

    if save_plot_path:
        cg.savefig(save_plot_path, dpi=150, bbox_inches="tight")

    return cg


def symlog_axis(vals, ax, which):
    """Sets a symmetrical log scale for the specified axis.

    Args:
        vals (array-like): Data values for setting the axis limits.
        ax (matplotlib.axes.Axes): The Axes object to modify.
        which (str): Axis to modify ("x" or "y").

    Returns:
        matplotlib.axes.Axes: The Axes object with the modified axis scale.
    """
    if which == "x":
        ax.set_xscale("symlog", linthresh=1, base=10, subs=np.arange(1, 11))
        op_ax = ax.xaxis
    elif which == "y":
        ax.set_yscale("symlog", linthresh=1, base=10, subs=np.arange(1, 11))
        op_ax = ax.yaxis
    else:
        raise ValueError(f'which must be one of "x" or "y".')

    op_ax.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=0.1, subs=np.arange(1, 10))
    )
    ax_min_sign, ax_max_sign = np.sign(vals.min()), np.sign(vals.max())
    ax_min, ax_max = (
        np.ceil(np.log10(abs(vals.min()))),
        np.ceil(np.log10(abs(vals.max()))),
    )
    op_ax.set_view_interval(ax_min_sign * 10**ax_min, ax_max_sign * 10**ax_max)

    ticklabels = []
    ticks = []
    if ax_min_sign == -1:
        if ax_max_sign == -1:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{-10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, ax_max, int(ax_min - ax_max) + 1)
                ]
            )
            ticks.append(-np.logspace(ax_min, ax_max, int(ax_min - ax_max) + 1))
        else:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{-10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, 0, int(ax_min) + 1)
                ]
            )
            ticks.append(-np.logspace(ax_min, 0, int(ax_min) + 1))

    if ax_max_sign == 1:
        if ax_min_sign == 1:
            ticklabels.extend(
                [
                    f"$\\mathdefault{{10^{{{int(n)}}}}}$"
                    for n in np.linspace(ax_min, ax_max, int(ax_max - ax_min) + 1)
                ]
            )
            ticks.append(np.linspace(ax_min, ax_max, int(ax_max - ax_min) + 1))
        else:
            ticklabels.append("0")
            ticklabels.extend(
                [
                    f"$\\mathdefault{{10^{{{int(n)}}}}}$"
                    for n in np.linspace(0, ax_max, int(ax_max) + 1)
                ]
            )
            ticks.append(np.array([0]))
            ticks.append(np.logspace(0, ax_max, int(ax_max) + 1))

    op_ax.set_ticks(np.concatenate(ticks))
    op_ax.set_ticklabels(ticklabels)
    return ax
