"""Shared utilties for evaluating steps and visualizing data."""

import numpy as np
import matplotlib.pyplot as plt


def plot_plate_heatmap(df, metric=None, metadata=None, **kwargs):
    """Plot the heatmap of a summary DataFrame by well and tile in a plate layout.

    Tiles are plotted at their actual spatial positions from metadata using scatter.
    Wells are auto-detected from the data and arranged in a grid of subplots.

    Args:
        df (pandas.DataFrame):
            Summary DataFrame of values to plot, expects one row per (well, tile).
        metric (str, optional):
            Column of `df` to use for the heatmap. If None, infers from columns.
        metadata (pandas.DataFrame, optional):
            Metadata DataFrame with 'well', 'tile', 'x_pos', 'y_pos' columns.
            When provided, tiles are plotted at their spatial positions.
        **kwargs:
            Keyword arguments passed to matplotlib scatter.

    Returns:
        matplotlib.figure.Figure: The figure object.
        matplotlib.colorbar.Colorbar: Colorbar object for the plot.
    """
    # Merge spatial positions from metadata if provided
    if metadata is not None and "x_pos" in metadata.columns:
        pos = metadata.drop_duplicates(subset=["well", "tile"])[
            ["well", "tile", "x_pos", "y_pos"]
        ]
        df = df.merge(pos, on=["well", "tile"], how="left")

    # Infer metric to plot if necessary
    non_metric_cols = ["plate", "well", "tile", "x_pos", "y_pos"]
    if not metric:
        metric = [col for col in df.columns if col not in non_metric_cols]
        if len(metric) != 1:
            raise ValueError(
                "Cannot infer metric to plot, can pass metric column name "
                "explicitly to metric kwarg"
            )
        metric = metric[0]

    # Create well layout from data
    wells = sorted(df["well"].unique())
    if len(wells) == 1:
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes = np.array([axes])
    else:
        nr = nc = int(np.ceil(np.sqrt(len(wells))))
        if (nr - 1) * nc >= len(wells):
            nr -= 1
        fig, axes = plt.subplots(nr, nc, figsize=(15, 10))

    # Define colorbar min and max
    cmin, cmax = df[metric].min(), df[metric].max()
    if 0 <= cmin and cmax <= 1:
        cmin, cmax = 0, 1

    # Remove imshow-specific kwargs that don't apply to scatter
    scatter_kwargs = {
        k: v for k, v in kwargs.items() if k not in ["interpolation", "aspect"]
    }

    # Plot wells
    use_spatial = "x_pos" in df.columns and "y_pos" in df.columns
    plot = None
    for ax, well in zip(axes.reshape(-1), wells):
        df_well = df.query("well==@well")
        if len(df_well) > 0:
            if use_spatial:
                plot = ax.scatter(
                    df_well["x_pos"],
                    df_well["y_pos"],
                    c=df_well[metric],
                    vmin=cmin,
                    vmax=cmax,
                    s=50,
                    marker="s",
                    **scatter_kwargs,
                )
                ax.set_aspect("equal")
            else:
                # Fallback: plot tiles as a square grid by tile ID
                tiles = max(len(df["tile"].unique()), df["tile"].astype(int).max())
                r = c = int(np.ceil(np.sqrt(tiles)))
                grid = np.full(r * c, np.nan)
                grid[:tiles] = range(tiles)
                grid = grid.reshape(r, c)
                values = grid.copy()
                for tile in range(tiles):
                    try:
                        values[grid == tile] = df_well.loc[
                            df_well.tile == tile, metric
                        ].values[0]
                    except Exception:
                        values[grid == tile] = np.nan
                plot = ax.imshow(values, vmin=cmin, vmax=cmax, **kwargs)
        ax.set_title(f"Well {well}", fontsize=24)
        ax.axis("off")

    # Hide unused axes
    for ax in axes.reshape(-1)[len(wells) :]:
        ax.set_visible(False)

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    if plot is not None:
        cbar = fig.colorbar(plot, cax=cbar_ax)
        cbar.set_label(metric, fontsize=18)
        cbar_ax.yaxis.set_ticks_position("left")
    else:
        raise ValueError("No data to plot")

    return fig, cbar
