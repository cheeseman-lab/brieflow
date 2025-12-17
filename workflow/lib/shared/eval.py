"""Shared utilties for evaluating steps and visualizing data."""

import string

import numpy as np
import matplotlib.pyplot as plt


def plot_plate_heatmap(
    df, metric=None, shape="square", plate="6W", snake_sites=True, **kwargs
):
    """Plot the heatmap of a summary DataFrame by well and tile in a convenient plate layout.

    Args:
        df (pandas.DataFrame):
            Summary DataFrame of values to plot, expects one row for each (well, tile) combination.
        metric (str, optional):
            Column of `df` to use for plotting the heatmap. If None, attempts to infer based on column names.
            Defaults to None.
        shape (str or list, optional):
            Shape of subplot for each well. Options are {'square', '6W_ph', '6W_sbs', list}. 'square' infers
            dimensions of the smallest square that fits the number of sites. '6W_ph' and '6W_sbs' use a common
            6 well tile map from a Nikon Ti2/Elements set-up with 20X and 10X objectives, respectively. Alternatively,
            a list can be passed containing the number of sites in each row of a tile layout. This is mapped into a
            centered shape within a rectangle. Unused corners of this rectangle are plotted as nan. The summation of
            this list should equal the total number of sites. Defaults to 'square'.
        plate (str):
            Plate type for plot_plate_heatmap. Options are {'6W', '24W', '96W'}.
        snake_sites (bool, optional):
            If true, plots tiles in a snake order similar to the order of sites acquired by many high throughput
            microscope systems. Defaults to True.
        **kwargs:
            Keyword arguments passed to matplotlib.pyplot.imshow().

    Returns:
        np.array: Array of matplotlib Axes objects for the plot.
        matplotlib.Colorbar: Colorbar object for the plot.
    """
    tiles = df["tile"].astype(int)
    tiles = max(len(tiles.unique()), tiles.max())

    # Define grid for plotting
    if shape == "square":
        r = c = int(np.ceil(np.sqrt(tiles)))
        grid = np.empty(r * c)
        grid[:] = np.nan
        grid[:tiles] = range(tiles)
        grid = grid.reshape(r, c)
    else:
        if shape == "6W_ph":
            rows = [
                7,
                13,
                17,
                21,
                25,
                27,
                29,
                31,
                33,
                33,
                35,
                35,
                37,
                37,
                39,
                39,
                39,
                41,
                41,
                41,
                41,
                41,
                41,
                41,
                39,
                39,
                39,
                37,
                37,
                35,
                35,
                33,
                33,
                31,
                29,
                27,
                25,
                21,
                17,
                13,
                7,
            ]
        elif shape == "6W_sbs":
            rows = [
                5,
                9,
                13,
                15,
                17,
                17,
                19,
                19,
                21,
                21,
                21,
                21,
                21,
                19,
                19,
                17,
                17,
                15,
                13,
                9,
                5,
            ]
        elif shape == "squid_ph":
            # Spatially accurate layout for Squid microscope phenotype tiles (1732 tiles)
            rows = [
                4,
                14,
                18,
                22,
                26,
                28,
                32,
                34,
                36,
                36,
                38,
                40,
                40,
                42,
                42,
                44,
                44,
                46,
                46,
                46,
                46,
                46,
                48,
                48,
                48,
                48,
                46,
                46,
                46,
                46,
                46,
                44,
                44,
                42,
                42,
                40,
                40,
                38,
                36,
                36,
                34,
                32,
                28,
                26,
                22,
                18,
                14,
                4,
            ]
        elif shape == "squid_sbs":
            # Spatially accurate layout for Squid microscope SBS tiles (94 tiles)
            rows = [2, 6, 8, 10, 10, 11, 11, 10, 10, 8, 6, 2]
        elif isinstance(shape, list):
            rows = shape
        else:
            raise ValueError(
                "{} shape not implemented, can pass custom shape as a"
                "list specifying number of sites per row".format(shape)
            )

        r, c = len(rows), max(rows)
        grid = np.empty((r, c))
        grid[:] = np.nan

        next_site = 0
        for row, row_sites in enumerate(rows):
            start = int((c - row_sites) / 2)
            grid[row, start : start + row_sites] = range(
                next_site, next_site + row_sites
            )
            next_site += row_sites

    if snake_sites:
        grid[1::2] = grid[1::2, ::-1]

    # Infer metric to plot if necessary
    if not metric:
        metric = [col for col in df.columns if col not in ["plate", "well", "tile"]]
        if len(metric) != 1:
            raise ValueError(
                "Cannot infer metric to plot, can pass metric column name explicitly to metric kwarg"
            )
        metric = metric[0]

    # Define subplots layout
    if df["well"].nunique() == 1:
        wells = df["well"].unique()
        fig, axes = plt.subplots(1, 1, figsize=(10, 10))
        axes = np.array([axes])
    elif plate == "6W":
        wells = [f"{r}{c}" for r in string.ascii_uppercase[:2] for c in range(1, 4)]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    elif plate == "24W":
        wells = [f"{r}{c}" for r in string.ascii_uppercase[:4] for c in range(1, 7)]
        fig, axes = plt.subplots(4, 6, figsize=(15, 10))
    elif plate == "96W":
        wells = [f"{r}{c}" for r in string.ascii_uppercase[:8] for c in range(1, 13)]
        fig, axes = plt.subplots(8, 12, figsize=(15, 10))
    else:
        wells = sorted(df["well"].unique())
        nr = nc = int(np.ceil(np.sqrt(len(wells))))
        if (nr - 1) * nc >= len(wells):
            nr -= 1
        fig, axes = plt.subplots(nr, nc, figsize=(15, 15))

    # Define colorbar min and max
    cmin, cmax = df[metric].min(), df[metric].max()
    if 0 <= cmin and cmax <= 1:
        cmin, cmax = 0, 1

    # Plot wells
    for ax, well in zip(axes.reshape(-1), wells):
        values = grid.copy()
        df_well = df.query("well==@well")
        if df_well.pipe(len) > 0:
            for tile in range(tiles):
                try:
                    values[grid == tile] = df_well.loc[
                        df_well.tile == tile, metric
                    ].values[0]
                except:
                    values[grid == tile] = np.nan
            plot = ax.imshow(values, vmin=cmin, vmax=cmax, **kwargs)
        ax.set_title("Well {}".format(well), fontsize=24)
        ax.axis("off")

    fig.subplots_adjust(right=0.9)
    cbar_ax = fig.add_axes([0.95, 0.15, 0.025, 0.7])
    try:
        cbar = fig.colorbar(plot, cax=cbar_ax)
    except:
        # Plot variable empty, no data plotted
        raise ValueError("No data to plot")
    cbar.set_label(metric, fontsize=18)
    cbar_ax.yaxis.set_ticks_position("left")

    return fig, cbar
