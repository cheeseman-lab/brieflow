"""Shared utilties for evaluating steps and visualizing data."""

import numpy as np
import matplotlib.pyplot as plt

from lib.shared.file_utils import split_well


def plot_plate_heatmap(df, metric=None, metadata=None, **kwargs):
    """Plot the heatmap of a summary DataFrame by well and tile in a plate layout.

    Tiles are plotted at their actual spatial positions from metadata using scatter.
    Wells are auto-detected from the data and placed in a grid of subplots by their
    plate row and column (e.g. "A1", "A01" or Opera Phenix "r01c01"), so any plate
    format keeps its shape; only the rows and columns present in the data are drawn.

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
        fig, axes = plt.subplots(1, 1, figsize=(10, 10), squeeze=False)
        positions = {wells[0]: (0, 0)}
        title_size = 24
    else:
        positions, nr, nc = _plate_layout(wells)
        panel = min(5.0, max(1.0, 15 / nc))
        fig, axes = plt.subplots(
            nr, nc, figsize=(panel * nc, panel * nr), squeeze=False
        )
        title_size = min(24, 6 * panel)

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
    for well in wells:
        ax = axes[positions[well]]
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
        ax.set_title(f"Well {well}", fontsize=title_size)
        ax.axis("off")

    # Hide unused axes
    used = set(positions.values())
    for position, ax in np.ndenumerate(axes):
        if position not in used:
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


def save_figure(fig, path, transparent=False, dpi=300):
    """Save an evaluation figure as a PNG with a white or transparent background.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        path (str or pathlib.Path): Output path.
        transparent (bool, optional): Save with a transparent background (e.g. for
            import into Illustrator) instead of white. Defaults to False.
        dpi (int, optional): Resolution in dots per inch. Defaults to 300.
    """
    fig.savefig(
        path,
        dpi=dpi,
        bbox_inches="tight",
        transparent=transparent,
        facecolor="none" if transparent else "white",
    )


def _plate_layout(wells):
    """Map each well to a (row, col) subplot position by its plate row and column.

    Rows and columns absent from the data are skipped. Wells that do not parse as
    plate positions fall back to a near-square grid in sorted order.

    Args:
        wells (list): Well identifiers, e.g. "A1", "B03" or "r02c05".

    Returns:
        tuple: Dict of well to (row, col), number of rows, number of columns.
    """
    try:
        parsed = {well: _well_row_col(well) for well in wells}
    except ValueError:
        nc = int(np.ceil(np.sqrt(len(wells))))
        nr = int(np.ceil(len(wells) / nc))
        positions = {well: divmod(i, nc) for i, well in enumerate(sorted(wells))}
        return positions, nr, nc
    rows = sorted({r for r, _ in parsed.values()})
    cols = sorted({c for _, c in parsed.values()})
    positions = {
        well: (rows.index(r), cols.index(c)) for well, (r, c) in parsed.items()
    }
    return positions, len(rows), len(cols)


def _well_row_col(well):
    """Convert a well id to 1-based plate (row, column) numbers.

    Args:
        well (str): Well id, e.g. "A1", "A01", "AB12" or Opera Phenix "r02c05".

    Returns:
        tuple[int, int]: Row and column numbers, e.g. "B03" -> (2, 3).
    """
    row, col = split_well(well)
    if row[0] in "Rr" and row[1:].isdigit():
        row_num = int(row[1:])
    else:
        row_num = 0
        for letter in row.upper():
            row_num = row_num * 26 + ord(letter) - ord("A") + 1
    return row_num, int(col.lstrip("Cc"))
