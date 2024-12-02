"""Utility functions for evaluating phenotype mapping results."""

from lib.shared.eval import plot_plate_heatmap


def plot_count_heatmap(
    df,
    tile="tile",
    shape="square",
    plate="6W",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plot the count of items in df by well and tile in a convenient plate layout.

    Useful for evaluating cell and read counts across wells. The colorbar label can
    be modified with: axes[0,0].get_figure().axes[-1].set_ylabel(LABEL)

    Args:
        df (pandas.DataFrame): Input data.
        tile (str, optional): The column name to be used to group tiles, as sometimes 'site' is used.
            Defaults to 'tile'.
        shape (str, optional): Shape of subplot for each well used in plot_plate_heatmap.
            Defaults to 'square'.
        plate (str): Plate type for plot_plate_heatmap. Must be one of {'6W', '24W', '96W'}.
        return_plot (bool, optional): If true, returns figure. Defaults to True.
        return_summary (bool, optional): If true, returns df_summary. Defaults to False.
        **kwargs: Keyword arguments passed to plot_plate_heatmap().

    Returns:
        np.ndarray: Array of matplotlib Axes objects, returned only if return_plot=True.
        pandas.DataFrame: DataFrame used for plotting, returned only if return_summary=True.
    """
    # Group data by well and tile and count the occurrences
    df_summary = (
        df.groupby(["well", tile]).size().rename("count").to_frame().reset_index()
    )

    if return_summary and return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return df_summary, axes
    elif return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return axes
    elif return_summary:
        return df_summary
    else:
        return None
