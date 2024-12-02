"""Helper functions for evaluating the feature extraction results of the phenotype process steps."""

from lib.shared.eval import plot_plate_heatmap


def plot_feature_heatmap(
    df,
    feature,
    tile="tile",
    shape="square",
    plate="6W",
    agg_func="median",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plot a heatmap of a specified feature in a DataFrame by well and tile in a convenient plate layout.

    Args:
        df (pandas.DataFrame): Input data containing the feature to be plotted.
        feature (str): The column name of the feature to be plotted.
        tile (str, optional): The column name used to group tiles. Defaults to 'tile'.
        shape (str, optional): Shape of subplot for each well used in the heatmap. Defaults to 'square'.
        plate (str): Plate type for the heatmap. Options are {'6W', '24W', '96W'}.
        agg_func (str | callable, optional): The aggregation function to use when grouping by well and tile.
            Options include 'mean', 'median', 'sum', or any function that can be passed to pandas' `agg()`. Defaults to 'mean'.
        return_plot (bool, optional): If True, returns figure. Defaults to True.
        return_summary (bool, optional): If True, returns `df_summary`. Defaults to False.
        **kwargs: Additional keyword arguments passed to `plot_plate_heatmap()`.

    Returns:
        pandas.DataFrame: Summary DataFrame used for plotting. Returned only if `return_summary=True`.
        np.ndarray: Array of matplotlib Axes objects.
    """
    # Group data by well and tile and aggregate the specified feature
    df_summary = df.groupby(["well", tile]).agg({feature: agg_func}).reset_index()

    if return_summary and return_plot:
        # Plot heatmap
        fig, _ = plot_plate_heatmap(
            df_summary, metric=feature, shape=shape, plate=plate, **kwargs
        )
        return df_summary, fig
    elif return_plot:
        # Plot heatmap
        fig, _ = plot_plate_heatmap(
            df_summary, metric=feature, shape=shape, plate=plate, **kwargs
        )
        return fig
    elif return_summary:
        return df_summary
    else:
        return None
