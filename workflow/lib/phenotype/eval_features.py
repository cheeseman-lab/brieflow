"""Helper functions for evaluating the feature extraction results of the phenotype process steps."""

from lib.shared.eval import plot_plate_heatmap


def plot_feature_heatmap(
    df,
    feature,
    tile="tile",
    metadata=None,
    agg_func="median",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plot a heatmap of a specified feature in a DataFrame by well and tile.

    Args:
        df (pandas.DataFrame): Input data containing the feature to be plotted.
        feature (str): The column name of the feature to be plotted.
        tile (str, optional): The column name used to group tiles. Defaults to 'tile'.
        metadata (pandas.DataFrame, optional): Metadata with x_pos/y_pos for spatial plotting.
        agg_func (str | callable, optional): Aggregation function. Defaults to 'median'.
        return_plot (bool, optional): If True, returns figure. Defaults to True.
        return_summary (bool, optional): If True, returns `df_summary`. Defaults to False.
        **kwargs: Additional keyword arguments passed to `plot_plate_heatmap()`.

    Returns:
        pandas.DataFrame: Summary DataFrame. Returned only if `return_summary=True`.
        matplotlib.figure.Figure: The figure object.
    """
    # Group data by well and tile and aggregate the specified feature
    df_summary = df.groupby(["well", tile]).agg({feature: agg_func}).reset_index()

    if return_summary and return_plot:
        # Plot heatmap
        fig, _ = plot_plate_heatmap(
            df_summary, metric=feature, metadata=metadata, **kwargs
        )
        return df_summary, fig
    elif return_plot:
        # Plot heatmap
        fig, _ = plot_plate_heatmap(
            df_summary, metric=feature, metadata=metadata, **kwargs
        )
        return fig
    elif return_summary:
        return df_summary
    else:
        return None
