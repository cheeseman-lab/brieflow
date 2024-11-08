"""Helper functions for evaluating the results of the SBS process steps."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from lib.shared.eval import plot_plate_heatmap


# helper function to load and concatenate hdfs
# helper function to load and concatenate HDFs
def load_and_concatenate_hdfs(hdf_files):
    """Load and concatenate HDF files from a provided list into a single DataFrame.

    Args:
        hdf_files (list of str): List of paths to HDF files.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
    """
    # Load each HDF file into a pandas DataFrame
    dfs = [pd.read_hdf(file) for file in hdf_files]

    # Concatenate all DataFrames into a single DataFrame
    concatenated_df = pd.concat(dfs, ignore_index=True)

    return concatenated_df


def plot_mapping_vs_threshold(
    df_reads, barcodes, threshold_var="peak", ax=None, **kwargs
):
    """Plot the mapping rate and number of mapped spots against varying thresholds of peak intensity, quality score, or a user-defined metric.

    Args:
        df_reads (pandas.DataFrame):
            Table of extracted reads from Snake.call_reads(). Can be concatenated results from
            multiple tiles, wells, etc.
        barcodes (list or set of str):
            Expected barcodes from the pool library design.
        threshold_var (str, optional):
            Variable to apply varying thresholds to for comparing mapping rates. Standard variables are
            'peak' and 'QC_min'. Can also use a user-defined variable, but must be a column of the df_reads
            table. Defaults to 'peak'.
        ax (matplotlib.axis, optional):
            Optional. If not None, this is an axis object to plot on. Helpful when plotting on
            a subplot of a larger figure. Defaults to None.
        **kwargs:
            Keyword arguments passed to sns.lineplot()

    Returns:
        pandas.DataFrame: Summary table of thresholds and associated mapping rates, number of spots mapped
        used for plotting.
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Exclude spots not in cells
    df_passed = df_reads.copy().query("cell > 0")

    # Map reads
    df_passed.loc[:, "mapped"] = df_passed["barcode"].isin(barcodes)

    # Define thresholds range
    if df_reads[threshold_var].max() < 100:
        thresholds = (
            np.array(
                range(0, int(np.quantile(df_passed[threshold_var], q=0.99) * 1000))
            )
            / 1000
        )
    else:
        thresholds = list(
            range(0, int(np.quantile(df_passed[threshold_var], q=0.99)), 10)
        )

    # Iterate over thresholds
    mapping_rate = []
    spots_mapped = []
    for threshold in thresholds:
        df_thresholded = df_passed.query(f"{threshold_var} > @threshold")
        spots_mapped.append(df_thresholded[df_thresholded["mapped"]].shape[0])
        mapping_rate.append(
            df_thresholded[df_thresholded["mapped"]].shape[0] / df_thresholded.shape[0]
        )

    # Create DataFrame for summary
    df_summary = pd.DataFrame(
        {
            f"{threshold_var}_threshold": thresholds,
            "mapping_rate": mapping_rate,
            "mapped_spots": spots_mapped,
        }
    )

    # Create a new figure if ax is not provided
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    # Plot on the main axis
    sns.lineplot(
        data=df_summary,
        x=f"{threshold_var}_threshold",
        y="mapping_rate",
        ax=ax,
        **kwargs,
    )
    ax.set_ylabel("mapping rate", fontsize=18)
    ax.set_xlabel(f"{threshold_var} threshold", fontsize=18)

    # Plot on the secondary axis
    ax_right = ax.twinx()
    sns.lineplot(
        data=df_summary,
        x=f"{threshold_var}_threshold",
        y="mapped_spots",
        ax=ax_right,
        color="coral",
        **kwargs,
    )
    ax_right.set_ylabel("mapped spots", fontsize=18)
    ax_right.legend(["mapped spots"], loc="upper right")

    # Main axis legend
    ax.legend(["mapping rate"], loc="upper left")

    return df_summary, fig


def plot_read_mapping_heatmap(
    df_reads,
    barcodes,
    shape="square",
    plate="6W",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plot the mapping rate of reads by well and tile in a convenient plate layout.

    Args:
        df_reads (pandas.DataFrame):
            DataFrame of all reads output from sbs mapping pipeline, e.g., concatenated outputs for all tiles
            and wells of Snake.call_reads().
        barcodes (list or set of str):
            Expected barcodes from the pool library design.
        shape (str, optional):
            Shape of subplot for each well used in plot_plate_heatmap. Defaults to 'square'.
        plate (str):
            Plate type for plot_plate_heatmap. Options are {'6W', '24W', '96W'}.
        return_plot (bool, optional):
            If true, returns df_summary. Defaults to True.
        return_summary (bool, optional):
            If true, returns df_summary. Defaults to False.
        **kwargs:
            Keyword arguments passed to plot_plate_heatmap().

    Returns:
        pandas.DataFrame: DataFrame used for plotting, optional output, only returns if return_summary=True.
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Mark reads as mapped or unmapped based on provided barcodes
    df_reads.loc[:, "mapped"] = df_reads["barcode"].isin(barcodes)

    # Calculate mapping rates by well and tile
    df_summary = (
        df_reads.groupby(["well", "tile"])["mapped"]
        .value_counts(normalize=True)
        .rename("fraction of reads mapping")
        .to_frame()
        .reset_index()
        .query("mapped")
        .drop(columns="mapped")
    )

    if return_summary and return_plot:
        # Plot heatmap
        fig, _ = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return df_summary, fig
    elif return_plot:
        # Plot heatmap
        fig, _ = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return fig
    elif return_summary:
        return df_summary
    else:
        return None


def plot_cell_mapping_heatmap(
    df_cells,
    df_sbs_info,
    barcodes,
    mapping_to="one",
    mapping_strategy="barcodes",
    shape="square",
    plate="6W",
    return_plot=True,
    return_summary=False,
    **kwargs,
):
    """Plot the mapping rate of cells by well and tile in a convenient plate layout.

    Args:
        df_cells (pandas.DataFrame):
            DataFrame of all cells output from sbs mapping pipeline, e.g., concatenated outputs for all tiles and wells
            of Snake.call_cells().
        df_sbs_info (pandas.DataFrame):
            DataFrame of all cells segmented from sbs images, e.g., concatenated outputs for all tiles and wells of
            Snake.extract_phenotype_minimal(data_phenotype=nuclei, nuclei=nuclei), often used as sbs_cell_info rule in
            Snakemake.
        barcodes (list or set of str):
            Expected barcodes from the pool library design.
        mapping_to (str):
            Cells to include as 'mapped'. 'one' only includes cells mapping to a single barcode, 'any' includes cells
            mapping to at least 1 barcode. Options are {'one', 'any'}.
        mapping_strategy (str): Strategy to use for mapping cells. Options are {'barcodes', 'gene_symbols'}.
        shape (str, optional):
            Shape of subplot for each well used in plot_plate_heatmap. Defaults to 'square'.
        plate (str):
            Plate type for plot_plate_heatmap. Options are {'6W', '24W', '96W'}.
        return_plot (bool, optional):
            If true, returns df_summary. Defaults to True.
        return_summary (bool, optional):
            If true, returns df_summary. Defaults to False.
        **kwargs:
            Keyword arguments passed to plot_plate_heatmap().

    Returns:
        pandas.DataFrame: DataFrame used for plotting, optional output, only returns if return_summary=True.
        np.array: Array of matplotlib Axes objects.
    """
    # Mark cells as mapped or unmapped based on provided barcodes or gene symbols
    if mapping_strategy == "barcodes":
        df_cells.loc[:, ["mapped_0", "mapped_1"]] = (
            df_cells[["cell_barcode_0", "cell_barcode_1"]].isin(barcodes).values
        )
    elif mapping_strategy == "gene_symbols":
        df_cells["mapped_0"] = (~df_cells["gene_symbol_0"].isna()).astype(int)
        df_cells["mapped_1"] = (~df_cells["gene_symbol_1"].isna()).astype(int)
    else:
        raise ValueError(
            f"Invalid mapping strategy: {mapping_strategy}. Choose 'barcodes' or 'gene_symbols'."
        )

    # Merge cell mapping information with sbs info
    df = df_sbs_info[["well", "tile", "cell"]].merge(
        df_cells[["well", "tile", "cell", "mapped_0", "mapped_1"]],
        how="left",
        on=["well", "tile", "cell"],
    )

    # Determine mapping criteria and calculate mapping rates
    if mapping_to == "one":
        metric = "fraction of cells mapping to 1 barcode"
        df = df.assign(mapped=lambda x: x[["mapped_0", "mapped_1"]].sum(axis=1) == 1)
    elif mapping_to == "any":
        metric = "fraction of cells mapping to >=1 barcode"
        df = df.assign(mapped=lambda x: x[["mapped_0", "mapped_1"]].sum(axis=1) > 0)
    else:
        raise ValueError(f"mapping_to={mapping_to} not implemented")

    # Calculate mapping rates by well and tile
    df_summary = (
        df.groupby(["well", "tile"])["mapped"]
        .value_counts(normalize=True)
        .rename(metric)
        .to_frame()
        .reset_index()
        .query("mapped")
        .drop(columns="mapped")
    )

    if return_summary and return_plot:
        # Plot heatmap
        fig = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return df_summary, axes
    elif return_plot:
        # Plot heatmap
        axes = plot_plate_heatmap(df_summary, shape=shape, plate=plate, **kwargs)
        return axes
    elif return_summary:
        return df_summary
    else:
        return None


def plot_reads_per_cell_histogram(df, x_cutoff=40):
    """Plot a histogram of the number of reads per cell.

    Args:
        df (pandas.DataFrame):
            DataFrame containing the data with columns including 'barcode_count' representing the number of reads
            per cell.
        x_cutoff (int, optional):
            Cutoff value for the x-axis. Defaults to 20.
        bins (int, optional):
            Number of bins for the histogram. Defaults to 40.

    Returns:
        pandas.Series: Series containing outlier values exceeding the x_cutoff.
    """
    plt.figure(figsize=(12, 7))
    sns.set_style("white")

    # Create bins from 0 to x_cutoff (inclusive)
    bins = range(x_cutoff + 1)

    # Plot the histogram
    sns.histplot(
        data=df, x="barcode_count", bins=bins, color="skyblue", edgecolor="black"
    )

    # Set title and axis labels
    plt.title("Histogram of Barcode Count", fontsize=16, fontweight="bold")
    plt.xlabel("Number of ISS reads per cell", fontsize=12)
    plt.ylabel("Number of cells", fontsize=12)

    # Find outlier values
    outliers = df[df["barcode_count"] > x_cutoff]["barcode_count"]

    # Restrict x-axis to stop at x_cutoff and set integer ticks
    plt.xlim(0, x_cutoff)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Format y-axis to use scientific notation
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Remove top and right spines
    sns.despine()

    # Adjust layout
    plt.tight_layout()

    return outliers


def plot_gene_symbol_histogram(df, x_cutoff=40):
    """Plot a histogram of the number of counts of each unique gene_symbol_0.

    Args:
        df (pandas.DataFrame):
            DataFrame containing the data with a column 'gene_symbol_0'.
        x_cutoff (int, optional):
            Cutoff value for the x-axis. Defaults to 40.
        bins (int, optional):
            Number of bins for the histogram. Defaults to 40.

    Returns:
        None
    """
    # Count occurrences of each unique gene_symbol_0
    gene_symbol_counts = df["gene_symbol_0"].value_counts()

    plt.figure(figsize=(12, 7))
    sns.set_style("white")

    # Set bin number
    if x_cutoff < 100:
        num_bins = x_cutoff
    else:
        num_bins = 100

    # Create 100 evenly spaced bins from 0 to x_cutoff
    bins = np.linspace(0, x_cutoff, num_bins + 1)  # 101 edges to create 100 bins

    # Plot the histogram
    sns.histplot(
        data=gene_symbol_counts, bins=bins, color="lightgreen", edgecolor="black"
    )

    # Set title and axis labels
    plt.title("Histogram of Gene Symbol Counts", fontsize=16, fontweight="bold")
    plt.xlabel("Number of cells per mapped gene", fontsize=12)
    plt.ylabel("Number of mapped genes", fontsize=12)

    outliers = gene_symbol_counts[gene_symbol_counts > x_cutoff]

    # Restrict x-axis to stop at x_cutoff and set integer ticks
    plt.xlim(0, x_cutoff)
    plt.gca().xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Format y-axis to use scientific notation
    plt.gca().yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    plt.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Remove top and right spines
    sns.despine()

    # Adjust layout
    plt.tight_layout()

    return outliers
