"""Helper functions for evaluating the results of the SBS process steps."""

import string

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


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
        df_passed = df_passed.query("{} > @threshold".format(threshold_var))
        spots_mapped.append(df_passed[df_passed["mapped"]].pipe(len))
        mapping_rate.append(
            df_passed[df_passed["mapped"]].pipe(len) / df_passed.pipe(len)
        )

    # Create DataFrame for summary
    df_summary = pd.DataFrame(
        np.array([thresholds, mapping_rate, spots_mapped]).T,
        columns=["{}_threshold".format(threshold_var), "mapping_rate", "mapped_spots"],
    )

    # Plot
    if not ax:
        ax = sns.lineplot(
            data=df_summary,
            x="{}_threshold".format(threshold_var),
            y="mapping_rate",
            **kwargs,
        )
    else:
        sns.lineplot(
            data=df_summary,
            x="{}_threshold".format(threshold_var),
            y="mapping_rate",
            ax=ax,
            **kwargs,
        )
    ax.set_ylabel("mapping rate", fontsize=18)
    ax.set_xlabel("{} threshold".format(threshold_var), fontsize=18)
    ax_right = ax.twinx()
    sns.lineplot(
        data=df_summary,
        x="{}_threshold".format(threshold_var),
        y="mapped_spots",
        ax=ax_right,
        color="coral",
        **kwargs,
    )
    ax_right.set_ylabel("mapped spots", fontsize=18)
    plt.legend(
        ax.get_lines() + ax_right.get_lines(), ["mapping rate", "mapped spots"], loc=7
    )

    return df_summary


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
        np.array: Array of matplotlib Axes objects.
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
            centered shape within a rectangle. Unused corners of this rectangle are plotted as NaN. The summation of
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
    tiles = max(len(df["tile"].unique()), df["tile"].max())

    # Define grid for plotting
    if shape == "square":
        r = c = int(np.ceil(np.sqrt(tiles)))
        grid = np.empty(r * c)
        grid[:] = np.NaN
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
        elif isinstance(shape, list):
            rows = shape
        else:
            raise ValueError(
                "{} shape not implemented, can pass custom shape as a"
                "list specifying number of sites per row".format(shape)
            )

        r, c = len(rows), max(rows)
        grid = np.empty((r, c))
        grid[:] = np.NaN

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
    cmin, cmax = (df[metric].min(), df[metric].max())

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

    return axes, cbar


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
