"""Helper functions for evaluating the results of the SBS process steps."""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib.lines import Line2D

from lib.shared.eval import plot_plate_heatmap


def plot_mapping_vs_threshold(
    df_reads, barcodes, threshold_var="peak", ax=None, num_thresholds=None, **kwargs
):
    """Plot the mapping rate and number of mapped spots against varying thresholds of peak intensity, quality score, or a user-defined metric.

    Args:
        df_reads (pandas.DataFrame):
            Table of extracted reads from call_reads. Can be concatenated results from
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
        num_thresholds (int, optional):
            Number of threshold points to evaluate. Controls the granularity of the curve.
            Lower values will run faster. If None, uses the original threshold generation logic.
            Defaults to None.
        **kwargs:
            Keyword arguments passed to sns.lineplot()

    Returns:
        pandas.DataFrame: Summary table of thresholds and associated mapping rates, for cell-associated reads only.
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Create figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Left plot (all reads)
    df_all = df_reads.copy()
    df_all.loc[:, "mapped"] = df_all["barcode"].isin(barcodes)
    # Right plot (cell-associated reads only)
    df_cells = df_reads.copy().query("cell > 0")
    df_cells.loc[:, "mapped"] = df_cells["barcode"].isin(barcodes)
    # Process data and create plots for both
    for df, ax, title in [
        (df_all, ax1, "All Reads"),
        (df_cells, ax2, "Cell-Associated Reads Only"),
    ]:
        if df_reads[threshold_var].max() < 100:
            thresholds = (
                np.array(range(0, int(np.quantile(df[threshold_var], q=0.99) * 1000)))
                / 1000
            )
        else:
            thresholds = list(range(0, int(np.quantile(df[threshold_var], q=0.99)), 10))

        if num_thresholds is not None:
            # Choose evenly spaced indices from the existing thresholds
            indices = np.linspace(0, len(thresholds) - 1, num_thresholds, dtype=int)
            thresholds = [thresholds[i] for i in indices]

        # Calculate metrics
        mapping_rate = []
        spots_mapped = []
        cells_mapped = []
        for threshold in thresholds:
            df_thresholded = df.query(f"{threshold_var} > @threshold")
            mapped_reads = df_thresholded[df_thresholded["mapped"]]
            spots_mapped.append(mapped_reads.shape[0])
            cells_mapped.append(len(mapped_reads.groupby(["well", "tile", "cell"])))
            mapping_rate.append(
                mapped_reads.shape[0] / df_thresholded.shape[0]
                if df_thresholded.shape[0] > 0
                else 0
            )

        # Create summary DataFrame
        df_summary = pd.DataFrame(
            {
                f"{threshold_var}_threshold": thresholds,
                "mapping_rate": mapping_rate,
                "mapped_spots": spots_mapped,
                "mapped_cells": cells_mapped,
            }
        )

        # Main axis plot
        sns.lineplot(
            data=df_summary,
            x=f"{threshold_var}_threshold",
            y="mapping_rate",
            ax=ax,
            color="dodgerblue",
            **kwargs,
        )

        # Secondary axis
        ax_right = ax.twinx()

        sns.lineplot(
            data=df_summary,
            x=f"{threshold_var}_threshold",
            y="mapped_spots",
            ax=ax_right,
            color="coral",
            **kwargs,
        )

        sns.lineplot(
            data=df_summary,
            x=f"{threshold_var}_threshold",
            y="mapped_cells",
            ax=ax_right,
            color="coral",
            linestyle=":",
            **kwargs,
        )

        # Labels and titles
        ax.set_ylabel(
            "Fraction of Reads\nMatching Expected Barcodes",
            fontsize=12,
            color="dodgerblue",
        )
        ax.set_xlabel(
            f"{threshold_var.replace('_', ' ').title()} Threshold Cutoff", fontsize=12
        )
        ax.set_title(f"Read Mapping Quality vs Threshold\n({title})", fontsize=14)
        ax_right.set_ylabel("Number of Mapped Features", fontsize=12, color="coral")

        # Color the y-axis tick labels to match
        ax.tick_params(axis="y", labelcolor="dodgerblue")
        ax_right.tick_params(axis="y", labelcolor="coral")

    # Create shared legend below plots
    legend_elements = [
        Line2D(
            [0],
            [0],
            color="dodgerblue",
            label="Mapping Rate:\nFraction of reads with valid barcodes",
        ),
        Line2D(
            [0],
            [0],
            color="coral",
            label="Total Mapped Spots:\nNumber of reads with valid barcodes",
        ),
        Line2D(
            [0],
            [0],
            color="coral",
            linestyle=":",
            label="Unique Mapped Cells:\nNumber of cells with ≥1 mapped read",
        ),
    ]

    # Add legend below the plots
    fig.legend(
        handles=legend_elements,
        loc="center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=9,
    )

    # Adjust layout to prevent overlap
    plt.tight_layout()
    # Adjust bottom margin to accommodate legend
    plt.subplots_adjust(bottom=0.15)

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
            and wells of call_reads.
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
            of call_cells.
        df_sbs_info (pandas.DataFrame):
            DataFrame of all cells segmented from sbs images, e.g., concatenated outputs for all tiles and wells of
            extract_phenotype_minimal(data_phenotype=nuclei, nuclei=nuclei), often used as sbs_cell_info rule in
            Snakemake.
        barcodes (list or set of str):
            Expected barcodes from the pool library design.
        mapping_to (str):
            Cells to include as 'mapped'. 'one' only includes cells mapping to a single barcode, 'any' includes cells
            mapping to at least 1 barcode. Options are {'one', 'any'}.
        mapping_strategy (str): Strategy to use for mapping cells. Options are {'barcodes', 'gene symbols'}.
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
    # Mark cells as mapped or unmapped based on provided barcodes or gene symbols
    if mapping_strategy == "barcodes":
        df_cells.loc[:, ["mapped_0", "mapped_1"]] = (
            df_cells[["cell_barcode_0", "cell_barcode_1"]].isin(barcodes).values
        )
    elif mapping_strategy == "gene symbols":
        df_cells["mapped_0"] = (~df_cells["gene_symbol_0"].isna()).astype(int)
        df_cells["mapped_1"] = (~df_cells["gene_symbol_1"].isna()).astype(int)
    else:
        raise ValueError(
            f"Invalid mapping strategy: {mapping_strategy}. Choose 'barcodes' or 'gene symbols'."
        )

    # Merge cell mapping information with sbs info
    df = df_sbs_info[["well", "tile", "cell"]].merge(
        df_cells[["well", "tile", "cell", "mapped_0", "mapped_1"]],
        how="left",
        on=["well", "tile", "cell"],
    )

    # Determine mapping criteria and calculate mapping rates
    if mapping_to == "one":
        metric = f"fraction of cells mapping to 1 {mapping_strategy}"
        df = df.assign(mapped=lambda x: x[["mapped_0", "mapped_1"]].sum(axis=1) == 1)
    elif mapping_to == "any":
        metric = f"fraction of cells mapping to >=1 {mapping_strategy}"
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


def plot_cell_metric_histogram(df, sort_by="count", x_cutoff=None):
    """Plot a histogram of cell metrics (reads per cell or peak intensity per cell).

    Args:
        df (pandas.DataFrame):
            DataFrame containing the data with columns for barcode counts or peak intensities.
        sort_by (str, optional):
            Type of metric to plot. 'count' uses barcode_count, 'peak' uses sum of peak intensities.
            Defaults to 'count'.
        x_cutoff (int, optional):
            Cutoff value for the x-axis. Defaults to 40.

    Returns:
        pandas.Series: Series containing outlier values exceeding the x_cutoff.
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Create the figure and axis
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_style("white")

    # Determine metric column and labels based on sort_by parameter
    if sort_by == "count":
        if "barcode_count" not in df.columns:
            raise ValueError(
                "DataFrame must contain 'barcode_count' column when sort_by='count'"
            )
        metric_col = "barcode_count"
        title = "Histogram of Barcode Count"
        xlabel = "Number of ISS reads per cell"
    elif sort_by == "peak":
        # Create combined peak intensity metric
        barcode_peak_0 = df.get("barcode_peak_0", 0).fillna(0)
        barcode_peak_1 = df.get("barcode_peak_1", 0).fillna(0)
        df_temp = df.copy()
        df_temp["peak_intensity_total"] = barcode_peak_0 + barcode_peak_1
        metric_col = "peak_intensity_total"
        title = "Histogram of Peak Intensity"
        xlabel = "Total peak intensity per cell"
        df = df_temp
    else:
        raise ValueError(f"sort_by must be 'count' or 'peak', got '{sort_by}'")

    # Auto-determine x_cutoff if not provided
    if x_cutoff is None:
        # Set cutoff to max of metric column
        x_cutoff = df[metric_col].max()

    # Create bins from 0 to x_cutoff (inclusive)
    bins = range(int(x_cutoff) + 1)

    # Plot the histogram
    color = "skyblue" if sort_by == "count" else "lightcoral"
    sns.histplot(
        data=df, x=metric_col, bins=bins, color=color, edgecolor="black", ax=ax
    )

    # Set title and axis labels
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Number of cells", fontsize=12)

    # Find outlier values
    outliers = df[df[metric_col] > x_cutoff][metric_col]

    # Restrict x-axis to stop at x_cutoff and set integer ticks
    ax.set_xlim(0, x_cutoff)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Format y-axis to use scientific notation
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Remove top and right spines
    sns.despine()

    # Adjust layout
    fig.tight_layout()

    return outliers, fig


def plot_gene_symbol_histogram(df, x_cutoff=None):
    """Plot a histogram of the number of counts of each unique gene_symbol_0.

    Args:
        df (pandas.DataFrame):
            DataFrame containing the data with a column 'gene_symbol_0'.
        x_cutoff (int, optional):
            Cutoff value for the x-axis. If None, will be calculated from data.

    Returns:
        pandas.Series: Series containing outlier gene symbols with counts exceeding x_cutoff.
        matplotlib.figure.Figure: The figure object containing the histogram plot.
    """
    # Count occurrences of each unique gene_symbol_0
    gene_symbol_counts = df["gene_symbol_0"].value_counts()

    # Create figure and axis for the plot
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.set_style("white")

    # Auto-determine x_cutoff if not provided
    if x_cutoff is None:
        # Use IQR method to determine cutoff
        q1, q3 = gene_symbol_counts.quantile(0.25), gene_symbol_counts.quantile(0.75)
        iqr = q3 - q1
        x_cutoff = q3 + 1.5 * iqr

    # Always create 100 evenly spaced bins from 0 to x_cutoff
    bins = np.linspace(0, x_cutoff, 101)  # 101 edges to create 100 bins

    # Plot the histogram
    sns.histplot(
        data=gene_symbol_counts, bins=bins, color="lightgreen", edgecolor="black", ax=ax
    )

    # Set title and axis labels
    ax.set_title("Histogram of Gene Symbol Counts", fontsize=16, fontweight="bold")
    ax.set_xlabel("Number of cells per mapped gene", fontsize=12)
    ax.set_ylabel("Number of mapped genes", fontsize=12)

    # Identify outlier values
    outliers = gene_symbol_counts[gene_symbol_counts > x_cutoff]

    # Restrict x-axis to stop at x_cutoff and set integer ticks
    ax.set_xlim(0, x_cutoff)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))

    # Format y-axis to use scientific notation
    ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
    ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

    # Remove top and right spines
    sns.despine()

    # Adjust layout
    fig.tight_layout()

    return outliers, fig


def plot_barcode_prefix_matching(
    df_reads, df_library, barcode_col="barcode", library_col=None
):
    """Plot barcode prefix matching rate vs random expectation.

    Compares how well read barcodes match the library at each prefix length.
    A meaningful signal shows the actual match rate staying well above the
    random expectation (1/4^n), indicating true barcode detection rather than noise.

    Args:
        df_reads (pandas.DataFrame):
            DataFrame of reads containing barcode sequences.
        df_library (pandas.DataFrame):
            DataFrame of library barcodes to match against.
        barcode_col (str, optional):
            Column name in df_reads containing barcode sequences. Defaults to "barcode".
        library_col (str, optional):
            Column name in df_library containing barcode sequences.
            If None, tries common column names: 'barcode', 'sgRNA', 'iBAR_2'.
            Defaults to None.

    Returns:
        pandas.DataFrame: Summary table with columns 'prefix_length', 'match_rate', 'random_rate'.
        matplotlib.figure.Figure: The figure object containing the plot.
    """
    # Determine library column
    if library_col is None:
        for col in ["barcode", "sgRNA", "iBAR_2"]:
            if col in df_library.columns:
                library_col = col
                break
        if library_col is None:
            raise ValueError(
                "Could not find barcode column in library. "
                "Please specify library_col parameter."
            )

    # Get barcode length from reads
    barcode_length = len(df_reads[barcode_col].iloc[0])

    # Calculate match rates for each prefix length
    match_rates = []
    random_rates = []
    for i in range(1, barcode_length + 1):
        # Get prefix matches
        read_prefixes = df_reads[barcode_col].str[:i]
        library_prefixes = set(df_library[library_col].str[:i])
        match_rate = read_prefixes.isin(library_prefixes).mean()
        match_rates.append(round(match_rate, 4))
        random_rates.append(1 / (4**i))

    # Create summary DataFrame
    df_summary = pd.DataFrame(
        {
            "prefix_length": range(1, barcode_length + 1),
            "match_rate": match_rates,
            "random_rate": random_rates,
        }
    )

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("white")

    # Plot actual match rate
    sns.scatterplot(
        data=df_summary,
        x="prefix_length",
        y="match_rate",
        color="dodgerblue",
        s=60,
        label="Actual match rate",
        ax=ax,
    )
    sns.lineplot(
        data=df_summary,
        x="prefix_length",
        y="match_rate",
        color="dodgerblue",
        ax=ax,
    )

    # Plot random expectation
    sns.scatterplot(
        data=df_summary,
        x="prefix_length",
        y="random_rate",
        color="gray",
        s=60,
        label="Random expectation (1/4ⁿ)",
        ax=ax,
    )
    sns.lineplot(
        data=df_summary,
        x="prefix_length",
        y="random_rate",
        color="gray",
        linestyle="--",
        ax=ax,
    )

    # Formatting
    ax.set_xlabel("Barcode Prefix Length (bases)", fontsize=12)
    ax.set_ylabel("Fraction of Reads Matching Library", fontsize=12)
    ax.set_title("Barcode Prefix Matching: Actual vs Random", fontsize=14)
    ax.set_xticks(range(1, barcode_length + 1))
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper right")
    sns.despine()
    fig.tight_layout()

    return df_summary, fig


def mapping_overview(sbs_info, cells, sort_by="count"):
    """Generate an overview of cell counts and mapping statistics per well.

    This function calculates the total number of cells per well and determines the counts and
    percentages of cells with specific barcode and gene symbol mappings. It returns a summary
    DataFrame with counts and percentages for each well, detailing:
        - Total cells
        - Cells with exactly 1 barcode mapping
        - Cells with 1 or more barcode mappings
        - Cells with exactly 1 gene symbol mapping
        - Cells with 1 or more gene symbol mappings

    Args:
        sbs_info (pandas.DataFrame): DataFrame with information on cells, including the 'well' column.
        cells (pandas.DataFrame): DataFrame containing cell data with 'well', 'barcode_count', 'gene_symbol_0',
                                  'gene_symbol_1', 'cell_barcode_0', and 'cell_barcode_1' columns.
        sort_by (str, optional): Type of metric to analyze. 'count' uses barcode_count, 'peak' uses
                                presence of valid barcode values. Defaults to 'count'.

    Returns:
        pandas.DataFrame: A summary DataFrame with mapping counts and percentages per well.
    """
    # Count the total number of cells per well
    cell_counts = sbs_info.groupby("well").size().reset_index(name="total_cells__count")

    # For both count and peak modes, use the actual barcode columns to assess mapping
    # Check if cell_barcode_0 and cell_barcode_1 are valid (not NaN and not empty)
    cells_temp = cells.copy()
    cells_temp["has_barcode_0"] = (~cells_temp["cell_barcode_0"].isna()) & (
        cells_temp["cell_barcode_0"] != ""
    )
    cells_temp["has_barcode_1"] = (~cells_temp["cell_barcode_1"].isna()) & (
        cells_temp["cell_barcode_1"] != ""
    )
    cells_temp["barcode_mapping_count"] = cells_temp["has_barcode_0"].astype(
        int
    ) + cells_temp["has_barcode_1"].astype(int)

    if sort_by == "count":
        # Use the existing barcode_count column for count mode
        one_barcode_mapping = (
            cells[cells["barcode_count"] == 1]
            .groupby("well")
            .size()
            .reset_index(name="1_barcode_cells__count")
        )
        multiple_barcode_mapping = (
            cells[cells["barcode_count"] >= 1]
            .groupby("well")
            .size()
            .reset_index(name="1_or_more_barcodes__count")
        )
    elif sort_by == "peak":
        # Use the presence of valid barcodes for peak mode
        one_barcode_mapping = (
            cells_temp[cells_temp["barcode_mapping_count"] == 1]
            .groupby("well")
            .size()
            .reset_index(name="1_barcode_cells__count")
        )
        multiple_barcode_mapping = (
            cells_temp[cells_temp["barcode_mapping_count"] >= 1]
            .groupby("well")
            .size()
            .reset_index(name="1_or_more_barcodes__count")
        )
    else:
        raise ValueError(f"sort_by must be 'count' or 'peak', got '{sort_by}'")

    # Calculate percentages
    one_barcode_mapping["1_barcode_cells__percent"] = (
        one_barcode_mapping["1_barcode_cells__count"]
        / cell_counts["total_cells__count"]
        * 100
    )
    multiple_barcode_mapping["1_or_more_barcodes__percent"] = (
        multiple_barcode_mapping["1_or_more_barcodes__count"]
        / cell_counts["total_cells__count"]
        * 100
    )

    # Count and calculate percent of cells with 1 gene symbol mapping per well
    one_gene_mapping = (
        cells[(~cells["gene_symbol_0"].isna()) & (cells["gene_symbol_1"].isna())]
        .groupby("well")
        .size()
        .reset_index(name="1_gene_cells__count")
    )
    one_gene_mapping["1_gene_cells__percent"] = (
        one_gene_mapping["1_gene_cells__count"]
        / cell_counts["total_cells__count"]
        * 100
    )

    # Count and calculate percent of cells with >=1 gene symbol mapping per well
    multiple_gene_mapping = (
        cells[(~cells["gene_symbol_0"].isna()) | (~cells["gene_symbol_1"].isna())]
        .groupby("well")
        .size()
        .reset_index(name="1_or_more_genes__count")
    )
    multiple_gene_mapping["1_or_more_genes__percent"] = (
        multiple_gene_mapping["1_or_more_genes__count"]
        / cell_counts["total_cells__count"]
        * 100
    )

    # Merge all counts and percents into a single DataFrame
    mapping_overview_df = (
        cell_counts.merge(one_barcode_mapping, on="well", how="left")
        .merge(multiple_barcode_mapping, on="well", how="left")
        .merge(one_gene_mapping, on="well", how="left")
        .merge(multiple_gene_mapping, on="well", how="left")
    )

    # Fill NaN values with 0 (for cases where no cells meet criteria)
    mapping_overview_df.fillna(0, inplace=True)

    return mapping_overview_df
