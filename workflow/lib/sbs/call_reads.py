"""Read Calling Utilities!

This module provides a set of functions for mapping and analyzing sequencing reads.
It includes functions for:

- Read Calling: Algorithms for calling bases from raw intensity data.
- Read Formatting: Tools to format and normalize sequencing reads.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# constants for calling reads
from lib.sbs.constants import (
    WELL,
    TILE,
    CELL,
    READ,
    CHANNEL,
    CYCLE,
    BARCODE,
    INTENSITY,
)


def call_reads(
    bases_data,
    peaks_data=None,
    correction_only_in_cells=True,
    normalize_bases_first=True,
    method="median",
):
    """Call reads for in situ sequencing data.

    Call reads by compensating for channel cross-talk and calling the base
    with the highest corrected intensity for each cycle.

    Args:
    bases_data : pandas DataFrame
        Table of base intensity for all candidate reads, output of extract_bases.

    peaks_data : None or numpy array, default None
        Peaks/local maxima score for each pixel (output of find_peaks) to be included
        in the df_reads table for downstream QC or other analysis. If None, does not include
        peaks scores in returned df_reads table.

    correction_only_in_cells : boolean, default True
        If True, restricts correction/compensation step to account only for reads that
        are within a cell, as defined by the cell segmentation mask passed into
        extract_bases. Often identified spots outside of cells are not true sequencing
        reads.

    normalize_bases_first : boolean, default True
        If True, normalizes the base intensities before performing median correction.
        Only applies when method="median".

    method : str, default "median"
        Method to use for correction. Options are "median" or "percentile".
        - "median": Uses median-based correction, performed independently for each tile.
        - "percentile": Uses percentile-based correction, performed independently for each tile.

    Returns:
    df_reads : pandas DataFrame
        Table of all reads with base calls resulting from SBS compensation and related metadata.
    """
    if bases_data.empty:
        # Get number of cycles from the most common value in the cycle column
        # Default to 0 if no cycles are found
        cycles = len(set(bases_data["cycle"])) if not bases_data.empty else 0

        # Create base columns that are always present
        base_columns = [
            "read",
            "cell",
            "i",
            "j",
            "tile",
            "well",
            "barcode",
        ]

        # Dynamically generate Q-score columns based on number of cycles
        q_columns = [f"Q_{i}" for i in range(cycles)]

        # Add Q_min and peak columns
        final_columns = base_columns + q_columns + ["Q_min", "peak"]

        return pd.DataFrame(columns=final_columns)

    if correction_only_in_cells:
        if len(bases_data.query("cell > 0")) == 0:
            return

    cycles = len(set(bases_data["cycle"]))
    channels = len(set(bases_data["channel"]))

    # Choose the appropriate method for read calling
    if method == "median":
        if normalize_bases_first:
            # Clean up and normalize base intensities, then perform median calling
            df_reads = (
                bases_data.pipe(clean_up_bases)
                .pipe(normalize_bases)
                .pipe(
                    do_median_call,
                    cycles,
                    channels=channels,
                    correction_only_in_cells=correction_only_in_cells,
                )
            )
        else:
            # Clean up bases and perform median calling without normalization
            df_reads = bases_data.pipe(clean_up_bases).pipe(
                do_median_call,
                cycles,
                channels=channels,
                correction_only_in_cells=correction_only_in_cells,
            )
    elif method == "percentile":
        # Clean up bases and perform percentile calling
        df_reads = bases_data.pipe(clean_up_bases).pipe(
            do_percentile_call,
            cycles=cycles,
            channels=channels,
            correction_only_in_cells=correction_only_in_cells,
        )
    else:
        raise ValueError(f"Unknown method: {method}. Use 'median' or 'percentile'.")

    # Include peaks scores if available
    if peaks_data is not None:
        i, j = df_reads[["i", "j"]].values.T
        df_reads["peak"] = peaks_data[i, j]

    return df_reads


def clean_up_bases(df_bases):
    """Sort DataFrame df_bases for pre-processing before dataframe_to_values.

    Args:
        df_bases (pandas.DataFrame): DataFrame containing raw base signal intensities.

    Returns:
        pandas.DataFrame: Sorted DataFrame.
    """
    # Sort DataFrame based on multiple columns
    return df_bases.sort_values([WELL, TILE, CELL, READ, CYCLE, CHANNEL])


def do_median_call(
    df_bases,
    cycles=12,
    channels=4,
    correction_quartile=0,
    correction_only_in_cells=False,
    correction_by_cycle=False,
):
    """Call reads from raw base signal using median correction.

    Args:
        df_bases (pandas.DataFrame): DataFrame containing raw base signal intensities.
        cycles (int): Number of sequencing cycles.
        channels (int): Number of sequencing channels.
        correction_quartile (int): Quartile used for correction.
        correction_only_in_cells (bool): Flag specifying whether correction is based on reads within cells or all reads.
        correction_by_cycle (bool): Flag specifying if correction should be done by cycle.

    Returns:
        pandas.DataFrame: DataFrame containing the called reads.
    """

    def correction(df, channels, correction_quartile, correction_only_in_cells):
        # Define the correction function
        if correction_only_in_cells:
            # Obtain transformation matrix W based on reads within cells
            X_ = dataframe_to_values(df.query("cell > 0"))
            _, W = transform_medians(
                X_.reshape(-1, channels), correction_quartile=correction_quartile
            )
            # Apply transformation to all data
            X = dataframe_to_values(df)
            Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
        else:
            # Apply correction to all data
            X = dataframe_to_values(df)
            Y, W = transform_medians(
                X.reshape(-1, channels), correction_quartile=correction_quartile
            )
        return Y, W

    # Apply correction either by cycle or to the entire dataset
    if correction_by_cycle:
        # Apply correction cycle by cycle
        Y = np.empty(df_bases.pipe(len), dtype=df_bases.dtypes["intensity"]).reshape(
            -1, channels
        )
        for cycle, (_, df_cycle) in enumerate(df_bases.groupby("cycle")):
            Y[cycle::cycles, :], _ = correction(
                df_cycle, channels, correction_quartile, correction_only_in_cells
            )
    else:
        # Apply correction to the entire dataset
        Y, W = correction(
            df_bases, channels, correction_quartile, correction_only_in_cells
        )

    # Call barcodes
    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads


def do_percentile_call(
    df_bases,
    cycles=12,
    channels=4,
    correction_only_in_cells=False,
):
    """Call reads from raw base signal using percentile-based correction.

    Args:
        df_bases (pandas.DataFrame): DataFrame containing raw base signal intensities.
        cycles (int): Number of sequencing cycles.
        channels (int): Number of sequencing channels.
        correction_only_in_cells (bool): Flag specifying whether correction is based on reads within cells or all reads.

    Returns:
        pandas.DataFrame: DataFrame containing the called reads.
    """
    if correction_only_in_cells:
        # First obtain transformation matrix W
        X_ = dataframe_to_values(df_bases.query("cell > 0"))
        _, W = transform_percentiles(X_.reshape(-1, channels))

        # Then apply to all data
        X = dataframe_to_values(df_bases)
        Y = W.dot(X.reshape(-1, channels).T).T.astype(int)
    else:
        X = dataframe_to_values(df_bases)
        Y, W = transform_percentiles(X.reshape(-1, channels))

    df_reads = call_barcodes(df_bases, Y, cycles=cycles, channels=channels)

    return df_reads


def dataframe_to_values(df, value="intensity"):
    """Convert a sorted DataFrame containing intensity values into a 3D NumPy array.

    Args:
        df (pandas.DataFrame): DataFrame containing intensity values.
        value (str): Column name containing the intensity values.

    Returns:
        numpy.ndarray: 3D NumPy array representing intensity values with dimensions N x cycles x channels.
    """
    # Calculate the number of cycles
    cycles = df[CYCLE].value_counts()
    assert len(set(cycles)) == 1
    n_cycles = len(cycles)

    # Calculate the number of channels
    n_channels = len(df[CHANNEL].value_counts())

    # Reshape intensity values into a 3D array
    x = np.array(df[value]).reshape(-1, n_cycles, n_channels)

    return x


def transform_medians(X, correction_quartile=0):
    """Compute a linear transformation matrix based on the median values of maximum points along each dimension of X.

    Args:
        X (numpy.ndarray): Input array.
        correction_quartile (float): Quartile used for correction.

    Returns:
        numpy.ndarray: Transformed array Y.
        numpy.ndarray: Transformation matrix W.
    """

    def get_medians(X, correction_quartile):
        arr = []
        for i in range(X.shape[1]):
            max_spots = X[X.argmax(axis=1) == i]
            try:
                arr.append(
                    np.median(
                        max_spots[
                            max_spots[:, i]
                            >= np.quantile(max_spots, axis=0, q=correction_quartile)[i]
                        ],
                        axis=0,
                    )
                )
            except:
                arr.append(np.median(max_spots, axis=0))
        M = np.array(arr)
        return M

    # Compute medians and construct matrix M
    M = get_medians(X, correction_quartile).T
    # Normalize matrix M
    M = M / M.sum(axis=0)
    # Compute the inverse of M to obtain the transformation matrix W
    W = np.linalg.inv(M)
    # Apply transformation to X
    Y = W.dot(X.T).T.astype(int)
    return Y, W


def transform_percentiles(X):
    """For each dimension, find points where that dimension is >=95th percentile intensity.

    Use median of those points to define new axes.
    Describe with linear transformation W so that W * X = Y.

    Args:
        X (numpy.ndarray): Input array of intensity values.

    Returns:
        Y (numpy.ndarray): Transformed array.
        W (numpy.ndarray): Transformation matrix.
    """

    def get_percentiles(X):
        arr = []
        for i in range(X.shape[1]):
            # Calculate relative intensities by dividing by rowsums
            rowsums = np.sum(X, axis=1)[:, np.newaxis]
            X_rel = X / rowsums

            # Find 95th percentile of relative intensities for channel i
            perc = np.nanpercentile(X_rel[:, i], 95)

            # Select spots where channel i has high relative intensity
            high = X[X_rel[:, i] >= perc]

            # Take median of those high-intensity spots
            arr += [np.median(high, axis=0)]

        M = np.array(arr)
        return M

    # Get percentile-based matrix
    M = get_percentiles(X).T

    # Normalize columns
    M = M / M.sum(axis=0)

    # Compute inverse to get transformation matrix
    W = np.linalg.inv(M)

    # Apply transformation
    Y = W.dot(X.T).T.astype(int)

    return Y, W


def call_barcodes(df_bases, Y, cycles=12, channels=4):
    """Assign barcode sequences to reads based on the transformed base signal obtained from sequencing data.

    Args:
        df_bases (pandas DataFrame): DataFrame containing base signal information for each read.
        Y (numpy array): Transformed base signal reshaped into a suitable format for the calling process.
        cycles (int): Number of sequencing cycles.
        channels (int): Number of sequencing channels.

    Returns:
        df_reads (pandas DataFrame): DataFrame with assigned barcode sequences and quality scores for each read.
    """
    # Extract unique bases
    bases = sorted(set(df_bases[CHANNEL]))

    # Check for weird bases
    if any(len(x) != 1 for x in bases):
        raise ValueError("supplied weird bases: {0}".format(bases))

    # Remove duplicate entries and create a copy for storing barcode calls
    df_reads = df_bases.drop_duplicates([WELL, TILE, READ]).copy()

    # Call barcodes based on the transformed base signal
    df_reads[BARCODE] = call_bases_fast(Y.reshape(-1, cycles, channels), bases)

    # Calculate quality scores for each read
    Q = quality(Y.reshape(-1, cycles, channels))

    # Store quality scores in DataFrame
    for i in range(len(Q[0])):
        df_reads["Q_%d" % i] = Q[:, i]

    # Assign minimum quality score for each read
    df_reads = df_reads.assign(Q_min=lambda x: x.filter(regex="Q_\d+").min(axis=1))

    # Drop unnecessary columns
    df_reads = df_reads.drop([CYCLE, CHANNEL, INTENSITY], axis=1)

    return df_reads


def call_bases_fast(values, bases):
    """Call bases based on the maximum intensity value for each cycle/channel combination.

    Args:
        values (numpy array): 3D array containing intensity values for each cycle, channel, and base.
        bases (str): String containing the base symbols corresponding to each channel.

    Returns:
        calls (list of str): List of called bases for each read.
    """
    # Check dimensions and base length
    assert values.ndim == 3
    assert values.shape[2] == len(bases)

    # Determine the index of the maximum intensity value for each cycle/channel
    calls = values.argmax(axis=2)

    # Map the index to the corresponding base symbol
    calls = np.array(list(bases))[calls]

    # Combine the base symbols for each cycle to form the called bases for each read
    return ["".join(x) for x in calls]


def quality(X):
    """Calculate quality scores for base calling.

    Quality score is based on the highest and second-highest intensity channels.
    Calculated as 1 - [log2(2 + second) / log2(2 + first)].

    Args:
        X (numpy array): Array containing intensity values.

    Returns:
        Q (numpy array): Array containing quality scores.
    """
    # Sort the intensity values and convert to float
    X = np.abs(np.sort(X, axis=-1).astype(float))

    # Calculate the quality scores
    Q = 1 - np.log(2 + X[..., -2]) / np.log(2 + X[..., -1])

    # Clip the quality scores to the range [0, 1]
    Q = (Q * 2).clip(0, 1)

    return Q


def normalize_bases(df):
    """Normalize the channel intensities by the median brightness of each channel in all spots.

    Args:
        df (pandas.DataFrame): DataFrame containing spot intensity data.

    Returns:
        pandas.DataFrame: DataFrame with normalized intensity values.
    """
    # Calculate median brightness of each channel
    df_medians = df.groupby("channel").intensity.median()

    # Normalize intensity values by dividing by respective channel median
    df_out = df.copy()
    df_out.intensity = df.apply(lambda x: x.intensity / df_medians[x.channel], axis=1)

    return df_out


def plot_normalization_comparison(
    df_bases,
    base_pairs=[("C", "A"), ("T", "G")],
    filter_to_compared_bases=True,
    base_order=["G", "T", "A", "C"],
):
    """Compare raw, median and percentile normalization with all cycles combined.

    Args:
        df_bases: DataFrame containing raw base intensities from extract_bases
        base_pairs: List of tuples representing base pairs to compare
        filter_to_compared_bases: If True, only show points where one of the compared bases was called
        base_order: List specifying the order of bases in the channel dimension (default: ["G", "T", "A", "C"])

    """
    # First, clean up the bases data
    df_bases_clean = clean_up_bases(df_bases)

    # Create figure with subplots - 3 rows (normalization methods) x 2 columns (base pairs)
    fig, axes = plt.subplots(3, 2, figsize=(14, 18))

    # Get channel indices based on the specified base order
    channel_to_idx = {base: idx for idx, base in enumerate(base_order)}

    # Define base colors - keep the original colors
    base_colors = {"G": "purple", "T": "cyan", "A": "green", "C": "red"}

    # Process the raw data
    for bp_idx, (base1, base2) in enumerate(base_pairs):
        # Filter the data for the relevant base pairs
        df_base1 = df_bases_clean[df_bases_clean["channel"] == base1][
            ["read", "cycle", "intensity"]
        ]
        df_base2 = df_bases_clean[df_bases_clean["channel"] == base2][
            ["read", "cycle", "intensity"]
        ]

        # Merge the two dataframes to get paired intensities
        df_merged = pd.merge(
            df_base1,
            df_base2,
            on=["read", "cycle"],
            suffixes=("_" + base1, "_" + base2),
        )

        # Get max intensities for base calling coloring
        all_bases = base_order.copy()  # Use the custom base order
        df_max = None

        for base in all_bases:
            df_temp = df_bases_clean[df_bases_clean["channel"] == base][
                ["read", "cycle", "intensity"]
            ]
            df_temp = df_temp.rename(columns={"intensity": "intensity_" + base})

            if df_max is None:
                df_max = df_temp
            else:
                df_max = pd.merge(df_max, df_temp, on=["read", "cycle"])

        # Determine which base has max intensity for each read/cycle
        for base in all_bases:
            if base == all_bases[0]:
                df_max["max_base"] = base
                df_max["max_intensity"] = df_max["intensity_" + base]
            else:
                mask = df_max["intensity_" + base] > df_max["max_intensity"]
                df_max.loc[mask, "max_base"] = base
                df_max.loc[mask, "max_intensity"] = df_max["intensity_" + base]

        # Merge with the paired data
        df_merged = pd.merge(
            df_merged, df_max[["read", "cycle", "max_base"]], on=["read", "cycle"]
        )

        # Plot raw data (first row)
        ax = axes[0, bp_idx]

        # Filter to only show the compared bases if requested
        if filter_to_compared_bases:
            plot_bases = [base1, base2]
        else:
            plot_bases = all_bases

        for base in plot_bases:
            mask = df_merged["max_base"] == base
            if mask.any():
                ax.scatter(
                    df_merged.loc[mask, "intensity_" + base2],
                    df_merged.loc[mask, "intensity_" + base1],
                    color=base_colors[base],
                    alpha=0.5,
                    s=10,
                    label=f"'{base}' base call",
                )

        title_suffix = " (filtered)" if filter_to_compared_bases else ""
        ax.set_title(f"All Cycles: {base1}-{base2} (Raw){title_suffix}")
        ax.set_xlabel(base2)
        ax.set_ylabel(base1)

        # Now process for median and percentile normalization
        # First get all base intensities for each read/cycle
        data_all = df_bases_clean.pivot_table(
            index=["read", "cycle"],
            columns="channel",
            values="intensity",
            aggfunc="first",
        ).reset_index()

        # Make sure all bases are present in data_all
        for base in all_bases:
            if base not in data_all.columns:
                data_all[base] = 0

        # Convert to numpy array for normalization functions (ensure correct order)
        X = data_all[all_bases].values

        # Apply median normalization
        # First normalize bases
        df_bases_norm = normalize_bases(df_bases_clean)
        X_norm = dataframe_to_values(df_bases_norm)
        Y_median, W_median = transform_medians(X_norm.reshape(-1, len(all_bases)))

        # Apply percentile normalization
        Y_percentile, W_percentile = transform_percentiles(
            X.reshape(-1, len(all_bases))
        )

        # Add normalized values back to the dataframe
        data_all["median_" + base1] = Y_median[:, channel_to_idx[base1]]
        data_all["median_" + base2] = Y_median[:, channel_to_idx[base2]]
        data_all["percentile_" + base1] = Y_percentile[:, channel_to_idx[base1]]
        data_all["percentile_" + base2] = Y_percentile[:, channel_to_idx[base2]]

        # Determine max channel after normalization
        for method in ["median", "percentile"]:
            values = Y_median if method == "median" else Y_percentile
            max_indices = np.argmax(values, axis=1)
            data_all[method + "_max_base"] = [all_bases[i] for i in max_indices]

        # Plot median normalized data (second row)
        ax = axes[1, bp_idx]

        # Filter to only show the compared bases if requested
        if filter_to_compared_bases:
            plot_bases = [base1, base2]
        else:
            plot_bases = all_bases

        for base in plot_bases:
            mask = data_all["median_max_base"] == base
            if mask.any():
                ax.scatter(
                    data_all.loc[mask, "median_" + base2],
                    data_all.loc[mask, "median_" + base1],
                    color=base_colors[base],
                    alpha=0.5,
                    s=10,
                    label=f"'{base}' base call",
                )

        # Add diagonal reference line
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.7)

        ax.set_title(f"All Cycles: {base1}-{base2} (Median){title_suffix}")
        ax.set_xlabel(base2)
        ax.set_ylabel(base1)

        # Plot percentile normalized data (third row)
        ax = axes[2, bp_idx]

        # Filter to only show the compared bases if requested
        if filter_to_compared_bases:
            plot_bases = [base1, base2]
        else:
            plot_bases = all_bases

        for base in plot_bases:
            mask = data_all["percentile_max_base"] == base
            if mask.any():
                ax.scatter(
                    data_all.loc[mask, "percentile_" + base2],
                    data_all.loc[mask, "percentile_" + base1],
                    color=base_colors[base],
                    alpha=0.5,
                    s=10,
                    label=f"'{base}' base call",
                )

        # Add diagonal reference line
        max_val = max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.7)

        ax.set_title(f"All Cycles: {base1}-{base2} (Percentile){title_suffix}")
        ax.set_xlabel(base2)
        ax.set_ylabel(base1)

    # Create a shared legend at the bottom
    # Choose which bases to show in the legend based on filter setting
    if filter_to_compared_bases:
        # Get unique bases from all base pairs
        legend_bases = set()
        for base1, base2 in base_pairs:
            legend_bases.add(base1)
            legend_bases.add(base2)
        legend_bases = sorted(legend_bases)
    else:
        legend_bases = sorted(all_bases)

    # Create legend elements
    legend_elements = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=base_colors[base],
            markersize=10,
            label=f"'{base}' base call",
        )
        for base in legend_bases
    ]

    # Add diagonal line to legend if present in plots
    legend_elements.append(
        Line2D([0], [0], linestyle="--", color="k", label="Diagonal reference (y=x)")
    )

    # Add legend at the bottom
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=len(legend_elements),
        fontsize=12,
    )

    # Adjust layout to make room for the legend
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
