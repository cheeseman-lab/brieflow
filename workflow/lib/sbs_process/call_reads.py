"""Read Calling Utilities!

This module provides a set of functions for mapping and analyzing sequencing reads.
It includes functions for:

- Read Calling: Algorithms for calling bases from raw intensity data.
- Read Formatting: Tools to format and normalize sequencing reads.
"""

import numpy as np

# constants for calling reads
from lib.sbs_process.sbs_process_constants import (
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
):
    """Call reads for in situ sequencing data.

    Call reads by compensating for channel cross-talk and calling the base
    with the highest corrected intensity for each cycle. Median correction
    is performed independently for each tile.

    Args:
    bases_data : pandas DataFrame
        Table of base intensity for all candidate reads, output of Snake.extract_bases().

    peaks_data : None or numpy array, default None
        Peaks/local maxima score for each pixel (output of Snake.find_peaks()) to be included
        in the df_reads table for downstream QC or other analysis. If None, does not include
        peaks scores in returned df_reads table.

    correction_only_in_cells : boolean, default True
        If True, restricts median correction/compensation step to account only for reads that
        are within a cell, as defined by the cell segmentation mask passed into
        Snake.extract_bases(). Often identified spots outside of cells are not true sequencing
        reads.

    normalize_bases_first : boolean, default True
        If True, normalizes the base intensities before performing median correction.

    Returns:
    df_reads : pandas DataFrame
        Table of all reads with base calls resulting from SBS compensation and related metadata.
    """
    if bases_data is None:
        return
    if correction_only_in_cells:
        if len(bases_data.query("cell > 0")) == 0:
            return

    cycles = len(set(bases_data["cycle"]))
    channels = len(set(bases_data["channel"]))

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
    """Calculate quality scores based on the intensity values.

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
