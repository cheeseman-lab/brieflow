"""Sequencing Data Processing and Base Calling Utilities!

This module provides a set of functions for mapping and analyzing sequencing reads
(step 2 -- sbs base calling). It includes functions for:

1. Base Intensity Extraction: Functions to extract base intensities from sequencing data.
2. Base Calling: Algorithms for calling bases from raw intensity data.
3. Read Formatting: Tools to format and normalize sequencing reads.
4. Quality Score Calculation: Functions to compute quality scores for sequencing reads.
5. Barcode Assignment: Utilities for assigning barcodes to cells based on sequencing data.
"""

import numpy as np
import pandas as pd

# constants for extracting bases
CHANNEL = "channel"
CYCLE = "cycle"
POSITION_I = "i"
POSITION_J = "j"
INTENSITY = "intensity"
CELL = "cell"
READ = "read"


def extract_bases(
    peaks_data, max_filtered_data, cells_data, threshold_peaks, wildcards, bases="GTAC"
):
    """Find the signal intensity from `maxed` at each point in `peaks` above `threshold_peaks`.

    Output is labeled by `wildcards` (e.g., well and tile) and label at that position in integer
    mask `cells`.

    Args:
        peaks_data (numpy.ndarray): Peaks/local maxima score for each pixel, output of Snake.find_peaks().
        max_filtered_data (numpy.ndarray): Base intensity at each point, output of Snake.max_filter(), expected dimensions of (CYCLE, CHANNEL, I, J).
        cells_data (numpy.ndarray): Labeled segmentation mask of cell boundaries for labeling reads.
        threshold_peaks (float): Threshold for identifying candidate sequencing reads based on peaks.
        wildcards (dict): Metadata to include in output table, e.g., well, tile, etc. In Snakemake, use wildcards object.
        bases (str, optional): Order of bases corresponding to the order of acquired SBS channels in `maxed`. Default is 'GTAC'.

    Returns:
        pandas.DataFrame: Table of all candidate sequencing reads with intensity of each base for every cycle,
            (I,J) position of read, and metadata from `wildcards`.
    """
    if max_filtered_data.ndim == 3:
        max_filtered_data = max_filtered_data[None]

    # TODO: determine if we need to go back to the original code
    # If so, remove transpose and fix shape while loading
    max_filtered_data = np.transpose(max_filtered_data, (0, 3, 1, 2))

    if len(bases) != max_filtered_data.shape[1]:
        error = "Sequencing {0} bases {1} but maxed data had shape {2}"
        raise ValueError(error.format(len(bases), bases, max_filtered_data.shape))

    # "cycle 0" is reserved for phenotyping
    cycles = list(range(1, max_filtered_data.shape[0] + 1))
    bases = list(bases)

    # Extract base intensity values, labels, and positions
    values, labels, positions = extract_base_intensity(
        max_filtered_data, peaks_data, cells_data, threshold_peaks
    )

    # Format base intensity data into DataFrame
    df_bases = format_bases(values, labels, positions, cycles, bases)

    # Add wildcard metadata to the DataFrame
    for k, v in sorted(wildcards.items()):
        df_bases[k] = v

    return df_bases


def extract_base_intensity(maxed, peaks, cells, threshold_peaks):
    """Extract base intensity values, labels, and positions.

    Args:
        maxed (numpy.ndarray): Base intensity at each point, with dimensions (CYCLE, CHANNEL, I, J).
        peaks (numpy.ndarray): Peaks/local maxima score for each pixel.
        cells (numpy.ndarray): Labeled segmentation mask of cell boundaries.
        threshold_peaks (float): Threshold for identifying candidate sequencing reads based on peaks.

    Returns:
        tuple: Tuple containing values (base intensity values), labels (cell labels), and positions (positions of reads).
    """
    # Create a mask to identify reads outside of cells based on the peaks exceeding the threshold
    read_mask = peaks > threshold_peaks

    # Select base intensity values, corresponding labels, and positions for the identified reads
    values = maxed[:, :, read_mask].transpose([2, 0, 1])
    labels = cells[read_mask]
    positions = np.array(np.where(read_mask)).T

    return values, labels, positions


def format_bases(values, labels, positions, cycles, bases):
    """Format extracted base intensity values, labels, and positions into a pandas DataFrame.

    Args:
        values (numpy.ndarray): Base intensity values extracted from the sequencing data.
        labels (numpy.ndarray): Labels corresponding to each read.
        positions (numpy.ndarray): Positions of the reads.
        cycles (int): Number of sequencing cycles.
        bases (list): List of bases corresponding to the sequencing channels.

    Returns:
        pandas.DataFrame: Formatted DataFrame containing base intensity values, labels, and positions.
    """
    index = (CYCLE, cycles), (CHANNEL, bases)

    try:
        # Attempt to reshape the extracted pixels to sequencing bases
        df = ndarray_to_dataframe(values, index)
    except ValueError:
        print(
            "Failed to reshape extracted pixels to sequencing bases, writing empty table"
        )
        return pd.DataFrame()

    # Create a DataFrame containing positions of the reads
    df_positions = pd.DataFrame(positions, columns=[POSITION_I, POSITION_J])

    # Stack the DataFrame to include cycles and bases as columns, reset the index, and rename columns
    df = (
        df.stack([CYCLE, CHANNEL])
        .reset_index()
        .rename(columns={0: INTENSITY, "level_0": READ})
        .join(pd.Series(labels, name=CELL), on=READ)
        .join(df_positions, on=READ)
        .sort_values([CELL, READ, CYCLE])
    )

    return df


def ndarray_to_dataframe(values, index):
    """Convert a numpy array to a DataFrame with MultiIndex columns.

    Args:
        values (np.ndarray): Input array.
        index (list of tuples): List of (name, levels) tuples for MultiIndex.

    Returns:
        pd.DataFrame: Resulting DataFrame.
    """
    names, levels = zip(*index)
    columns = pd.MultiIndex.from_product(levels, names=names)
    df = pd.DataFrame(values.reshape(values.shape[0], -1), columns=columns)
    return df
