"""Utility functions for calling cells from multi-barcode sequencing reads."""

import pandas as pd
import numpy as np
import Levenshtein

# constants for calling cells
from lib.sbs.constants import (
    PREFIX,
    SGRNA,
    GENE_SYMBOL,
    GENE_ID,
    WELL,
    TILE,
    CELL,
    READ,
    BARCODE,
    BARCODE_COUNT,
    BARCODE_0,
    BARCODE_1,
    BARCODE_COUNT_0,
    BARCODE_COUNT_1,
    POSITION_I,
    POSITION_J,
)


def call_cells_multi(
    reads_data,
    df_barcode_library=None,
    q_min=0,
    map_col="prefix_map",
    error_correct=False,
    recomb_col=None,
    recomb_filter_col=None,
    recomb_q_thresh=0.1,
    barcode_info_cols=[GENE_SYMBOL],
    **kwargs,
):
    """Process multi-barcode sequencing reads to identify cell barcodes, optionally mapping them to a pool design.

    This function takes multi-barcode sequencing read data and identifies the top barcodes for each cell,
    prioritizing the brightest spots per cell. Intended for multi-barcode IVT detection protocols.

    Args:
        reads_data (DataFrame): DataFrame containing read information.
        df_barcode_library (DataFrame, optional): DataFrame containing pool information. Default is None.
        q_min (int, optional): Minimum quality threshold. Default is 0.
        map_col (str, optional): The column within df_reads and df_barcode_library containing the barcode for mapping.
            Default is 'prefix_map'.
        error_correct (bool, optional): Whether to perform error correction of barcodes. Default is False.
        recomb_col (None, str, optional): The column within df_reads and df_barcode_library containing the barcode for
            determining barcode recombination. Default is None.
        recomb_filter_col (None, str, optional): Column to use for filtering recombination identification confidence.
            Default is None.
        recomb_q_thresh (float, optional): Threshold for recomb_filter_col below which recombination events
            will be indeterminate. Default is 0.1.
        barcode_info_cols (list, optional): List of columns from df_barcode_library to maintain. Default is [GENE_SYMBOL].
        **kwargs: Additional arguments passed to error_correct_reads if error_correct is True.

    Returns:
        DataFrame: DataFrame containing corrected cells with top barcodes.
    """
    # Check if reads_data is None or empty and return if so
    if reads_data is None or reads_data.empty:
        columns = [
            "cell",
            "tile",
            "well",
            "Q_min",
            "peak",
            "cell_barcode_0",
            "peak_0",
            "cell_barcode_1",
            "peak_1",
        ]
        return pd.DataFrame(columns=columns)

    # Apply quality filter
    df_reads = reads_data.query("Q_min >= @q_min")

    # Process with appropriate helper function based on whether df_barcode_library is provided
    if df_barcode_library is None:
        # Use helper function for no mapping case
        return call_cells_multi_helper(df_reads)
    else:
        # Use mapping function when a reference pool is provided
        return call_cells_multi_mapping(
            df_reads,
            df_barcode_library,
            map_col=map_col,
            recomb_col=recomb_col,
            recomb_filter_col=recomb_filter_col,
            recomb_q_thresh=recomb_q_thresh,
            error_correct=error_correct,
            barcode_info_cols=barcode_info_cols,
            **kwargs,
        )


def call_cells_multi_helper(df_reads):
    """Determine the top barcodes for each cell based on peak intensity without mapping to a reference.

    Parameters:
        df_reads (pandas.DataFrame): DataFrame containing sequencing reads.

    Returns:
        pandas.DataFrame: DataFrame with the top barcodes for each cell.
    """
    # Columns for grouping
    cols = [WELL, TILE, CELL]

    # Sort by peak intensity and group by cell
    s = df_reads.sort_values("peak", ascending=False).groupby(cols)

    # Create DataFrame with top 2 barcodes by peak intensity
    df_cells = (
        df_reads.join(
            s.nth(0)[cols + ["barcode", "peak"]]
            .rename(columns={"barcode": BARCODE_0, "peak": "peak_0"})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[cols + ["barcode", "peak"]]
            .rename(columns={"barcode": BARCODE_1, "peak": "peak_1"})
            .set_index(cols),
            on=cols,
        )
        .drop_duplicates(cols)
        .drop([READ, BARCODE, "peak"], axis=1)
        .drop([POSITION_I, POSITION_J], axis=1)  # drop the read coordinates
        .filter(regex="^(?!Q_)")  # remove read quality scores
        .query("cell > 0")  # remove reads not in a cell
    )

    return df_cells


def call_cells_multi_mapping(
    df_reads,
    df_barcode_library,
    map_col="prefix_map",
    recomb_col=None,
    recomb_filter_col=None,
    recomb_q_thresh=0.1,
    error_correct=False,
    barcode_info_cols=[GENE_SYMBOL],
    **kwargs,
):
    """Determine the top barcodes for each cell with mapping to a reference pool, prioritizing by peak intensity.

    Parameters:
        df_reads (pandas.DataFrame): DataFrame containing sequencing reads.
        df_barcode_library (pandas.DataFrame): DataFrame containing the reference barcode sequences.
        map_col (str): The column within df_reads and df_barcode_library containing the barcode for mapping.
        recomb_col (None, str): Optional. The column for determining barcode recombination.
        recomb_filter_col (None, str): Column for filtering recombination identification confidence.
        recomb_q_thresh (float): Threshold for recomb_filter_col below which recombination events are indeterminate.
        error_correct (bool): Whether to perform error correction of barcodes.
        barcode_info_cols (list): List of columns from df_barcode_library to maintain.

    Returns:
        pandas.DataFrame: DataFrame with the top barcodes for each cell mapped to reference.
    """
    # Columns for grouping
    cols = [WELL, TILE, CELL]

    # Optionally perform error correction on barcodes
    if error_correct:
        print("performing error correction")
        pre_correct_col = map_col + "_pre_correction"
        # Store original values before correction
        df_reads[pre_correct_col] = df_reads[map_col]
        # Perform error correction
        df_reads[map_col] = error_correct_reads(
            df_reads[map_col],
            df_barcode_library[map_col],
            **kwargs,
        )

    # Map reads to the reference pool
    df_barcode_library["map_temp"] = df_barcode_library[map_col]
    df_mapped = (
        pd.merge(
            df_reads,
            df_barcode_library[["map_temp"]],
            how="left",
            left_on=map_col,
            right_on="map_temp",
        )
        .assign(
            mapped=lambda x: pd.notnull(x["map_temp"])
        )  # Flag indicating if barcode is mapped
        .drop("map_temp", axis=1)  # Drop the temporary prefix column
    )

    # Handle recombination detection if specified
    if recomb_col is not None:
        # Create mapping of expected recombination values
        recomb_map = df_barcode_library.set_index(map_col)[recomb_col].to_dict()
        # Flag sequences where actual recombination matches expected recombination
        df_mapped["no_recomb"] = (df_mapped[map_col].map(recomb_map)) == (
            df_mapped[recomb_col]
        )
        # Unmapped cells have undetermined recombination status
        df_mapped.loc[~df_mapped.mapped, "no_recomb"] = np.nan
        df_mapped = df_mapped.drop(columns=[recomb_col])

        # Apply quality threshold for recombination status if specified
        if recomb_filter_col is not None:
            # If below quality threshold, mark as undetermined recombination status
            df_mapped.loc[
                df_mapped[recomb_filter_col] < recomb_q_thresh, "no_recomb"
            ] = np.nan
    else:
        # No recombination detection requested
        df_mapped["no_recomb"] = np.nan

    # Group by cell and sort by mapped status (True first) and peak intensity (highest first)
    s = (
        df_mapped.drop_duplicates([WELL, TILE, READ])
        .sort_values(
            ["mapped", "peak"], ascending=[False, False]
        )  # True comes after False by default, so use False, False
        .groupby(cols)
    )

    # Create cells dataframe with top barcodes
    if error_correct:
        # Include pre-correction information when error correction was performed
        df_cells = (
            df_reads.join(
                s.nth(0)[cols + [map_col, pre_correct_col, "no_recomb", "peak"]]
                .rename(
                    columns={
                        map_col: BARCODE_0,
                        pre_correct_col: str("pre_correction_" + BARCODE_0),
                        "no_recomb": "no_recomb_0",
                        "peak": "peak_0",
                    }
                )
                .set_index(cols),
                on=cols,
            )
            .join(
                s.nth(1)[cols + [map_col, "no_recomb", "peak"]]
                .rename(
                    columns={
                        map_col: BARCODE_1,
                        "no_recomb": "no_recomb_1",
                        "peak": "peak_1",
                    }
                )
                .set_index(cols),
                on=cols,
            )
            .drop_duplicates(cols)
            .drop(
                [READ, BARCODE, map_col, pre_correct_col, "peak"], axis=1
            )  # Drop unnecessary columns
            .drop([POSITION_I, POSITION_J], axis=1)  # Drop the read coordinates
            .filter(regex="^(?!Q_)")  # Remove read quality scores
            .query("cell > 0")  # Remove reads not in a cell
        )
    else:
        # Standard processing without error correction information
        df_cells = (
            df_reads.join(
                s.nth(0)[cols + [map_col, "no_recomb", "peak"]]
                .rename(
                    columns={
                        map_col: BARCODE_0,
                        "no_recomb": "no_recomb_0",
                        "peak": "peak_0",
                    }
                )
                .set_index(cols),
                on=cols,
            )
            .join(
                s.nth(1)[cols + [map_col, "no_recomb", "peak"]]
                .rename(
                    columns={
                        map_col: BARCODE_1,
                        "no_recomb": "no_recomb_1",
                        "peak": "peak_1",
                    }
                )
                .set_index(cols),
                on=cols,
            )
            .drop_duplicates(cols)
            .drop([READ, BARCODE, map_col, "peak"], axis=1)  # Drop unnecessary columns
            .drop([POSITION_I, POSITION_J], axis=1)  # Drop the read coordinates
            .filter(regex="^(?!Q_)")  # Remove read quality scores
            .query("cell > 0")  # Remove reads not in a cell
        )

    # Remove any lingering 'no_recomb' column that might exist
    if "no_recomb" in df_cells.columns:
        df_cells.drop(columns=["no_recomb"], inplace=True)

    # Merge guide information for barcode 0
    df_cells = (
        pd.merge(
            df_cells,
            df_barcode_library[[map_col] + barcode_info_cols],
            how="left",
            left_on=BARCODE_0,
            right_on=map_col,
        )
        .rename(
            {col: col + "_0" for col in barcode_info_cols}, axis=1
        )  # Rename columns for clarity
        .drop(map_col, axis=1)  # Drop the temporary map column
    )

    # Merge guide information for barcode 1
    df_cells = (
        pd.merge(
            df_cells,
            df_barcode_library[[map_col] + barcode_info_cols],
            how="left",
            left_on=BARCODE_1,
            right_on=map_col,
        )
        .rename(
            {col: col + "_1" for col in barcode_info_cols}, axis=1
        )  # Rename columns for clarity
        .drop(map_col, axis=1)  # Drop the temporary map column
    )

    return df_cells


def error_correct_reads(reads, reference, max_distance=2, distance_metric="hamming"):
    """Error correct reads against a reference set of barcodes.

    Compares each read to the reference set and corrects it to the closest unique reference
    if within the specified distance threshold.

    Args:
        reads (pd.Series): Series with reads for error correction
        reference (pd.Series): Series with reference sequences
        max_distance (int, optional): Maximum distance for correction. Correction is performed
            only if (1) one reference sequence is closest (no ties) and (2) that unique reference
            sequence is within this distance. Default is 2.
        distance_metric (str, optional): Distance metric to compare barcodes.
            Options are 'hamming' (default) and 'levenshtein'.

    Returns:
        pd.Series: Corrected reads
    """
    # Calculate distance from each read to each reference barcode
    dist_to_ref = barcode_distance_matrix(
        reads.to_list(),
        reference.to_list(),
        distance_metric=distance_metric,
    )

    # Find minimum distance to reference for each read
    min_dist_to_ref = dist_to_ref.min(axis=1)

    # Determine which reads have a unique closest match
    unique_dist = np.array(
        [
            np.sum(dist_to_ref[x] == min_dist_to_ref[x]) == 1
            for x in range(dist_to_ref.shape[0])
        ]
    )

    # Filter for reads that have a unique closest match within max_distance
    corrected_subset = (unique_dist) & (min_dist_to_ref <= max_distance)

    # Get the corrected barcodes for eligible reads
    corrected_barcodes = reference.loc[
        dist_to_ref[corrected_subset].argmin(axis=1)
    ].values

    # Create copy of reads and update only the ones that can be corrected
    corrected_reads = reads.copy()
    corrected_reads.loc[corrected_subset] = corrected_barcodes

    return corrected_reads


def barcode_distance_matrix(barcodes_1, barcodes_2=False, distance_metric="hamming"):
    """Calculate distances between two sets of barcodes.

    Creates a matrix of distances between all pairs of barcodes from two sets.
    If only one set is provided, computes self-distances.

    Args:
        barcodes_1 (list): First list of barcode sequences
        barcodes_2 (list or bool, optional): Second list of barcode sequences.
            If False, uses barcodes_1 for both sets. Default is False.
        distance_metric (str, optional): Type of distance to calculate.
            Options are 'hamming' or 'levenshtein'. Default is 'hamming'.

    Returns:
        numpy.ndarray: Matrix of distances between barcode pairs
    """
    import warnings

    # Define the distance function based on chosen metric
    if distance_metric == "hamming":
        distance = lambda i, j: Levenshtein.hamming(i, j)
    elif distance_metric == "levenshtein":
        distance = lambda i, j: Levenshtein.distance(i, j)
    else:
        warnings.warn(
            'distance_metric must be "hamming" or "levenshtein" - defaulting to "hamming"'
        )
        distance = lambda i, j: Levenshtein.hamming(i, j)

    # If second set not provided, use the first set
    if isinstance(barcodes_2, bool):
        barcodes_2 = barcodes_1

    # Create distance matrix for all barcode pairs
    bc_distance_matrix = np.zeros((len(barcodes_1), len(barcodes_2)))
    for a, i in enumerate(barcodes_1):
        for b, j in enumerate(barcodes_2):
            bc_distance_matrix[a, b] = distance(i, j)

    return bc_distance_matrix


def prep_multi_reads(
    df_reads,
    map_start,
    map_end,
    recomb_start,
    recomb_end,
    map_col="prefix_map",
    recomb_col="prefix_recomb",
):
    """Prepare reads DataFrame for multi cell calling by creating necessary columns for mapping and recombination detection.

    Args:
        df_reads (DataFrame): DataFrame containing raw sequencing reads
        map_start (int): Starting cycle number for mapping (1-indexed)
        map_end (int): Ending cycle number for mapping (1-indexed, inclusive)
        recomb_start (int): Starting cycle number for recombination detection (1-indexed)
        recomb_end (int): Ending cycle number for recombination detection (1-indexed, inclusive)
        map_col (str, optional): Name for the mapping column. Default is 'prefix_map'
        recomb_col (str, optional): Name for the recombination column. Default is 'prefix_recomb'
    Returns:
        DataFrame: Prepared reads DataFrame with mapping and recombination columns
    """
    # Make a copy to avoid modifying the original DataFrame
    df = df_reads.copy()

    # Handle empty DataFrame case
    if df.empty:
        print(
            "Warning: DataFrame is empty, returning empty DataFrame with required columns"
        )
        # Add the required columns with appropriate dtypes
        df[map_col] = pd.Series(dtype="object")
        df[recomb_col] = pd.Series(dtype="object")
        df["Q_recomb"] = pd.Series(dtype="float64")
        return df

    # Check what quality columns are available
    available_q_cols = [
        col for col in df.columns if col.startswith("Q_") and col[2:].isdigit()
    ]
    max_cycle = (
        max([int(col[2:]) for col in available_q_cols]) + 1 if available_q_cols else 0
    )

    print(f"Available quality columns: {sorted(available_q_cols)}")
    print(f"Maximum cycle available: {max_cycle}")
    print(f"Requested mapping range: cycles {map_start}-{map_end}")
    print(f"Requested recombination range: cycles {recomb_start}-{recomb_end}")

    # Validate mapping range
    if map_end > max_cycle:
        raise ValueError(
            f"Mapping end cycle {map_end} exceeds available cycles (max: {max_cycle})"
        )

    # Validate recombination range
    if recomb_end > max_cycle:
        raise ValueError(
            f"Recombination end cycle {recomb_end} exceeds available cycles (max: {max_cycle})"
        )

    # Create mapping column from specified cycles (adjust for 0-indexing)
    df[map_col] = df["barcode"].str.slice(map_start - 1, map_end)

    # Create recombination column from specified cycles (adjust for 0-indexing)
    df[recomb_col] = df["barcode"].str.slice(recomb_start - 1, recomb_end)

    # Create quality column for recombination detection
    recomb_cycles = list(range(recomb_start, recomb_end + 1))
    recomb_q_cols = [f"Q_{c - 1}" for c in recomb_cycles]

    # Check if all required quality columns exist
    missing_cols = [col for col in recomb_q_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Missing quality columns for recombination detection: {missing_cols}"
        )

    df["Q_recomb"] = df[recomb_q_cols].min(axis=1)

    return df
