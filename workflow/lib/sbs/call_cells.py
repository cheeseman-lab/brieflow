"""Utility functions for calling cells from sequencing reads."""

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
    UMI_0,
    UMI_1,
    UMI_COUNT,
    UMI_COUNT_0,
    UMI_COUNT_1,
)


def call_cells(
    reads_data,
    df_barcode_library=None,
    q_min=0,
    barcode_col="sgRNA",
    prefix_col=None,
    df_UMI=None,
    error_correct=False,
    **kwargs,
):
    """Process sequencing reads to identify cell barcodes, optionally mapping them to a pool design.

    This function takes sequencing read data and identifies the most frequent barcodes for each cell.
    If a pool design is provided, it will map the barcodes to the corresponding guide RNAs and gene information.

    Args:
        reads_data (DataFrame): DataFrame containing read information.
        df_barcode_library (DataFrame, optional): DataFrame containing barcode library information. Default is None.
        q_min (int, optional): Minimum quality threshold. Default is 0.
        barcode_col (str, optional): Column in df_barcode_library with full barcode sequences for dynamic prefix creation.
            Default is 'sgRNA'. Only used if prefix_col is None.
        prefix_col (str, optional): Column in df_barcode_library with pre-computed prefixes for barcode matching.
            If specified, uses these prefixes directly instead of creating them from barcode_col.
            Useful for cycle-skipping scenarios. Default is None.
        df_UMI (DataFrame, optional): DataFrame containing UMI reads. Default is None.
        error_correct (bool, optional): Whether to perform error correction on barcodes. Default is False.
        **kwargs: Additional arguments passed to error_correct_reads if error_correct is True.
                 Common options include:
                 - max_distance (int): Maximum distance threshold for correction (default: 2)
                 - distance_metric (str): Type of distance ('hamming' or 'levenshtein')
    Returns:
        DataFrame: DataFrame containing cell-level barcode calling results.
    """
    # Check if df_reads is None and return if so
    if reads_data.empty:
        columns = [
            "cell",
            "tile",
            "well",
            "Q_0",
            "Q_1",
            "Q_2",
            "Q_3",
            "Q_4",
            "Q_5",
            "Q_6",
            "Q_7",
            "Q_8",
            "Q_9",
            "Q_10",
            "Q_min",
            "peak",
            "cell_barcode_0",
            "cell_barcode_count_0",
            "cell_barcode_1",
            "cell_barcode_count_1",
            "barcode_count",
            "sgRNA_0",
            "gene_symbol_0",
            "gene_id_0",
            "sgRNA_1",
            "gene_symbol_1",
            "gene_id_1",
        ]
        return pd.DataFrame(columns=columns)

    # Columns for grouping
    cols = [WELL, TILE, CELL]

    # Check if df_barcode_library is None
    if df_barcode_library is None:
        # Filter reads by quality threshold and call cells no ref
        df_cells = reads_data.query("Q_min >= @q_min").pipe(call_cells_no_ref)
    else:
        if prefix_col is not None:
            # Use pre-computed prefixes from the library
            if prefix_col not in df_barcode_library.columns:
                raise ValueError(f"Column '{prefix_col}' not found in barcode library")
            df_barcode_library[PREFIX] = df_barcode_library[prefix_col]
            print(f"Using pre-computed prefixes from '{prefix_col}' column")
        else:
            # Determine the experimental prefix length and create prefixes
            prefix_length = len(reads_data.iloc[0].barcode)
            df_barcode_library[PREFIX] = df_barcode_library.apply(
                lambda x: x[barcode_col][:prefix_length], axis=1
            )
            print(
                f"Created prefixes by truncating '{barcode_col}' to length {prefix_length}"
            )

        # Filter reads by quality threshold and call cells mapping
        df_cells = reads_data.query("Q_min >= @q_min").pipe(
            call_cells_mapping,
            df_barcode_library,
            error_correct=error_correct,
            **kwargs,
        )

    # If UMI data is provided, add UMI information to the cell data
    if df_UMI is not None:
        return call_cells_add_UMIs(df_cells, df_UMI, cols=cols)

    return df_cells


def call_cells_no_ref(df_reads):
    """Determine the count of top barcodes for each cell based on peak intensity without mapping to a reference.

    Args:
        df_reads (pandas.DataFrame): DataFrame containing sequencing reads.

    Returns:
        pandas.DataFrame: DataFrame with the count of top barcodes for each cell.
    """
    cols = [WELL, TILE, CELL]
    s = (
        df_reads.drop_duplicates([WELL, TILE, READ])  # Drop duplicate reads
        .groupby(cols)[BARCODE]  # Group by well, tile, and cell, and barcode
        .value_counts()  # Count occurrences of each barcode within each group
        .rename("count")  # Rename the resulting series to 'count'
        .sort_values(ascending=False)  # Sort in descending order
        .reset_index()  # Reset the index
        .groupby(cols)  # Group again by well, tile, and cell
    )

    return (
        df_reads.join(
            s.nth(0)[["well", "tile", "cell", "barcode"]]
            .rename(columns={"barcode": BARCODE_0})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(0)[["well", "tile", "cell", "count"]]
            .rename(columns={"count": BARCODE_COUNT_0})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[["well", "tile", "cell", "barcode"]]
            .rename(columns={"barcode": BARCODE_1})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[["well", "tile", "cell", "count"]]
            .rename(columns={"count": BARCODE_COUNT_1})
            .set_index(cols),
            on=cols,
        )
        .join(s["count"].sum().rename(BARCODE_COUNT), on=cols)
        .assign(
            **{
                BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0),
                BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0),
            }
        )
        .drop_duplicates(cols)
        .drop([READ, BARCODE], axis=1)  # drop the read
        .drop([POSITION_I, POSITION_J], axis=1)  # drop the read coordinates
        .filter(regex="^(?!Q_)")  # remove read quality scores
        .query("cell > 0")  # remove reads not in a cell
    )


def call_cells_mapping(
    df_reads,
    df_barcode_library,
    barcode_info_cols=[SGRNA, GENE_SYMBOL, GENE_ID],
    error_correct=False,
    **kwargs,
):
    """Determine the count of top barcodes, with prioritization given to barcodes mapping to the given pool design.
    
    Args:
        df_reads (DataFrame): DataFrame containing read data.
        df_barcode_library (DataFrame): DataFrame containing barcode library information.
        barcode_info_cols (list, optional): Columns related to guide information. Default is [SGRNA, GENE_SYMBOL, GENE_ID].
        error_correct (bool, optional): Whether to perform error correction. Default is False.
        **kwargs: Additional arguments passed to error_correct_reads if error_correct is True.

    Returns:
        DataFrame: DataFrame containing the count of top barcodes along with merged guide information.
    """
    # Optionally perform error correction
    if error_correct:
        print("performing error correction")
        df_reads[BARCODE] = error_correct_reads(
            df_reads[BARCODE], df_barcode_library[PREFIX], **kwargs
        )

    # Map reads to the pool design
    df_mapped = (
        pd.merge(
            df_reads,
            df_barcode_library[[PREFIX]],
            how="left",
            left_on=BARCODE,
            right_on=PREFIX,
        )
        .assign(
            mapped=lambda x: pd.notnull(x[PREFIX])
        )  # Flag indicating if barcode is mapped
        .drop(PREFIX, axis=1)  # Drop the temporary prefix column
    )

    # Choose top 2 barcodes, priority given by (mapped, count)
    cols = [WELL, TILE, CELL]
    s = (
        df_mapped.drop_duplicates([WELL, TILE, READ])
        .groupby(cols + ["mapped"])[BARCODE]
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values(["mapped", "count"], ascending=False)
        .groupby(cols)
    )

    # Create DataFrame containing top barcodes and their counts
    df_cells = (
        df_reads.join(
            s.nth(0)[["well", "tile", "cell", "barcode"]]
            .rename(columns={"barcode": BARCODE_0})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(0)[["well", "tile", "cell", "count"]]
            .rename(columns={"count": BARCODE_COUNT_0})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[["well", "tile", "cell", "barcode"]]
            .rename(columns={"barcode": BARCODE_1})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[["well", "tile", "cell", "count"]]
            .rename(columns={"count": BARCODE_COUNT_1})
            .set_index(cols),
            on=cols,
        )
        .join(s["count"].sum().rename(BARCODE_COUNT), on=cols)
        .assign(
            **{
                BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0),
                BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0),
            }
        )
        .drop_duplicates(cols)  # Remove duplicate rows
        .drop([READ, BARCODE], axis=1)  # Drop unnecessary columns
        .drop([POSITION_I, POSITION_J], axis=1)  # Drop the read coordinates
        .query("cell > 0")  # Remove reads not in a cell
    )

    # Merge guide information for barcode 0
    df_cells = (
        pd.merge(
            df_cells,
            df_barcode_library[[PREFIX] + barcode_info_cols],
            how="left",
            left_on=BARCODE_0,
            right_on=PREFIX,
        )
        .rename(
            {col: col + "_0" for col in barcode_info_cols}, axis=1
        )  # Rename columns for clarity
        .drop(PREFIX, axis=1)  # Drop the temporary prefix column
    )
    # Merge guide information for barcode 1
    df_cells = (
        pd.merge(
            df_cells,
            df_barcode_library[[PREFIX] + barcode_info_cols],
            how="left",
            left_on=BARCODE_1,
            right_on=PREFIX,
        )
        .rename(
            {col: col + "_1" for col in barcode_info_cols}, axis=1
        )  # Rename columns for clarity
        .drop(PREFIX, axis=1)  # Drop the temporary prefix column
    )

    return df_cells


def call_cells_add_UMIs(df_cells, df_UMI, cols=[WELL, TILE, CELL]):
    """Add UMI (Unique Molecular Identifier) information to called cells.
    Args:
        df_cells (DataFrame): DataFrame containing called cells.
        df_UMI (DataFrame): DataFrame containing UMI reads.
        cols (list, optional): List of columns to use to merge DataFrames. Default is [WELL, TILE, CELL].
    Returns:
        DataFrame: df_cells DataFrame with top UMI counts.
    """
    s = (
        df_UMI.drop_duplicates([WELL, TILE, READ])  # Drop duplicate reads
        .groupby(cols)[BARCODE]  # Group by well, tile, and cell, and barcode
        .value_counts()  # Count occurrences of each barcode within each group
        .rename("count")  # Rename the resulting series to 'count'
        .sort_values(ascending=False)  # Sort in descending order
        .reset_index()  # Reset the index
        .groupby(cols)  # Group again by well, tile, and cell
    )
    df_cells_UMI = (
        df_UMI.join(
            s.nth(0)[["well", "tile", "cell", "barcode"]]
            .rename(columns={"barcode": UMI_0})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(0)[["well", "tile", "cell", "count"]]
            .rename(columns={"count": UMI_COUNT_0})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[["well", "tile", "cell", "barcode"]]
            .rename(columns={"barcode": UMI_1})
            .set_index(cols),
            on=cols,
        )
        .join(
            s.nth(1)[["well", "tile", "cell", "count"]]
            .rename(columns={"count": UMI_COUNT_1})
            .set_index(cols),
            on=cols,
        )
        .join(s["count"].sum().rename(UMI_COUNT), on=cols)
        .assign(
            **{
                UMI_COUNT_0: lambda x: x[UMI_COUNT_0].fillna(0).astype(int),
                UMI_COUNT_1: lambda x: x[UMI_COUNT_1].fillna(0).astype(int),
            }
        )
        .drop_duplicates(cols)
        .drop([READ, BARCODE], axis=1)  # drop the read
        .drop([POSITION_I, POSITION_J], axis=1)  # drop the read coordinates
        .filter(regex="^(?!Q_)")  # remove read quality scores
        .query("cell > 0")  # remove reads not in a cell
    )

    cols_to_use = list(df_cells_UMI.columns.difference(df_cells.columns))

    return df_cells.merge(
        df_cells_UMI[cols_to_use + cols], left_on=cols, right_on=cols, how="inner"
    )


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
