"""Utility functions for calling cells from sequencing reads."""

import pandas as pd

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


def call_cells(reads_data, df_pool=None, q_min=0):
    """Perform median correction independently for each tile.

    Args:
        reads_data (DataFrame): DataFrame containing read information.
        df_pool (DataFrame, optional): DataFrame containing pool information. Default is None.
        q_min (int, optional): Minimum quality threshold. Default is 0.

    Returns:
        DataFrame: DataFrame containing corrected cells.
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

    # Check if df_pool is None
    if df_pool is None:
        # Filter reads by quality threshold and call cells
        return reads_data.query("Q_min >= @q_min").pipe(call_cells_helper)
    else:
        # Determine the experimental prefix length
        prefix_length = len(reads_data.iloc[0].barcode)

        # Add prefix to the pool DataFrame
        df_pool[PREFIX] = df_pool.apply(lambda x: x.sgRNA[:prefix_length], axis=1)

        # Filter reads by quality threshold and call cells mapping
        return reads_data.query("Q_min >= @q_min").pipe(call_cells_mapping, df_pool)


def call_cells_helper(df_reads):
    """Determine the count of top barcodes for each cell.

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


def call_cells_mapping(df_reads, df_pool):
    """Determine the count of top barcodes, with prioritization given to barcodes mapping to the given pool design.

    Args:
        df_reads (DataFrame): DataFrame containing read data.
        df_pool (DataFrame): DataFrame containing pool design information.

    Returns:
        DataFrame: DataFrame containing the count of top barcodes along with merged guide information.
    """
    # Columns related to guide information
    guide_info_cols = [SGRNA, GENE_SYMBOL, GENE_ID]

    # Map reads to the pool design
    df_mapped = (
        pd.merge(
            df_reads, df_pool[[PREFIX]], how="left", left_on=BARCODE, right_on=PREFIX
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
            df_pool[[PREFIX] + guide_info_cols],
            how="left",
            left_on=BARCODE_0,
            right_on=PREFIX,
        )
        .rename(
            {col: col + "_0" for col in guide_info_cols}, axis=1
        )  # Rename columns for clarity
        .drop(PREFIX, axis=1)  # Drop the temporary prefix column
    )
    # Merge guide information for barcode 1
    df_cells = (
        pd.merge(
            df_cells,
            df_pool[[PREFIX] + guide_info_cols],
            how="left",
            left_on=BARCODE_1,
            right_on=PREFIX,
        )
        .rename(
            {col: col + "_1" for col in guide_info_cols}, axis=1
        )  # Rename columns for clarity
        .drop(PREFIX, axis=1)  # Drop the temporary prefix column
    )

    return df_cells
