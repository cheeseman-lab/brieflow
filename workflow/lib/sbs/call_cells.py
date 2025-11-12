"""Unified utility functions for calling cells from sequencing reads.

This module supports both single-barcode and multi-barcode protocols with consistent function signatures.
Outputs are standardized across all modes for consistent downstream processing.
"""

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
    # Barcode extraction parameters (mutually exclusive modes)
    barcode_col="sgRNA",  # For auto-truncation mode
    prefix_col=None,  # For pre-computed prefix mode
    map_start=None,  # For cycle-based extraction (multi-barcode)
    map_end=None,
    map_col="prefix_map",  # Column name for mapping barcode
    # Recombination detection (multi-barcode only)
    recomb_start=None,
    recomb_end=None,
    recomb_col="prefix_recomb",  # Column name for recombination barcode
    recomb_filter_col=None,  # Quality column to filter recombination
    recomb_q_thresh=0.1,  # Threshold for recombination quality
    # Sorting and error correction
    sort_calls="peak",  # "peak" or "count"
    error_correct=False,
    max_distance=2,
    distance_metric="hamming",
    # Output customization
    barcode_info_cols=None,  # Defaults to [GENE_SYMBOL, GENE_ID] if available
    # Optional UMI support
    df_UMI=None,
    **kwargs,
):
    """Unified cell calling that supports both single and multi-barcode protocols.

    This function identifies cell barcodes from sequencing reads, with support for:
    - Single-barcode protocols (standard SBS)
    - Multi-barcode protocols (with recombination detection)
    - Count-based or peak-based sorting
    - Error correction with pre-correction tracking
    - UMI information (optional)

    BARCODE EXTRACTION MODES (mutually exclusive):

    1. **Cycle-based mode** (for multi-barcode):
       Specify map_start/map_end (and optionally recomb_start/recomb_end).
       Extracts specific cycle ranges from full barcode via slicing.
       Example: map_start=1, map_end=15, recomb_start=16, recomb_end=30

    2. **Pre-computed prefix mode**:
       Specify prefix_col with pre-computed prefixes in library.
       Useful for cycle-skipping scenarios.
       Example: prefix_col="custom_prefix"

    3. **Auto-truncation mode** (default):
       Uses barcode_col and automatically truncates to match read length.
       Example: barcode_col="sgRNA" (default)

    Args:
        reads_data (DataFrame): DataFrame containing read information with columns:
            well, tile, cell, read, barcode, peak, Q_min, Q_0, Q_1, ...
        df_barcode_library (DataFrame, optional): DataFrame containing barcode library.
            Must contain PREFIX column or map_col for mapping.
        q_min (int, optional): Minimum quality threshold. Default is 0.

        barcode_col (str, optional): Column in library with full sequences for
            auto-truncation. Default is 'sgRNA'.
        prefix_col (str, optional): Column in library with pre-computed prefixes.
            Overrides barcode_col if specified.
        map_start (int, optional): Starting cycle for mapping barcode (1-indexed).
        map_end (int, optional): Ending cycle for mapping barcode (1-indexed, inclusive).
        map_col (str, optional): Name for mapping barcode column. Default is 'prefix_map'.

        recomb_start (int, optional): Starting cycle for recombination (1-indexed).
        recomb_end (int, optional): Ending cycle for recombination (1-indexed, inclusive).
        recomb_col (str, optional): Name for recombination column. Default is 'prefix_recomb'.
        recomb_filter_col (str, optional): Quality column for filtering recombination calls.
        recomb_q_thresh (float, optional): Minimum quality for recombination. Default is 0.1.

        sort_calls (str, optional): Sorting criterion - 'count' or 'peak'. Default is 'peak'.
        error_correct (bool, optional): Whether to perform error correction. Default is False.
        max_distance (int, optional): Maximum edit distance for correction. Default is 2.
        distance_metric (str, optional): 'hamming' or 'levenshtein'. Default is 'hamming'.

        barcode_info_cols (list, optional): Columns from library to merge.
            Defaults to [GENE_SYMBOL, GENE_ID] if available.
        df_UMI (DataFrame, optional): DataFrame with UMI reads for UMI support.

        kwargs: Additional arguments for error correction.

    Returns:
        DataFrame: Standardized output with columns:
            - well, tile, cell
            - Q_min, Q_0, Q_1, ... (quality scores)
            - cell_barcode_0, cell_barcode_1 (ALWAYS named cell_barcode, never sgRNA)
            - peak_0, peak_1 (always present)
            - barcode_count_0, barcode_count_1, barcode_count (if sort_calls="count")
            - no_recomb_0, no_recomb_1 (if recombination enabled, else NaN)
            - pre_correction_cell_barcode_0, pre_correction_cell_barcode_1
              (if error_correct=True)
            - gene_symbol_0, gene_symbol_1 (if library provided)
            - gene_id_0, gene_id_1 (if available in library)
            - UMI columns (if df_UMI provided)

    Examples:
        # Single-barcode with count sorting (original behavior)
        >>> call_cells(reads, library, barcode_col="sgRNA", sort_calls="count")

        # Single-barcode with peak sorting
        >>> call_cells(reads, library, barcode_col="sgRNA", sort_calls="peak")

        # Multi-barcode with recombination detection
        >>> call_cells(reads, library,
        ...           map_start=1, map_end=15,
        ...           recomb_start=16, recomb_end=30,
        ...           recomb_filter_col="Q_recomb",
        ...           sort_calls="peak")

        # Cycle-based for single-barcode (use full barcode range)
        >>> call_cells(reads, library,
        ...           map_start=1, map_end=15,
        ...           sort_calls="count")
    """
    # Handle empty input
    if reads_data is None or reads_data.empty:
        return _get_empty_output()

    cols = [WELL, TILE, CELL]

    # Step 1: Determine barcode extraction mode and prepare reads
    if map_start is not None and map_end is not None:
        # Cycle-based mode (multi-barcode or explicit cycle specification)
        print(f"Using cycle-based extraction: map cycles {map_start}-{map_end}")
        df_reads = prep_multi_reads(
            reads_data,
            map_start=map_start,
            map_end=map_end,
            recomb_start=recomb_start or map_start,  # Default to map range
            recomb_end=recomb_end or map_end,
            map_col=map_col,
            recomb_col=recomb_col,
        )
        barcode_column = map_col
        enable_recomb = recomb_start is not None and recomb_col in df_reads.columns
        library_key = map_col

    elif prefix_col is not None:
        # Pre-computed prefix mode
        if (
            df_barcode_library is not None
            and prefix_col not in df_barcode_library.columns
        ):
            raise ValueError(f"Column '{prefix_col}' not found in barcode library")
        print(f"Using pre-computed prefixes from '{prefix_col}' column")
        df_reads = reads_data
        barcode_column = BARCODE
        enable_recomb = False
        library_key = PREFIX
        if df_barcode_library is not None:
            df_barcode_library[PREFIX] = df_barcode_library[prefix_col]

    else:
        # Auto-truncation mode (original behavior)
        df_reads = reads_data
        barcode_column = BARCODE
        enable_recomb = False
        library_key = PREFIX
        if df_barcode_library is not None:
            # Determine experimental prefix length from first read
            prefix_length = len(reads_data.iloc[0].barcode)
            df_barcode_library[PREFIX] = df_barcode_library.apply(
                lambda x: x[barcode_col][:prefix_length], axis=1
            )
            print(
                f"Created prefixes by truncating '{barcode_col}' to length {prefix_length}"
            )

    # Step 2: Quality filter
    df_reads = df_reads.query("Q_min >= @q_min")

    # Step 3: Call cells
    if df_barcode_library is None:
        # No reference library
        df_cells = _call_cells_no_ref(df_reads, barcode_column, sort_calls=sort_calls)

    else:
        # With reference library
        # Set default barcode_info_cols if not specified
        if barcode_info_cols is None:
            # Default to gene_symbol and gene_id if available
            barcode_info_cols = []
            if GENE_SYMBOL in df_barcode_library.columns:
                barcode_info_cols.append(GENE_SYMBOL)
            if GENE_ID in df_barcode_library.columns:
                barcode_info_cols.append(GENE_ID)
            # Fallback to sgRNA if no gene columns
            if not barcode_info_cols and SGRNA in df_barcode_library.columns:
                barcode_info_cols.append(SGRNA)

        df_cells = _call_cells_mapping(
            df_reads,
            df_barcode_library,
            barcode_column=barcode_column,
            library_key=library_key,
            barcode_info_cols=barcode_info_cols,
            enable_recomb=enable_recomb,
            recomb_col=recomb_col if enable_recomb else None,
            recomb_filter_col=recomb_filter_col,
            recomb_q_thresh=recomb_q_thresh,
            error_correct=error_correct,
            sort_calls=sort_calls,
            max_distance=max_distance,
            distance_metric=distance_metric,
            **kwargs,
        )

    # Step 4: Add UMI information if provided
    if df_UMI is not None:
        df_cells = call_cells_add_UMIs(df_cells, df_UMI, cols=cols)

    return df_cells


def _get_empty_output():
    """Return empty DataFrame with standardized column names."""
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
        "barcode_count_0",
        "barcode_count_1",
        "barcode_count",
        "no_recomb_0",
        "no_recomb_1",
        "gene_symbol_0",
        "gene_symbol_1",
        "gene_id_0",
        "gene_id_1",
    ]
    return pd.DataFrame(columns=columns)


def _call_cells_no_ref(df_reads, barcode_column, sort_calls="peak"):
    """Call cells without reference library.

    Supports both count-based and peak-based sorting.

    Args:
        df_reads (DataFrame): Filtered read data
        barcode_column (str): Name of barcode column to use
        sort_calls (str): "count" or "peak"

    Returns:
        DataFrame: Cell calls with standardized column names
    """
    cols = [WELL, TILE, CELL]

    if sort_calls == "count":
        # Count-based sorting (original behavior)
        s = (
            df_reads.drop_duplicates([WELL, TILE, READ])
            .groupby(cols)[barcode_column]
            .value_counts()
            .rename("count")
            .sort_values(ascending=False)
            .reset_index()
            .groupby(cols)
        )

        df_cells = (
            df_reads.join(
                s.nth(0)[["well", "tile", "cell", barcode_column]]
                .rename(columns={barcode_column: BARCODE_0})
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
                s.nth(1)[["well", "tile", "cell", barcode_column]]
                .rename(columns={barcode_column: BARCODE_1})
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
            .drop([READ, BARCODE, barcode_column], axis=1, errors="ignore")
            .drop([POSITION_I, POSITION_J], axis=1, errors="ignore")
            .filter(regex="^(?!Q_)")
            .query("cell > 0")
        )

        # Add peak columns as NaN (not available in count mode)
        df_cells["peak_0"] = np.nan
        df_cells["peak_1"] = np.nan

    else:
        # Peak-based sorting (multi behavior)
        s = df_reads.sort_values("peak", ascending=False).groupby(cols)

        df_cells = (
            df_reads.join(
                s.nth(0)[cols + [barcode_column, "peak"]]
                .rename(columns={barcode_column: BARCODE_0, "peak": "peak_0"})
                .set_index(cols),
                on=cols,
            )
            .join(
                s.nth(1)[cols + [barcode_column, "peak"]]
                .rename(columns={barcode_column: BARCODE_1, "peak": "peak_1"})
                .set_index(cols),
                on=cols,
            )
            .drop_duplicates(cols)
            .drop([READ, BARCODE, barcode_column, "peak"], axis=1, errors="ignore")
            .drop([POSITION_I, POSITION_J], axis=1, errors="ignore")
            .filter(regex="^(?!Q_)")
            .query("cell > 0")
        )

        # Add count columns as NaN (not calculated in peak mode)
        df_cells[BARCODE_COUNT_0] = np.nan
        df_cells[BARCODE_COUNT_1] = np.nan
        df_cells[BARCODE_COUNT] = np.nan

    # Add recombination columns as NaN (no library to detect recombination)
    df_cells["no_recomb_0"] = np.nan
    df_cells["no_recomb_1"] = np.nan

    return df_cells


def _call_cells_mapping(
    df_reads,
    df_barcode_library,
    barcode_column,
    library_key,
    barcode_info_cols,
    enable_recomb,
    recomb_col,
    recomb_filter_col,
    recomb_q_thresh,
    error_correct,
    sort_calls,
    max_distance,
    distance_metric,
    **kwargs,
):
    """Call cells with reference library mapping.

    Supports:
    - Error correction with pre-correction tracking
    - Recombination detection
    - Both count and peak sorting
    - Configurable barcode info columns

    Args:
        df_reads (DataFrame): Filtered read data
        df_barcode_library (DataFrame): Reference barcode library
        barcode_column (str): Name of barcode column in reads
        library_key (str): Name of key column in library (PREFIX or map_col)
        barcode_info_cols (list): Columns to merge from library
        enable_recomb (bool): Whether to perform recombination detection
        recomb_col (str): Name of recombination column
        recomb_filter_col (str): Quality column for filtering recombination
        recomb_q_thresh (float): Quality threshold for recombination
        error_correct (bool): Whether to perform error correction
        sort_calls (str): "count" or "peak"
        max_distance (int): Max edit distance for error correction
        distance_metric (str): "hamming" or "levenshtein"
        kwargs: Additional arguments for error correction

    Returns:
        DataFrame: Cell calls with gene info and standardized columns
    """
    cols = [WELL, TILE, CELL]
    pre_correct_col = None

    # Optional error correction
    if error_correct:
        print("performing error correction")
        pre_correct_col = f"pre_correction_{barcode_column}"
        # Store original values before correction
        df_reads[pre_correct_col] = df_reads[barcode_column]
        # Perform error correction
        df_reads[barcode_column] = error_correct_reads(
            df_reads[barcode_column],
            df_barcode_library[library_key],
            max_distance=max_distance,
            distance_metric=distance_metric,
            **kwargs,
        )

    # Map reads to library
    df_barcode_library["_temp_key"] = df_barcode_library[library_key]
    df_mapped = (
        pd.merge(
            df_reads,
            df_barcode_library[["_temp_key"]],
            how="left",
            left_on=barcode_column,
            right_on="_temp_key",
        )
        .assign(mapped=lambda x: pd.notnull(x["_temp_key"]))
        .drop("_temp_key", axis=1)
    )

    # Optional recombination detection
    if enable_recomb and recomb_col is not None:
        # Create mapping of expected recombination values
        recomb_map = df_barcode_library.set_index(library_key)[recomb_col].to_dict()
        # Flag sequences where actual matches expected
        df_mapped["no_recomb"] = (df_mapped[barcode_column].map(recomb_map)) == (
            df_mapped[recomb_col]
        )
        # Unmapped cells have undetermined recombination status
        df_mapped.loc[~df_mapped.mapped, "no_recomb"] = np.nan
        # Drop the recomb barcode column (we only need the boolean)
        df_mapped = df_mapped.drop(columns=[recomb_col], errors="ignore")

        # Apply quality threshold for recombination status if specified
        if recomb_filter_col is not None:
            df_mapped.loc[
                df_mapped[recomb_filter_col] < recomb_q_thresh, "no_recomb"
            ] = np.nan
    else:
        # No recombination detection
        df_mapped["no_recomb"] = np.nan

    # Sort by count or peak
    if sort_calls == "count":
        # Count-based sorting
        s = (
            df_mapped.drop_duplicates([WELL, TILE, READ])
            .groupby(cols + ["mapped"])[barcode_column]
            .value_counts()
            .rename("count")
            .reset_index()
            .sort_values(["mapped", "count"], ascending=False)
            .groupby(cols)
        )

        # Build output with counts
        if error_correct and pre_correct_col:
            # Include pre-correction values
            df_cells = (
                df_reads.join(
                    s.nth(0)[["well", "tile", "cell", barcode_column]]
                    .rename(columns={barcode_column: BARCODE_0})
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
                    s.nth(1)[["well", "tile", "cell", barcode_column]]
                    .rename(columns={barcode_column: BARCODE_1})
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
            )
        else:
            df_cells = (
                df_reads.join(
                    s.nth(0)[["well", "tile", "cell", barcode_column]]
                    .rename(columns={barcode_column: BARCODE_0})
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
                    s.nth(1)[["well", "tile", "cell", barcode_column]]
                    .rename(columns={barcode_column: BARCODE_1})
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
            )

        df_cells = df_cells.assign(
            **{
                BARCODE_COUNT_0: lambda x: x[BARCODE_COUNT_0].fillna(0),
                BARCODE_COUNT_1: lambda x: x[BARCODE_COUNT_1].fillna(0),
            }
        )

        # Add peak as NaN (not calculated in count mode)
        df_cells["peak_0"] = np.nan
        df_cells["peak_1"] = np.nan

    else:
        # Peak-based sorting
        s = (
            df_mapped.drop_duplicates([WELL, TILE, READ])
            .sort_values(["mapped", "peak"], ascending=[False, False])
            .groupby(cols)
        )

        # Build output with peaks and recombination info
        if error_correct and pre_correct_col:
            # Include pre-correction values
            df_cells = df_reads.join(
                s.nth(0)[cols + [barcode_column, pre_correct_col, "no_recomb", "peak"]]
                .rename(
                    columns={
                        barcode_column: BARCODE_0,
                        pre_correct_col: f"pre_correction_{BARCODE_0}",
                        "no_recomb": "no_recomb_0",
                        "peak": "peak_0",
                    }
                )
                .set_index(cols),
                on=cols,
            ).join(
                s.nth(1)[cols + [barcode_column, "no_recomb", "peak"]]
                .rename(
                    columns={
                        barcode_column: BARCODE_1,
                        "no_recomb": "no_recomb_1",
                        "peak": "peak_1",
                    }
                )
                .set_index(cols),
                on=cols,
            )
        else:
            df_cells = df_reads.join(
                s.nth(0)[cols + [barcode_column, "no_recomb", "peak"]]
                .rename(
                    columns={
                        barcode_column: BARCODE_0,
                        "no_recomb": "no_recomb_0",
                        "peak": "peak_0",
                    }
                )
                .set_index(cols),
                on=cols,
            ).join(
                s.nth(1)[cols + [barcode_column, "no_recomb", "peak"]]
                .rename(
                    columns={
                        barcode_column: BARCODE_1,
                        "no_recomb": "no_recomb_1",
                        "peak": "peak_1",
                    }
                )
                .set_index(cols),
                on=cols,
            )

        # Add count columns as NaN (not calculated in peak mode)
        df_cells[BARCODE_COUNT_0] = np.nan
        df_cells[BARCODE_COUNT_1] = np.nan
        df_cells[BARCODE_COUNT] = np.nan

    # Clean up temporary columns
    df_cells = (
        df_cells.drop_duplicates(cols)
        .drop([READ, BARCODE, barcode_column], axis=1, errors="ignore")
        .drop([POSITION_I, POSITION_J], axis=1, errors="ignore")
        .drop(["no_recomb"], axis=1, errors="ignore")  # Already split into _0 and _1
    )

    # Remove pre-correction temp column if it exists
    if pre_correct_col and pre_correct_col in df_cells.columns:
        df_cells = df_cells.drop([pre_correct_col], axis=1, errors="ignore")

    # Filter to cells only
    df_cells = df_cells.query("cell > 0")

    # Merge barcode info from library
    # Merge for barcode 0
    df_cells = (
        pd.merge(
            df_cells,
            df_barcode_library[[library_key] + barcode_info_cols],
            how="left",
            left_on=BARCODE_0,
            right_on=library_key,
        )
        .rename({col: col + "_0" for col in barcode_info_cols}, axis=1)
        .drop(library_key, axis=1, errors="ignore")
    )

    # Merge for barcode 1
    df_cells = (
        pd.merge(
            df_cells,
            df_barcode_library[[library_key] + barcode_info_cols],
            how="left",
            left_on=BARCODE_1,
            right_on=library_key,
        )
        .rename({col: col + "_1" for col in barcode_info_cols}, axis=1)
        .drop(library_key, axis=1, errors="ignore")
    )

    return df_cells


def prep_multi_reads(
    df_reads,
    map_start,
    map_end,
    recomb_start,
    recomb_end,
    map_col="prefix_map",
    recomb_col="prefix_recomb",
):
    """Prepare reads for multi-barcode calling by extracting cycle-specific barcodes.

    This function extracts specific cycle ranges from the full barcode sequence
    to create separate mapping and recombination barcodes. It also computes
    quality scores for the recombination region.

    Args:
        df_reads (DataFrame): DataFrame containing raw sequencing reads with columns:
            barcode, Q_0, Q_1, Q_2, ... (per-cycle quality scores)
        map_start (int): Starting cycle number for mapping barcode (1-indexed)
        map_end (int): Ending cycle number for mapping barcode (1-indexed, inclusive)
        recomb_start (int): Starting cycle for recombination barcode (1-indexed)
        recomb_end (int): Ending cycle for recombination barcode (1-indexed, inclusive)
        map_col (str, optional): Name for mapping barcode column. Default is 'prefix_map'
        recomb_col (str, optional): Name for recombination column. Default is 'prefix_recomb'

    Returns:
        DataFrame: Input DataFrame with added columns:
            - map_col: Extracted mapping barcode
            - recomb_col: Extracted recombination barcode
            - Q_recomb: Minimum quality score across recombination cycles

    Example:
        >>> df = prep_multi_reads(reads, map_start=1, map_end=15,
        ...                       recomb_start=16, recomb_end=30)
        >>> # Now df has columns: prefix_map, prefix_recomb, Q_recomb
    """
    # Make a copy to avoid modifying the original
    df = df_reads.copy()

    # Handle empty DataFrame
    if df.empty:
        print(
            "Warning: DataFrame is empty, returning empty DataFrame with required columns"
        )
        df[map_col] = pd.Series(dtype="object")
        df[recomb_col] = pd.Series(dtype="object")
        df["Q_recomb"] = pd.Series(dtype="float64")
        return df

    # Check available quality columns
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

    # Extract mapping barcode from specified cycles (adjust for 0-indexing)
    df[map_col] = df["barcode"].str.slice(map_start - 1, map_end)

    # Extract recombination barcode from specified cycles (adjust for 0-indexing)
    df[recomb_col] = df["barcode"].str.slice(recomb_start - 1, recomb_end)

    # Compute quality score for recombination region
    recomb_cycles = list(range(recomb_start, recomb_end + 1))
    recomb_q_cols = [f"Q_{c - 1}" for c in recomb_cycles]

    # Check if all required quality columns exist
    missing_cols = [col for col in recomb_q_cols if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing quality columns: {missing_cols}")
        available_cols = [col for col in recomb_q_cols if col in df.columns]
        if available_cols:
            df["Q_recomb"] = df[available_cols].min(axis=1)
        else:
            df["Q_recomb"] = pd.Series([np.nan] * len(df))
    else:
        df["Q_recomb"] = df[recomb_q_cols].min(axis=1)

    return df


def call_cells_add_UMIs(df_cells, df_UMI, cols=[WELL, TILE, CELL]):
    """Add UMI (Unique Molecular Identifier) information to called cells.

    Args:
        df_cells (DataFrame): DataFrame containing called cells
        df_UMI (DataFrame): DataFrame containing UMI reads with same structure as regular reads
        cols (list, optional): Columns for merging. Default is [WELL, TILE, CELL]

    Returns:
        DataFrame: df_cells with added UMI columns:
            - UMI_0, UMI_count_0: Top UMI and its count
            - UMI_1, UMI_count_1: Second UMI and its count
            - UMI_count: Total UMI count
    """
    s = (
        df_UMI.drop_duplicates([WELL, TILE, READ])
        .groupby(cols)[BARCODE]
        .value_counts()
        .rename("count")
        .sort_values(ascending=False)
        .reset_index()
        .groupby(cols)
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
        .drop([READ, BARCODE], axis=1, errors="ignore")
        .drop([POSITION_I, POSITION_J], axis=1, errors="ignore")
        .filter(regex="^(?!Q_)")
        .query("cell > 0")
    )

    cols_to_use = list(df_cells_UMI.columns.difference(df_cells.columns))

    return df_cells.merge(
        df_cells_UMI[cols_to_use + cols], left_on=cols, right_on=cols, how="inner"
    )


def error_correct_reads(reads, reference, max_distance=2, distance_metric="hamming"):
    """Error correct reads against a reference set of barcodes.

    Compares each read to the reference set and corrects it to the closest unique
    reference if within the specified distance threshold.

    Args:
        reads (pd.Series): Series with reads for error correction
        reference (pd.Series): Series with reference sequences
        max_distance (int, optional): Maximum distance for correction. Correction
            is performed only if (1) one reference sequence is closest (no ties)
            and (2) that unique reference is within this distance. Default is 2.
        distance_metric (str, optional): Distance metric to compare barcodes.
            Options are 'hamming' (default) and 'levenshtein'.

    Returns:
        pd.Series: Corrected reads (unchanged reads returned as-is)
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
