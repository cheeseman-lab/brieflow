"""Utility functions for calling cells from sequencing reads.

Supports single-barcode and multi-barcode protocols with per-barcode quality tracking.
"""

import pandas as pd
import numpy as np
import Levenshtein

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
    BARCODE_0,
    POSITION_I,
    POSITION_J,
    UMI_0,
    UMI_1,
    UMI_COUNT,
    UMI_COUNT_0,
    UMI_COUNT_1,
)

COLS = [WELL, TILE, CELL]


def call_cells(
    reads_data,
    df_barcode_library=None,
    q_min=0,
    barcode_col="sgRNA",
    prefix_col=None,
    map_start=None,
    map_end=None,
    prefix_map="prefix_map",
    recomb_start=None,
    recomb_end=None,
    prefix_recomb="prefix_recomb",
    recomb_filter_col=None,
    recomb_q_thresh=0.1,
    sort_calls="peak",
    error_correct=False,
    max_distance=2,
    distance_metric="hamming",
    n_barcodes=2,
    barcode_info_cols=None,
    df_UMI=None,
    **kwargs,
):
    """Call cells from sequencing reads.

    Identifies cell barcodes from reads with support for single/multi-barcode protocols,
    count/peak sorting, error correction, recombination detection, and UMIs.

    Args:
        reads_data: DataFrame with read data (well, tile, cell, read, barcode, peak, Q_min, ...).
        df_barcode_library: Reference barcode library. None for no-library mode.
        q_min: Minimum quality threshold for reads.
        barcode_col: Library column with full barcode sequences (auto-truncation mode).
        prefix_col: Library column with pre-computed prefixes (overrides barcode_col).
        map_start: Starting cycle for mapping barcode, 1-indexed (cycle-based mode).
        map_end: Ending cycle for mapping barcode, 1-indexed inclusive.
        prefix_map: Column name for extracted mapping barcode.
        recomb_start: Starting cycle for recombination barcode.
        recomb_end: Ending cycle for recombination barcode.
        prefix_recomb: Column name for recombination barcode.
        recomb_filter_col: Quality column for filtering recombination calls.
        recomb_q_thresh: Minimum quality for recombination detection.
        sort_calls: "count" (by read frequency) or "peak" (by intensity).
        error_correct: Whether to correct sequencing errors against library.
        max_distance: Maximum edit distance for error correction.
        distance_metric: "hamming" or "levenshtein".
        n_barcodes: Number of ranked barcodes to store per cell. Default 2.
        barcode_info_cols: Library columns to merge (defaults to gene_symbol, gene_id).
        df_UMI: Optional UMI reads DataFrame.
        kwargs: Additional arguments for error correction.

    Returns:
        DataFrame with one row per cell and per-barcode columns:
            cell_barcode_{0,1}, barcode_peak_{0,1}, Q_min_{0,1}, no_recomb_{0,1},
            gene_symbol_{0,1}, etc.
    """
    if n_barcodes < 2:
        print(f"Warning: n_barcodes must be >= 2, got {n_barcodes}. Setting to 2.")
        n_barcodes = 2

    if reads_data is None or reads_data.empty:
        return _get_empty_output()

    # === STEP 1: Extract barcodes based on mode ===

    if map_start is not None and map_end is not None:
        print(f"Using cycle-based extraction: map cycles {map_start}-{map_end}")
        df_reads = prep_multi_reads(
            reads_data,
            map_start=map_start,
            map_end=map_end,
            recomb_start=recomb_start or map_start,
            recomb_end=recomb_end or map_end,
            prefix_map=prefix_map,
            prefix_recomb=prefix_recomb,
        )
        barcode_column = prefix_map
        enable_recomb = recomb_start is not None and prefix_recomb in df_reads.columns
        library_key = prefix_map
    elif prefix_col is not None:
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
        df_reads = reads_data
        barcode_column = BARCODE
        enable_recomb = False
        library_key = PREFIX
        if df_barcode_library is not None:
            prefix_length = len(reads_data.iloc[0].barcode)
            df_barcode_library[PREFIX] = df_barcode_library.apply(
                lambda x: x[barcode_col][:prefix_length], axis=1
            )
            print(
                f"Created prefixes by truncating '{barcode_col}' to length {prefix_length}"
            )

    # === STEP 2: Quality filter ===

    df_reads = df_reads.query("Q_min >= @q_min")

    # === STEP 3: Error correction (requires library) ===

    pre_correct_col = None
    if error_correct and df_barcode_library is not None:
        print("performing error correction")
        pre_correct_col = f"pre_correction_{barcode_column}"
        df_reads[pre_correct_col] = df_reads[barcode_column]
        df_reads[barcode_column] = error_correct_reads(
            df_reads[barcode_column],
            df_barcode_library[library_key],
            max_distance=max_distance,
            distance_metric=distance_metric,
            **kwargs,
        )

    # === STEP 4: Map reads to library ===

    if df_barcode_library is not None:
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
    else:
        df_mapped = df_reads.assign(mapped=True)

    # === STEP 5: Recombination detection ===

    if enable_recomb and prefix_recomb is not None and df_barcode_library is not None:
        recomb_map = df_barcode_library.set_index(library_key)[prefix_recomb].to_dict()
        expected_recomb = df_mapped[barcode_column].map(recomb_map)
        actual_recomb = df_mapped[prefix_recomb]
        both_valid = expected_recomb.notna() & actual_recomb.notna()
        no_recomb = pd.array([np.nan] * len(df_mapped), dtype="boolean")
        no_recomb[both_valid] = (
            expected_recomb[both_valid].values == actual_recomb[both_valid].values
        )
        df_mapped["no_recomb"] = no_recomb
        df_mapped.loc[~df_mapped.mapped, "no_recomb"] = np.nan
        if recomb_filter_col is not None:
            df_mapped.loc[
                df_mapped[recomb_filter_col] < recomb_q_thresh, "no_recomb"
            ] = np.nan
    else:
        df_mapped["no_recomb"] = np.nan

    # === STEP 6: Rank barcodes per cell ===

    df_unique = df_mapped.drop_duplicates([WELL, TILE, READ])

    if sort_calls == "count":
        barcode_counts = (
            df_unique.groupby(COLS + ["mapped", barcode_column])
            .size()
            .rename("count")
            .reset_index()
            .sort_values(["mapped", "count"], ascending=[False, False])
        )
        barcode_ranks = barcode_counts.groupby(COLS)
        total_counts = barcode_counts.groupby(COLS)["count"].sum()
    else:
        barcode_ranks = df_unique.sort_values(
            ["mapped", "peak"], ascending=[False, False]
        ).groupby(COLS)

    # === STEP 7: Build per-barcode output ===

    q_cols = [c for c in df_mapped.columns if c.startswith("Q_")]
    per_read_cols = q_cols + ["no_recomb", "peak", "mapped"]
    if pre_correct_col and pre_correct_col in df_mapped.columns:
        per_read_cols.append(pre_correct_col)

    # Metadata columns to carry (plate, etc.)
    internal_cols = set(
        COLS + [barcode_column, READ, BARCODE, POSITION_I, POSITION_J, "mapped"]
    )
    metadata_cols = [
        c
        for c in df_mapped.columns
        if c not in internal_cols and c not in per_read_cols and c not in q_cols
    ]

    rank_dfs = []
    for rank in range(n_barcodes):
        try:
            rank_row = barcode_ranks.nth(rank).reset_index(drop=True)
        except (IndexError, ValueError):
            rank_row = pd.DataFrame()

        if len(rank_row) == 0:
            rank_dfs.append(pd.DataFrame(columns=COLS))
            continue

        available_cols = [c for c in per_read_cols if c in df_mapped.columns]

        if sort_calls == "peak":
            # Peak mode: nth(rank) IS the representative read
            extra_cols = metadata_cols if rank == 0 else []
            keep_cols = list(
                dict.fromkeys(COLS + [barcode_column] + available_cols + extra_cols)
            )
            best_reads = rank_row[[c for c in keep_cols if c in rank_row.columns]]
        else:
            # Count mode: look up best read from df_mapped for this barcode
            winners = rank_row[COLS + [barcode_column]]
            extra_cols = metadata_cols if rank == 0 else []
            select_cols = list(
                dict.fromkeys(COLS + [barcode_column] + available_cols + extra_cols)
            )
            best_reads = (
                df_mapped[select_cols]
                .merge(winners, on=COLS + [barcode_column])
                .sort_values("Q_min", ascending=False)
                .drop_duplicates(COLS)
            )

        # Rename columns with rank suffix
        rename = {
            barcode_column: f"cell_barcode_{rank}",
            "no_recomb": f"no_recomb_{rank}",
            "peak": f"cell_barcode_peak_{rank}",
        }
        for qc in q_cols:
            rename[qc] = f"{qc}_{rank}"

        # Pre-correction barcode (rank 0 only)
        if pre_correct_col and pre_correct_col in best_reads.columns:
            if rank == 0:
                rename[pre_correct_col] = f"pre_correction_{BARCODE_0}"
            else:
                best_reads = best_reads.drop(columns=[pre_correct_col])

        # Barcode count (count mode only)
        if sort_calls == "count":
            winner_counts = barcode_ranks.nth(rank)[COLS + ["count"]].reset_index(
                drop=True
            )
            winner_counts = winner_counts.rename(
                columns={"count": f"cell_barcode_count_{rank}"}
            )
            best_reads = best_reads.merge(winner_counts, on=COLS, how="left")

        best_reads = best_reads.drop(columns=["mapped"], errors="ignore")
        best_reads = best_reads.rename(columns=rename)
        rank_dfs.append(best_reads)

    # === STEP 8: Merge ranks into one-row-per-cell output ===

    df_cells = rank_dfs[0]
    for rank_df in rank_dfs[1:]:
        if len(rank_df) > 0:
            df_cells = df_cells.merge(rank_df, on=COLS, how="left")

    # Mode-specific columns (only output relevant metrics, no NaN placeholders)
    if sort_calls == "count":
        for rank in range(n_barcodes):
            count_col = f"cell_barcode_count_{rank}"
            if count_col in df_cells.columns:
                df_cells[count_col] = df_cells[count_col].fillna(0)
            else:
                df_cells[count_col] = 0
            # Drop peak columns (not meaningful in count mode)
            df_cells = df_cells.drop(
                columns=[f"cell_barcode_peak_{rank}"], errors="ignore"
            )

    # Filter to valid cell IDs
    df_cells = df_cells.query("cell > 0")

    # === STEP 9: Add gene annotations from library ===

    if df_barcode_library is not None:
        if barcode_info_cols is None:
            barcode_info_cols = []
            if GENE_SYMBOL in df_barcode_library.columns:
                barcode_info_cols.append(GENE_SYMBOL)
            if GENE_ID in df_barcode_library.columns:
                barcode_info_cols.append(GENE_ID)
            if not barcode_info_cols and SGRNA in df_barcode_library.columns:
                barcode_info_cols.append(SGRNA)

        for rank in range(n_barcodes):
            barcode_col_name = f"cell_barcode_{rank}"
            if barcode_col_name in df_cells.columns:
                df_cells = (
                    pd.merge(
                        df_cells,
                        df_barcode_library[[library_key] + barcode_info_cols],
                        how="left",
                        left_on=barcode_col_name,
                        right_on=library_key,
                    )
                    .rename({col: f"{col}_{rank}" for col in barcode_info_cols}, axis=1)
                    .drop(library_key, axis=1, errors="ignore")
                )

    # === STEP 10: Add UMIs ===

    if df_UMI is not None:
        df_cells = call_cells_add_UMIs(df_cells, df_UMI, cols=COLS)

    return df_cells


def _get_empty_output():
    """Return empty DataFrame with standardized column names."""
    columns = [
        "cell",
        "tile",
        "well",
        "cell_barcode_0",
        "cell_barcode_1",
        "no_recomb_0",
        "no_recomb_1",
        "Q_min_0",
        "Q_min_1",
        "gene_symbol_0",
        "gene_symbol_1",
        "gene_id_0",
        "gene_id_1",
    ]
    return pd.DataFrame(columns=columns)


# === Utility functions ===


def prep_multi_reads(
    df_reads,
    map_start,
    map_end,
    recomb_start,
    recomb_end,
    prefix_map="prefix_map",
    prefix_recomb="prefix_recomb",
):
    """Extract cycle-specific barcodes for multi-barcode protocols.

    Args:
        df_reads: DataFrame with raw reads (barcode, Q_0, Q_1, ...).
        map_start: Starting cycle for mapping barcode (1-indexed).
        map_end: Ending cycle for mapping barcode (1-indexed, inclusive).
        recomb_start: Starting cycle for recombination barcode.
        recomb_end: Ending cycle for recombination barcode.
        prefix_map: Column name for mapping barcode.
        prefix_recomb: Column name for recombination barcode.

    Returns:
        DataFrame with added prefix_map, prefix_recomb, and Q_recomb columns.
    """
    df = df_reads.copy()

    if df.empty:
        df[prefix_map] = pd.Series(dtype="object")
        df[prefix_recomb] = pd.Series(dtype="object")
        df["Q_recomb"] = pd.Series(dtype="float64")
        return df

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

    df[prefix_map] = df["barcode"].str.slice(map_start - 1, map_end)
    df[prefix_recomb] = df["barcode"].str.slice(recomb_start - 1, recomb_end)

    recomb_cycles = list(range(recomb_start, recomb_end + 1))
    recomb_q_cols = [f"Q_{c - 1}" for c in recomb_cycles]

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


def call_cells_add_UMIs(df_cells, df_UMI, cols=None):
    """Add UMI information to called cells.

    Args:
        df_cells: DataFrame with called cells.
        df_UMI: DataFrame with UMI reads.
        cols: Columns for merging. Default is [well, tile, cell].

    Returns:
        DataFrame with added UMI_0, UMI_1, UMI_count_0, UMI_count_1, UMI_count.
    """
    if cols is None:
        cols = [WELL, TILE, CELL]

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

    Only corrects when there is a unique closest match within max_distance.

    Args:
        reads: Series with reads to correct.
        reference: Series with reference barcodes.
        max_distance: Maximum edit distance for correction.
        distance_metric: "hamming" or "levenshtein".

    Returns:
        Series with corrected reads.
    """
    dist_to_ref = _barcode_distance_matrix(
        reads.to_list(),
        reference.to_list(),
        distance_metric=distance_metric,
    )

    min_dist_to_ref = dist_to_ref.min(axis=1)
    unique_dist = np.array(
        [
            np.sum(dist_to_ref[x] == min_dist_to_ref[x]) == 1
            for x in range(dist_to_ref.shape[0])
        ]
    )

    corrected_subset = unique_dist & (min_dist_to_ref <= max_distance)
    corrected_barcodes = reference.loc[
        dist_to_ref[corrected_subset].argmin(axis=1)
    ].values

    corrected_reads = reads.copy()
    corrected_reads.loc[corrected_subset] = corrected_barcodes
    return corrected_reads


def _barcode_distance_matrix(barcodes_1, barcodes_2=False, distance_metric="hamming"):
    """Calculate distances between two sets of barcodes.

    Args:
        barcodes_1: First list of barcode sequences.
        barcodes_2: Second list, or False to use barcodes_1 for both.
        distance_metric: "hamming" or "levenshtein".

    Returns:
        numpy.ndarray distance matrix.
    """
    if distance_metric == "hamming":
        distance = Levenshtein.hamming
    elif distance_metric == "levenshtein":
        distance = Levenshtein.distance
    else:
        import warnings

        warnings.warn(
            'distance_metric must be "hamming" or "levenshtein" - defaulting to "hamming"'
        )
        distance = Levenshtein.hamming

    if isinstance(barcodes_2, bool):
        barcodes_2 = barcodes_1

    bc_distance_matrix = np.zeros((len(barcodes_1), len(barcodes_2)))
    for a, i in enumerate(barcodes_1):
        for b, j in enumerate(barcodes_2):
            bc_distance_matrix[a, b] = distance(i, j)

    return bc_distance_matrix
