"""Combine per-tile tables (parquet or tsv) into one dtype-normalized DataFrame.

polars reads and concatenates the tiles; pandas takes over if polars rejects the
schema. polars parses CSV floats correctly rounded where pandas' default parser is
1-2 ULP off, so TSV-mode output differs from the old pandas concat by ~1e-12 on
high-precision float columns.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow.parquet as pq

from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import read_parquet


def _tile_files(paths):
    """(path, format) for every non-empty tile file; 0-byte tiles are skipped."""
    files = []
    for p in map(Path, paths):
        if p.exists() and p.stat().st_size > 0:
            files.append((str(p), "parquet" if p.suffix == ".parquet" else "tsv"))
    return files


def _columns(path, fmt):
    """Column names of one tile file, in file order."""
    if fmt == "parquet":
        return pl.scan_parquet(path).collect_schema().names()
    with open(path) as fh:
        return fh.readline().rstrip("\n").split("\t")


def _is_array_column(series):
    """True if the first non-null cell holds a list/ndarray (e.g. sbs_info bounds)."""
    nonnull = series.dropna()
    return len(nonnull) > 0 and isinstance(nonnull.iloc[0], (list, np.ndarray))


def _concat_polars(files):
    """Diagonal concat of the tiles, reproducing pandas' column union and order."""
    # 0-row parquet tiles infer object columns as String, which cannot concat with a
    # typed List column, so they only contribute their column names below.
    kept = [
        (f, fmt)
        for f, fmt in files
        if not (fmt == "parquet" and pq.ParquetFile(f).metadata.num_rows == 0)
    ]
    if not kept:
        raise ValueError("no tile has rows")
    scans = [
        pl.scan_parquet(f)
        if fmt == "parquet"
        else pl.scan_csv(f, separator="\t", infer_schema_length=None)
        for f, fmt in kept
    ]
    df = pl.concat(scans, how="diagonal_relaxed").collect()
    # pandas NaN-fills (float upcast) a column only when a row-contributing tile lacks it.
    kept_columns = [set(_columns(f, fmt)) for f, fmt in kept]
    missing_somewhere = {c for s in kept_columns for c in df.columns if c not in s}
    df = df.with_columns(
        [
            pl.col(c).cast(pl.Float64)
            for c in missing_somewhere
            if df.schema[c].is_integer()
        ]
    )
    # Column order is first appearance across all tiles, as pandas concat does.
    order = list(dict.fromkeys(c for f, fmt in files for c in _columns(f, fmt)))
    result = df.select([c for c in order if c in df.columns]).to_pandas()
    for c in order:  # declared only by a 0-row tile: all-null object, as in pandas
        if c not in result.columns:
            result[c] = pd.Series([None] * len(result), dtype="object")
    return result[order]


def _concat_pandas(files):
    """Per-file pandas reads and concat; None if no tile yields a frame."""
    dfs = []
    for f, fmt in files:
        if fmt == "parquet":
            dfs.append(read_parquet(f))
            continue
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except pd.errors.EmptyDataError:  # empty tile; any other parse error must raise
            pass
    return pd.concat(dfs, ignore_index=True) if dfs else None


def combine_tile_dfs(paths):
    """Concatenate per-tile tables and normalize dtypes.

    Args:
        paths (Iterable[str | Path]): Per-tile parquet or tsv files.

    Returns:
        pandas.DataFrame | None: The combined table, or None if no tile yielded a frame.
    """
    files = _tile_files(paths)
    if not files:
        return None
    try:
        combined = _concat_polars(files)
    except Exception:  # schema/parse mismatch: fall back to pandas
        combined = _concat_pandas(files)
    if combined is None:
        return None

    # List-valued columns are set aside: validate_dtypes would stringify them.
    order = list(combined.columns)
    array_columns = {
        c: combined[c]
        for c in combined.select_dtypes(include="object").columns
        if _is_array_column(combined[c])
    }
    combined = validate_dtypes(combined.drop(columns=list(array_columns)))
    # Numeric-looking object columns (cells no_recomb_*/gene_id_*) become numeric.
    for col in combined.select_dtypes(include="object").columns:
        converted = pd.to_numeric(combined[col], errors="coerce")
        if converted.notna().sum() >= combined[col].notna().sum() * 0.95:
            combined[col] = converted
    for c, series in array_columns.items():
        combined[c] = series
    return combined[order]
