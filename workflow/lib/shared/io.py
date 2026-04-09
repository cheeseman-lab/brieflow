"""Fast parquet I/O using polars with pandas compatibility.

Drop-in replacements for pd.read_parquet / df.to_parquet that use polars
under the hood for faster reads and writes. All functions accept and return
pandas DataFrames so downstream code is unchanged.

Usage:
    from lib.shared.io import read_parquet, write_parquet, read_parquets

    # Single file read (returns pandas DataFrame)
    df = read_parquet("data.parquet")
    df = read_parquet("data.parquet", columns=["gene", "feature_1"])

    # Multiple file concat (replaces pd.concat([pd.read_parquet(p) for p in paths]))
    df = read_parquets(paths)
    df = read_parquets(paths, columns=["gene", "feature_1"])

    # Write (accepts pandas DataFrame)
    write_parquet(df, "output.parquet")
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import pandas as pd

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False


def read_parquet(
    path: Union[str, Path],
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read a parquet file, returning a pandas DataFrame.

    Uses polars lazy scan for speed when available, falls back to pandas.
    """
    if _HAS_POLARS:
        try:
            lf = pl.scan_parquet(path)
            if columns is not None:
                lf = lf.select(columns)
            return lf.collect().to_pandas()
        except (pl.exceptions.SchemaError, pl.exceptions.ComputeError):
            # Mixed types in columns — fall back to pandas which is more lenient
            return pd.read_parquet(path, columns=columns)
    else:
        return pd.read_parquet(path, columns=columns)


def read_parquets(
    paths: Sequence[Union[str, Path]],
    columns: Optional[list[str]] = None,
) -> pd.DataFrame:
    """Read and concatenate multiple parquet files into one pandas DataFrame.

    Replaces the common pattern:
        pd.concat([pd.read_parquet(p) for p in paths], ignore_index=True)
    """
    if not paths:
        return pd.DataFrame()

    if _HAS_POLARS:
        try:
            lf = pl.scan_parquet(paths)
            if columns is not None:
                lf = lf.select(columns)
            return lf.collect().to_pandas()
        except pl.exceptions.SchemaError:
            # Schema mismatch across files — fall back to per-file reads
            dfs = [read_parquet(p, columns=columns) for p in paths]
            return pd.concat(dfs, ignore_index=True)
    else:
        dfs = [pd.read_parquet(p, columns=columns) for p in paths]
        return pd.concat(dfs, ignore_index=True)


def write_parquet(
    df: pd.DataFrame,
    path: Union[str, Path],
) -> None:
    """Write a pandas DataFrame to parquet.

    Uses polars for faster writes when available, falls back to pandas.
    """
    if _HAS_POLARS:
        pl.from_pandas(df).write_parquet(str(path))
    else:
        df.to_parquet(path, index=False)
