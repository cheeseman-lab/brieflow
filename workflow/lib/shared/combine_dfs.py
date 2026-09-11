"""Combine per-tile intermediates (parquet-or-tsv, prefer parquet) into one
dtype-normalized pandas DataFrame.

Used by the Snakemake combine script (workflow/scripts/shared/combine_dfs.py)
so the concat + dtype-normalization logic lives in exactly one place for every
caller.

NOTE on float parity: polars parses CSV floats with correct round-to-nearest
(matching pandas float_precision="round_trip" and Python float()). pandas'
DEFAULT read_csv parser is imprecise (xstrtod), so the fast path differs from
the old pandas-default output by up to ~5e-13 on high-precision float columns
(e.g. sbs_info i/j, cells Q_*). Values are numerically equivalent, NOT byte
identical. See _READ_STATS to confirm which backend ran.

DECISION (Phase 1, Option A, 2026-08-28): we ACCEPT this ~1e-12 difference.
polars is the more-correct parser; downstream analysis is unaffected. Validated
both ways on real plate-4 well A1: round_trip==polars EXACT, and default-vs-
polars max |diff| 1.8e-12 (reads) / 9.1e-13 (cells) / 2.3e-13 (sbs_info), all
<< rtol=1e-9. Identity tests compare float columns with np.allclose(rtol=1e-9)
and everything else byte-exactly (see tests/sbs/test_combine_dfs_opt.py).

Phase 2 (2026-08-29): inputs are now parquet-or-tsv per tile (prefer parquet
via resolve_table_path). Parquet round-trips floats exactly, so parquet-mode
combine is EXACT to the in-memory frame and stays within Option A vs the
TSV-mode golden. Production TSV-only wells still combine unchanged.
"""

import os

import numpy as np
import pandas as pd

from lib.shared.file_utils import validate_dtypes
from lib.shared.parquet_io import resolve_table_path, read_parquet

try:
    import polars as pl

    _HAS_POLARS = True
except ImportError:
    _HAS_POLARS = False

# Backend counter so tests can assert the polars fast path actually ran.
_READ_STATS = {"polars": 0, "pandas_fallback": 0}


def _col_list(f, fmt):
    """Ordered column names for a resolved tile file, by format."""
    if fmt == "parquet":
        return list(pl.scan_parquet(f).collect_schema().names())
    with open(f) as fh:
        return fh.readline().rstrip("\n").split("\t")


def _read_concat(paths):
    """Fast multithreaded read + vertical (union) concat of per-tile tables.

    Each path is resolved to its parquet-else-tsv sibling; scan_parquet/scan_csv
    is chosen per file. Uses one polars diagonal_relaxed concat over per-file
    lazy scans, which
    reproduces the pandas concat column UNION (missing column -> null). 0-byte
    files are pre-filtered (pandas skips them via EmptyDataError; scan_csv can
    choke on them); header-only 0-row files are kept so their columns still
    join the union, matching pandas. Any polars error (schema/parse) falls
    back to per-file pandas reads + concat. Returns None only when no file
    yields a frame (mirrors the old `if not dfs` guard).
    """
    resolved = []
    for p_in in paths:
        f, fmt = resolve_table_path(p_in)
        if f is not None:
            resolved.append((f, fmt))

    if _HAS_POLARS and resolved:
        try:
            import pyarrow.parquet as pq

            def _is_empty(f, fmt):
                # Skip ONLY 0-row parquet tiles: their object columns infer as
                # String, which poisons a List-typed column (sbs_info "bounds")
                # in the diagonal concat (polars cannot cast List <-> String, so
                # the whole fast path throws). Skipping them lets the parquet
                # fast path run and preserves bounds as a native List(Int64).
                # Empty TSV tiles are deliberately NOT skipped: leaving them in
                # makes the fast path poison (empty String cols vs typed data)
                # and fall back to per-file pandas, which reproduces the ORIGINAL
                # TSV combine's dtypes exactly (a 0-row TSV tile's object cols
                # -> Int64 via validate_dtypes). Parquet tiles are typed, so no
                # such upcast -- hence parquet mode yields clean int64.
                return fmt == "parquet" and pq.ParquetFile(f).metadata.num_rows == 0

            kept = [(fp, fm) for fp, fm in resolved if not _is_empty(fp, fm)]
            if not kept:
                raise ValueError("no non-empty frames")  # -> pandas fallback
            lfs = [
                pl.scan_parquet(fp)
                if fm == "parquet"
                else pl.scan_csv(fp, separator="\t", infer_schema_length=None)
                for fp, fm in kept
            ]
            df = pl.concat(lfs, how="diagonal_relaxed").collect()
            # A column is NaN-filled (float upcast) in pandas only when a
            # ROW-CONTRIBUTING tile lacks it; a 0-row tile injects no NaN. So
            # compute union_missing from the non-empty (kept) tiles only, else a
            # header-only tile missing e.g. `plate` would wrongly upcast it.
            kept_cols = [set(_col_list(fp, fm)) for fp, fm in kept]
            union_missing = {
                c for s in kept_cols for c in df.columns if c not in s
            }
            casts = [
                pl.col(c).cast(pl.Float64)
                for c in union_missing
                if df.schema[c].is_integer()
            ]
            if casts:
                df = df.with_columns(casts)
            # Column order == pandas first-appearance order across ALL tiles
            # (empty tiles still contribute their columns/order in pandas).
            file_col_lists = [_col_list(fp, fm) for fp, fm in resolved]
            ordered_union = list(
                dict.fromkeys(c for cols in file_col_lists for c in cols)
            )
            df = df.select([c for c in ordered_union if c in df.columns])
            result = df.to_pandas()
            # A column declared only by a skipped (0-row) tile -- e.g. a stale,
            # fuller schema in an empty-FOV placeholder -- is absent from the
            # data concat. pandas keeps it as an all-null object column; mirror
            # that so the union is complete and byte-identical to the reference.
            missing = [c for c in ordered_union if c not in result.columns]
            for c in missing:
                result[c] = pd.Series([None] * len(result), dtype="object")
            if missing:
                result = result[ordered_union]
            _READ_STATS["polars"] += 1
            return result
        except Exception:
            pass  # schema/parse error -> pandas fallback below

    dfs = []
    for f, fmt in resolved:
        if fmt == "parquet":
            dfs.append(read_parquet(f))  # resolver already filtered 0-byte
            continue
        try:
            dfs.append(pd.read_csv(f, sep="\t"))
        except pd.errors.EmptyDataError:
            pass  # skip empty tiles ONLY, matching the original get_file semantics.
            # Any other parse error (truncated/corrupt tile) must propagate and
            # fail the rule loud -- silently dropping a tile is data loss.
    if not dfs:
        return None
    _READ_STATS["pandas_fallback"] += 1
    return pd.concat(dfs, ignore_index=True)


def combine_tile_dfs(paths):
    """Read per-tile TSVs, concat, and normalize dtypes.

    Returns a pandas DataFrame with the same dtype normalization the previous
    pandas-only implementation produced (validate_dtypes, then a 95%-threshold
    numeric coercion of leftover object columns -- kept because it genuinely
    fires on real data, e.g. cells no_recomb_*/gene_id_*). Returns None if no
    input file yielded a frame, so callers can skip writing.
    """
    combined = _read_concat(paths)
    if combined is None:
        return None

    # Columns holding a per-row array/list (e.g. sbs_info "bounds", a regionprops
    # bbox that comes out of the parquet fast path as an ndarray) are not scalar:
    # validate_dtypes' astype("string") would flatten each cell to its ndarray
    # repr, and the to_numeric sweep can't touch it. Set such columns aside,
    # normalize the scalar columns, then restore them unchanged so write_parquet
    # stores them as a native List column instead of a mangled string.
    orig_order = list(combined.columns)
    array_cols = {}
    for c in combined.select_dtypes(include="object").columns:
        nonnull = combined[c].dropna()
        if len(nonnull) and isinstance(nonnull.iloc[0], (list, np.ndarray)):
            array_cols[c] = combined[c]
    if array_cols:
        combined = combined.drop(columns=list(array_cols))

    combined = validate_dtypes(combined)
    for col in combined.select_dtypes(include="object").columns:
        converted = pd.to_numeric(combined[col], errors="coerce")
        if converted.notna().sum() >= combined[col].notna().sum() * 0.95:
            combined[col] = converted

    if array_cols:
        for c, series in array_cols.items():
            combined[c] = series
        combined = combined[orig_order]

    if os.environ.get("COMBINE_DFS_DEBUG"):
        import sys

        print(f"[combine_dfs] backends={_READ_STATS}", file=sys.stderr)

    return combined
