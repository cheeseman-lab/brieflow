"""Tests for the streaming final merge that attaches CP features to the dedup merge.

final_merge replaces a pandas full-load .merge() with a polars
scan_parquet -> left join -> sink_parquet so the ~3,600-col x ~1M-row phenotype
table never has to fit in memory. The join contract has to survive that swap:

1. Every deduplicated row is preserved, with null CP columns where no phenotype
   row matched, and unmatched phenotype rows are dropped.
2. Column set and order match what the pandas path produced -- dedup columns,
   then CP feature columns with 'label' renamed to 'cell_0'.
3. Key dtypes are reconciled before the join. Phenotype parquets can carry
   plate/tile as String while the dedup side has them as Int64; polars does not
   coerce across that, so an unreconciled join silently matches zero rows and
   writes an all-null feature table instead of raising.
4. exclude_markers drops the requested marker columns, and the stitch approach
   renames the dedup coordinate columns to their global names.
"""

import sys
from pathlib import Path

import polars as pl
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.merge.final_merge import final_merge  # noqa: E402


@pytest.fixture
def paths(tmp_path):
    return (
        tmp_path / "dedup.parquet",
        tmp_path / "phenotype_cp.parquet",
        tmp_path / "merge_final.parquet",
    )


def write_dedup(path, **extra):
    """Dedup side: Int64 plate/tile, one row (cell_0=11) with no phenotype match."""
    pl.DataFrame(
        {
            "plate": [4, 4, 4],
            "well": ["A1"] * 3,
            "tile": [1, 1, 2],
            "cell_0": [10, 11, 20],
            "i_0": [0.0, 1.0, 2.0],
            **extra,
        }
    ).write_parquet(path)


def write_cp(path, **extra):
    """CP side: String plate/tile (the real-world dtype mismatch), one extra row."""
    pl.DataFrame(
        {
            "plate": ["4", "4", "4"],
            "well": ["A1"] * 3,
            "tile": ["1", "2", "2"],
            "label": [10, 20, 99],
            "feat_x": [1.5, 2.5, 9.9],
            **extra,
        }
    ).write_parquet(path)


def test_left_join_preserves_dedup_rows_and_drops_unmatched_cp(paths):
    dedup, cp, out = paths
    write_dedup(dedup)
    write_cp(cp)

    final_merge(dedup, cp, out)
    result = pl.read_parquet(out)

    assert result.height == 3  # all dedup rows kept, CP row 99 dropped
    matched = dict(zip(result["cell_0"], result["feat_x"]))
    assert matched[10] == 1.5
    assert matched[20] == 2.5
    assert matched[11] is None  # no phenotype row -> null feature


def test_column_set_and_order_match_the_pandas_path(paths):
    dedup, cp, out = paths
    write_dedup(dedup)
    write_cp(cp)

    final_merge(dedup, cp, out)

    assert pl.read_parquet(out).columns == [
        "plate",
        "well",
        "tile",
        "cell_0",
        "i_0",
        "feat_x",
    ]


def test_string_keys_are_cast_rather_than_silently_matching_nothing(paths):
    """Without the key cast this join returns 3 rows of all-null features, not an error."""
    dedup, cp, out = paths
    write_dedup(dedup)
    write_cp(cp)

    final_merge(dedup, cp, out)
    result = pl.read_parquet(out)

    assert result["feat_x"].null_count() == 1  # only the genuinely unmatched row


def test_exclude_markers_drops_matching_feature_columns(paths):
    dedup, cp, out = paths
    write_dedup(dedup)
    write_cp(cp, cell_DAPI_mean=[1.0, 2.0, 3.0], cell_GFP_mean=[4.0, 5.0, 6.0])

    final_merge(dedup, cp, out, exclude_markers=["DAPI"])
    columns = pl.read_parquet(out).columns

    assert "cell_DAPI_mean" not in columns
    assert "cell_GFP_mean" in columns


def test_stitch_approach_renames_dedup_coordinates_to_global(paths):
    dedup, cp, out = paths
    write_dedup(dedup, j_0=[3.0, 4.0, 5.0])
    write_cp(cp)

    final_merge(dedup, cp, out, approach="stitch")
    columns = pl.read_parquet(out).columns

    assert "global_i_0" in columns and "global_j_0" in columns
    assert "i_0" not in columns and "j_0" not in columns


def test_fast_approach_leaves_coordinates_alone(paths):
    dedup, cp, out = paths
    write_dedup(dedup)
    write_cp(cp)

    final_merge(dedup, cp, out, approach="fast")

    assert "i_0" in pl.read_parquet(out).columns
