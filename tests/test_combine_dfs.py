"""combine_tile_dfs must reproduce the pandas concat contract over parquet and tsv tiles."""

import numpy as np
import pandas as pd
import polars as pl
import pytest

from lib.shared.combine_dfs import combine_tile_dfs


def _write(path, df):
    if path.suffix == ".parquet":
        pl.from_pandas(df).write_parquet(path)
    else:
        df.to_csv(path, sep="\t", index=False)
    return path


@pytest.mark.unit
@pytest.mark.parametrize("ext", [".parquet", ".tsv"])
def test_union_of_columns_in_first_appearance_order(tmp_path, ext):
    a = _write(
        tmp_path / f"a{ext}", pd.DataFrame({"well": ["A1"], "tile": [0], "x": [1]})
    )
    b = _write(
        tmp_path / f"b{ext}", pd.DataFrame({"well": ["A1"], "tile": [1], "y": [2.5]})
    )

    out = combine_tile_dfs([a, b])

    assert list(out.columns) == ["well", "tile", "x", "y"]
    assert out["tile"].tolist() == [0, 1]
    assert out["x"].isna().tolist() == [False, True]  # NaN-filled where a tile lacks it


@pytest.mark.unit
def test_zero_row_parquet_tile_contributes_only_its_columns(tmp_path):
    full = _write(
        tmp_path / "a.parquet", pd.DataFrame({"well": ["A1"], "tile": [0], "x": [1]})
    )
    empty = _write(
        tmp_path / "b.parquet", pd.DataFrame({"well": [], "tile": [], "extra": []})
    )

    out = combine_tile_dfs([full, empty])

    assert len(out) == 1
    assert "extra" in out.columns and out["extra"].isna().all()


@pytest.mark.unit
def test_list_column_survives_dtype_normalization(tmp_path):
    bounds = [np.array([1, 2, 3, 4]), np.array([5, 6, 7, 8])]
    tile = tmp_path / "sbs_info.parquet"
    pl.DataFrame(
        {"cell": [1, 2], "bounds": [b.tolist() for b in bounds]}
    ).write_parquet(tile)

    out = combine_tile_dfs([tile])

    assert [list(b) for b in out["bounds"]] == [[1, 2, 3, 4], [5, 6, 7, 8]]
    assert out["cell"].dtype == "int64"


@pytest.mark.unit
def test_zero_byte_tiles_are_skipped_and_all_empty_returns_none(tmp_path):
    (tmp_path / "empty.tsv").touch()
    assert combine_tile_dfs([tmp_path / "empty.tsv"]) is None

    full = _write(tmp_path / "a.tsv", pd.DataFrame({"well": ["A1"], "tile": [0]}))
    assert len(combine_tile_dfs([tmp_path / "empty.tsv", full])) == 1
