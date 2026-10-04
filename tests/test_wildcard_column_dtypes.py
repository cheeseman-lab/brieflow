"""Numeric wildcards must survive a parquet round-trip as integers.

Snakemake hands wildcards over as strings. TSV output hid that — `pd.read_csv`
re-inferred `plate`/`tile`/`cycle` as int64 on read — but parquet preserves the
stored dtype, so a string `tile` reaches the reader and every downstream merge
against the int64 `tile` in the metadata tables raises
"You are trying to merge on object and int64 columns".
"""

import pandas as pd
import pytest

from lib.shared.file_utils import add_wildcard_columns
from lib.shared.parquet_io import read_parquet, write_parquet


@pytest.mark.unit
def test_numeric_wildcards_become_integers_and_others_stay_strings():
    df = add_wildcard_columns(
        pd.DataFrame({"label": [1, 2]}),
        {"plate": "1", "well": "A1", "tile": "0", "row": "r02", "col": "c02"},
    )
    assert df["plate"].dtype == "int64"
    assert df["tile"].dtype == "int64"
    assert df["well"].dtype == object
    assert df["row"].dtype == object
    assert df["col"].dtype == object


@pytest.mark.unit
def test_tile_still_merges_with_int_metadata_after_a_parquet_round_trip(tmp_path):
    """The end-to-end shape of the failure: eval_features merges the per-well
    table against combined_metadata, which stores tile as int64."""
    fp = tmp_path / "phenotype_cp.parquet"
    write_parquet(
        add_wildcard_columns(
            pd.DataFrame({"label": [1, 2], "area": [10.0, 20.0]}),
            {"well": "A1", "tile": "7"},
        ),
        fp,
    )
    metadata = pd.DataFrame({"well": ["A1"], "tile": [7], "x_pos": [0.0]})
    merged = read_parquet(fp).merge(metadata, on=["well", "tile"], how="left")
    assert merged["x_pos"].notna().all()
