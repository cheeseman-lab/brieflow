"""Unit tests for compartment filtering in the aggregate module."""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Make the workflow's `lib` package importable without installing the package
WORKFLOW_LIB_DIR = Path(__file__).resolve().parents[1] / "workflow"
sys.path.insert(0, str(WORKFLOW_LIB_DIR))

from lib.aggregate.cell_data_utils import (
    COMPARTMENT_PREFIXES,
    SECOND_OBJ_EXTRA_COLS,
    compartment_combo_subset,
)

ALL_COMPARTMENTS_4 = ["cell", "nucleus", "cytoplasm", "second_obj"]
ALL_COMPARTMENTS_3 = ["cell", "nucleus", "cytoplasm"]


def _make_features_4c():
    return pd.DataFrame(
        {
            "cell_DAPI_mean": [1.0],
            "cell_area": [2.0],
            "nucleus_DAPI_mean": [3.0],
            "nucleus_area": [4.0],
            "cytoplasm_DAPI_mean": [5.0],
            "cytoplasm_area": [6.0],
            "second_obj_DAPI_mean": [7.0],
            "second_obj_area": [8.0],
            "total_second_obj_area": [10.0],
            "mean_second_obj_diameter": [11.0],
            "mean_distance_to_nucleus": [12.0],
        }
    )


def test_all_compartments_kept():
    df = _make_features_4c()
    out = compartment_combo_subset(df, ALL_COMPARTMENTS_4, ALL_COMPARTMENTS_4)
    assert list(out.columns) == list(df.columns)


def test_nucleus_only_drops_other_prefixes():
    df = _make_features_4c()
    out = compartment_combo_subset(df, ["nucleus"], ALL_COMPARTMENTS_4)
    assert set(out.columns) == {"nucleus_DAPI_mean", "nucleus_area"}


def test_cell_plus_nucleus():
    df = _make_features_4c()
    out = compartment_combo_subset(df, ["cell", "nucleus"], ALL_COMPARTMENTS_4)
    expected = {"cell_DAPI_mean", "cell_area", "nucleus_DAPI_mean", "nucleus_area"}
    assert set(out.columns) == expected


def test_second_obj_exclusion_drops_extra_cols():
    df = _make_features_4c()
    out = compartment_combo_subset(df, ALL_COMPARTMENTS_3, ALL_COMPARTMENTS_4)
    for c in SECOND_OBJ_EXTRA_COLS:
        assert c not in out.columns
    assert "second_obj_DAPI_mean" not in out.columns
    assert "cell_area" in out.columns


def test_second_obj_included_keeps_extra_cols():
    df = _make_features_4c()
    out = compartment_combo_subset(df, ["second_obj"], ALL_COMPARTMENTS_4)
    for c in SECOND_OBJ_EXTRA_COLS:
        assert c in out.columns


def test_compartment_prefixes_constant_shape():
    assert set(COMPARTMENT_PREFIXES) == {"cell", "nucleus", "cytoplasm", "second_obj"}
    for name, prefix in COMPARTMENT_PREFIXES.items():
        assert prefix == f"{name}_"


from lib.aggregate.cell_data_utils import resolve_aggregate_combos


def test_resolve_defaults_4_when_detection_true():
    out = resolve_aggregate_combos([{"channels": ["DAPI"]}], second_obj_detection=True)
    assert out == [
        {
            "channels": ["DAPI"],
            "compartments": ["cell", "nucleus", "cytoplasm", "second_obj"],
        }
    ]


def test_resolve_defaults_3_when_detection_false():
    out = resolve_aggregate_combos([{"channels": ["DAPI"]}], second_obj_detection=False)
    assert out == [
        {"channels": ["DAPI"], "compartments": ["cell", "nucleus", "cytoplasm"]}
    ]


def test_resolve_rejects_unknown_compartment():
    with pytest.raises(ValueError, match="unknown compartment"):
        resolve_aggregate_combos(
            [{"channels": ["DAPI"], "compartments": ["nucleous"]}],
            second_obj_detection=True,
        )


def test_resolve_rejects_second_obj_when_detection_false():
    with pytest.raises(ValueError, match="second_obj_detection"):
        resolve_aggregate_combos(
            [{"channels": ["DAPI"], "compartments": ["second_obj"]}],
            second_obj_detection=False,
        )


def test_resolve_rejects_empty_compartments():
    with pytest.raises(ValueError, match="at least one compartment"):
        resolve_aggregate_combos(
            [{"channels": ["DAPI"], "compartments": []}],
            second_obj_detection=True,
        )


def test_resolve_rejects_empty_channels():
    with pytest.raises(ValueError, match="at least one channel"):
        resolve_aggregate_combos(
            [{"channels": [], "compartments": ["cell"]}],
            second_obj_detection=True,
        )


def test_resolve_dedupes_compartments_within_combo():
    out = resolve_aggregate_combos(
        [{"channels": ["DAPI"], "compartments": ["cell", "cell", "nucleus"]}],
        second_obj_detection=True,
    )
    assert out[0]["compartments"] == ["cell", "nucleus"]


def test_resolve_dedupes_duplicate_combos():
    out = resolve_aggregate_combos(
        [
            {"channels": ["DAPI"], "compartments": ["cell"]},
            {"channels": ["DAPI"], "compartments": ["cell"]},
        ],
        second_obj_detection=True,
    )
    assert len(out) == 1


def test_resolve_preserves_distinct_combos():
    out = resolve_aggregate_combos(
        [
            {"channels": ["DAPI"], "compartments": ["cell"]},
            {"channels": ["DAPI"], "compartments": ["nucleus"]},
        ],
        second_obj_detection=True,
    )
    assert len(out) == 2
