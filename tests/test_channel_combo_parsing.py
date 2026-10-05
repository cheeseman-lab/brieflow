"""Regression tests for channel names that contain "_" or contain another channel name.

Aggregate combos are "_"-joined channel names. Splitting the combo on "_" dropped
any channel whose name holds an underscore, and substring column matching let an
excluded channel ("LIPID") remove the columns of a kept one ("LIPID_Low").
"""

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd
import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.cell_data_utils import (  # noqa: E402
    channel_combo_subset,
    column_channels,
    parse_channel_combo,
)


def _columns(channels):
    """Feature columns in the pipeline's naming scheme for a channel set."""
    cols = ["cell_area", "nucleus_eccentricity", "cytoplasm_zernike_9_1"]
    for comp in ("nucleus", "cell", "cytoplasm"):
        for ch in channels:
            cols += [f"{comp}_{ch}_mean", f"{comp}_{ch}_int", f"{comp}_{ch}_radial_cv"]
        for a, b in combinations(channels, 2):
            cols += [f"{comp}_correlation_{a}_{b}", f"{comp}_lstsq_slope_{b}_{a}"]
    return cols


def _old_channel_combo_subset(features, channel_combo, all_channels):
    """The substring-matching implementation this change replaces."""
    remove = [ch for ch in all_channels if ch not in channel_combo]
    return features[[c for c in features.columns if not any(ch in c for ch in remove)]]


@pytest.mark.parametrize(
    "combo, channels, expected",
    [
        ("DAPI_COXIV_CENPA_WGA", ["DAPI", "COXIV", "CENPA", "WGA"], None),
        ("DAPI", ["DAPI", "COXIV", "CENPA", "WGA"], ["DAPI"]),
        (
            "DAPI_LIPID_Low_LIPID_WGA",
            ["DAPI", "LIPID_Low", "LIPID", "WGA"],
            ["DAPI", "LIPID_Low", "LIPID", "WGA"],
        ),
        ("LIPID_LIPID_Low", ["LIPID", "LIPID_Low"], ["LIPID", "LIPID_Low"]),
        ("DAPI2_DAPI", ["DAPI", "DAPI2"], ["DAPI2", "DAPI"]),
        ("A_B_C", ["A_B", "B_C", "A"], ["A", "B_C"]),
    ],
)
def test_parse_channel_combo(combo, channels, expected):
    expected = expected if expected is not None else combo.split("_")
    assert parse_channel_combo(combo, channels) == expected


def test_parse_channel_combo_unparseable_names_leftover():
    with pytest.raises(ValueError, match="unmatched text: 'Lo_WGA'"):
        parse_channel_combo("DAPI_Lo_WGA", ["DAPI", "LIPID_Low", "WGA"])


def test_column_channels_longest_token_match():
    chans = ["DAPI", "DAPI2", "LIPID", "LIPID_Low"]
    assert column_channels("cell_LIPID_Low_mean", chans) == ["LIPID_Low"]
    assert column_channels("cell_LIPID_mean", chans) == ["LIPID"]
    assert column_channels("cell_correlation_DAPI2_LIPID_Low", chans) == [
        "DAPI2",
        "LIPID_Low",
    ]
    assert column_channels("cell_area", chans) == []


@pytest.mark.parametrize(
    "channels, keep",
    [
        (["DAPI", "LIPID_Low", "LIPID", "WGA"], ["DAPI", "LIPID_Low", "WGA"]),
        (["DAPI", "LIPID_Low", "LIPID", "WGA"], ["DAPI", "LIPID"]),
        (["DAPI", "DAPI2", "WGA"], ["DAPI2", "WGA"]),
    ],
)
def test_channel_combo_subset_keeps_listed_removes_unlisted(channels, keep):
    features = pd.DataFrame(columns=_columns(channels))
    kept = channel_combo_subset(features, keep, channels).columns
    for col in features.columns:
        named = column_channels(col, channels)
        assert (col in kept) == all(ch in keep for ch in named), col
    for ch in keep:
        assert f"cell_{ch}_mean" in kept


def test_issue_reproduction():
    all_ch = ["DAPI", "LIPID_Low", "LIPID", "WGA"]
    cols = [
        "cell_DAPI_mean",
        "cell_LIPID_Low_mean",
        "cell_LIPID_mean",
        "cell_WGA_mean",
        "cell_area",
    ]
    f = pd.DataFrame(columns=cols)
    combo = parse_channel_combo("DAPI_LIPID_Low_LIPID_WGA", all_ch)
    assert list(channel_combo_subset(f, combo, all_ch).columns) == cols
    kept = channel_combo_subset(f, ["DAPI", "LIPID_Low", "WGA"], all_ch).columns
    assert list(kept) == [
        "cell_DAPI_mean",
        "cell_LIPID_Low_mean",
        "cell_WGA_mean",
        "cell_area",
    ]


@pytest.mark.parametrize("n_keep", [1, 2, 3, 4])
def test_unchanged_for_small_test_channel_set(n_keep):
    channels = ["DAPI", "COXIV", "CENPA", "WGA"]
    features = pd.DataFrame(columns=_columns(channels))
    for keep in combinations(channels, n_keep):
        combo = parse_channel_combo("_".join(keep), channels)
        assert combo == "_".join(keep).split("_")
        new = channel_combo_subset(features, combo, channels).columns
        old = _old_channel_combo_subset(features, combo, channels).columns
        assert list(new) == list(old)
