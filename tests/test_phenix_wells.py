"""Regression tests for Opera Phenix rNNcNN well nomenclature + SBS z-plane convert.

Covers the two bug classes fixed for the first Phenix screens (PoTC, zargun):

1. Well->(row,col) derivation must handle both the alphanumeric convention (A1) and
   Opera Phenix (r02c05), via the single canonical `split_well`/`split_well_to_cols`
   in lib.shared.file_utils. The naive `well[0], well[1:]` split turned r02c05 into
   ("r","02c05") and 404'd every HCS-nested path; `discover_plate_structure`'s
   alpha-only guard also silently dropped Phenix wells from tile enumeration.
2. `get_sample_fps` must not collapse z-planes for SBS (no round_order): keying a
   dict by channel dropped the n_z_planes rows/channel, yielding single-channel zarrs.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`),
# so hcs.py's own `from lib.shared.file_utils import ...` resolves too.
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.shared.file_utils import split_well, split_well_to_cols  # noqa: E402
from lib.preprocess.file_utils import get_sample_fps  # noqa: E402
from lib.shared.hcs import discover_plate_structure  # noqa: E402


# --- Bug class 1: canonical well split ------------------------------------------

@pytest.mark.parametrize(
    "well,expected",
    [
        ("A1", ("A", "1")),
        ("B12", ("B", "12")),
        ("A01", ("A", "01")),  # zeros preserved so read paths match write paths
        ("r02c05", ("r02", "c05")),
        ("r06c10", ("r06", "c10")),
        ("R2C5", ("R2", "C5")),
    ],
)
def test_split_well(well, expected):
    assert split_well(well) == expected


def test_split_well_roundtrips():
    for well in ("A1", "B12", "r02c05", "r06c10"):
        row, col = split_well(well)
        assert f"{row}{col}" == well  # str(row)+str(col) reconstructs the well


def test_split_well_raises_on_unknown():
    with pytest.raises(ValueError):
        split_well("weird-99")


def test_split_well_to_cols_matches_scalar():
    df = pd.DataFrame({"well": ["A1", "B12", "r02c05", "r06c10"]})
    out = split_well_to_cols(df)
    for _, r in out.iterrows():
        assert (r["row"], r["col"]) == split_well(r["well"])


def test_discover_plate_structure_includes_phenix(tmp_path):
    """Phenix wells must not be silently dropped from tile enumeration."""
    plate = tmp_path / "image_1.zarr"
    # one alpha well and one Phenix well, each a tile marker <row>/<col>/<tile>/zarr.json
    for row, col in [("A", "1"), ("r02", "c03")]:
        d = plate / row / col / "1"
        d.mkdir(parents=True)
        (d / "zarr.json").write_text("{}")
    found = set(discover_plate_structure(plate))
    assert ("A", "1", "1") in found
    assert ("r02", "c03", "1") in found  # would be dropped by the old alpha-only guard


# --- Bug class 2: SBS z-plane convert -------------------------------------------

def _tile_df(channels, z_planes, well="r05c04"):
    rows = []
    for ch in channels:
        for zp in z_planes:
            rows.append(
                {
                    "plate": "1", "well": well, "tile": "24", "cycle": "1",
                    "channel": ch, "z": zp, "sample_fp": f"{ch}_z{zp}.tiff",
                }
            )
    return pd.DataFrame(rows)


def test_get_sample_fps_multiz_keeps_all_channels_and_planes():
    """3 channels x 3 z must return 9 files, channel-major, z ascending within channel."""
    df = _tile_df(["C", "T", "DAPI"], [1, 2, 3])
    res = get_sample_fps(df, plate="1", channel_order=["C", "T", "DAPI"])
    assert res == [
        "C_z1.tiff", "C_z2.tiff", "C_z3.tiff",
        "T_z1.tiff", "T_z2.tiff", "T_z3.tiff",
        "DAPI_z1.tiff", "DAPI_z2.tiff", "DAPI_z3.tiff",
    ]


def test_get_sample_fps_singlez_unaffected():
    """1 z per channel (e.g. PoTC): dict-collapse never bit this — still N files."""
    df = _tile_df(["DAPI", "A568", "A488", "A647"], [1])
    res = get_sample_fps(df, plate="1", channel_order=["DAPI", "A568", "A488", "A647"])
    assert res == ["DAPI_z1.tiff", "A568_z1.tiff", "A488_z1.tiff", "A647_z1.tiff"]


def test_get_sample_fps_bare_z_guard_raises_on_multichannel():
    """Multi-channel z-stack with NO channel_order must raise, not interleave."""
    df = _tile_df(["C", "T"], [1, 2, 3])
    with pytest.raises(ValueError):
        get_sample_fps(df, plate="1")  # no channel_order -> ambiguous ordering
