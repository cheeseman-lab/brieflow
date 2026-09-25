"""Tests for counting XY positions in well-organized ND2 files (issue #292)."""

import sys
from pathlib import Path

import numpy as np
import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

import lib.preprocess.preprocess as preprocess  # noqa: E402
from lib.preprocess.preprocess import nd2_position_count  # noqa: E402


class _FakeND2:
    """Minimal stand-in for nd2.ND2File: frame i is filled with the value i + 1."""

    def __init__(self, sizes):
        self.sizes = sizes

    def _seq_index_from_coords(self, coords):
        return (
            coords[0] if len(coords) == 1 else coords[0] * self.sizes["Z"] + coords[1]
        )

    def read_frame(self, index):
        return np.full((self.sizes.get("C", 1), 4, 4), index + 1, dtype=np.uint16)

    def close(self):
        pass


@pytest.mark.parametrize(
    "sizes,expected",
    [
        ({"P": 333, "C": 5, "Y": 1480, "X": 1480}, 333),
        ({"P": 12, "Z": 3, "C": 4, "Y": 8, "X": 8}, 12),
        ({"Z": 6, "C": 3, "Y": 18271, "X": 18271}, 1),
        ({"C": 3, "Y": 8, "X": 8}, 1),
    ],
)
def test_nd2_position_count(sizes, expected):
    fake = _FakeND2(sizes)
    assert nd2_position_count(fake) == expected


@pytest.mark.parametrize(
    "sizes,expected_max",
    [({"Z": 6, "C": 3, "Y": 4, "X": 4}, 6), ({"C": 3, "Y": 4, "X": 4}, 1)],
)
def test_convert_well_single_position(monkeypatch, sizes, expected_max):
    monkeypatch.setattr(preprocess.nd2, "ND2File", lambda path: _FakeND2(sizes))
    image, tiles = preprocess.convert_nd2_to_array_well(
        "well.nd2", position=0, return_tiles=True
    )
    assert tiles == 1
    assert image.shape == (3, 4, 4)
    assert image.max() == expected_max


def test_convert_well_multi_position_unchanged(monkeypatch):
    sizes = {"P": 5, "Z": 2, "C": 3, "Y": 4, "X": 4}
    monkeypatch.setattr(preprocess.nd2, "ND2File", lambda path: _FakeND2(sizes))
    image, tiles = preprocess.convert_nd2_to_array_well(
        "well.nd2", position=3, return_tiles=True
    )
    assert tiles == 5
    assert image.max() == 3 * 2 + 1 + 1
