# tests/test_stitching_core.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.shared.stitching.types import TileOffsets
from workflow.lib.shared.stitching.prep import select_registration_plane


@pytest.mark.unit
def test_tileoffsets_roundtrip():
    df = pd.DataFrame({"tile": [0, 1], "y": [0.0, 5.5], "x": [0.0, -3.0]})
    off = TileOffsets.from_frame(df)
    pd.testing.assert_frame_equal(off.to_frame(), df)


@pytest.mark.unit
def test_select_registration_plane_4d_and_3d():
    stack4d = np.zeros((11, 5, 8, 8), dtype=np.uint16)
    stack4d[0, 0] = 7  # cycle 0, channel 0
    plane = select_registration_plane(stack4d, channel=0, cycle=0)
    assert plane.shape == (8, 8) and plane[0, 0] == 7

    stack3d = np.zeros((3, 8, 8), dtype=np.uint16)
    stack3d[2] = 4
    plane = select_registration_plane(stack3d, channel=2, cycle=None)
    assert plane.shape == (8, 8) and plane[0, 0] == 4


from workflow.lib.shared.stitching.register import register_pair
from scipy.ndimage import shift as ndi_shift


@pytest.mark.unit
def test_register_pair_recovers_known_translation():
    rng = np.random.default_rng(0)
    full = rng.random((256, 256)).astype(np.float32)
    ref = full[:, :200]                      # left tile
    true = np.array([0.0, 160.0])            # mov shifted right by 160 px
    mov = ndi_shift(full, shift=-true, order=1)[:, :200].astype(np.float32)
    shift_yx, conf = register_pair(
        ref, mov, expected_shift=(0.0, 150.0), overlap_fraction=0.25, max_shift=40.0
    )
    assert conf > 0.5
    np.testing.assert_allclose(shift_yx, true, atol=1.0)


from workflow.lib.shared.stitching.place import solve_global_offsets


@pytest.mark.unit
def test_solve_global_offsets_chain():
    # 3 tiles in a row; edge (i->j) carries the true offset of j relative to i.
    edges = [
        (0, 1, np.array([0.0, 100.0]), 0.9),
        (1, 2, np.array([0.0, 100.0]), 0.9),
    ]
    prior = {0: (0.0, 0.0), 1: (0.0, 90.0), 2: (0.0, 190.0)}
    off = solve_global_offsets(3, edges, prior, min_confidence=0.2).to_frame()
    off = off.sort_values("tile").reset_index(drop=True)
    np.testing.assert_allclose(off["x"].to_numpy(), [0.0, 100.0, 200.0], atol=1e-6)
    np.testing.assert_allclose(off["y"].to_numpy(), [0.0, 0.0, 0.0], atol=1e-6)


@pytest.mark.unit
def test_solve_global_offsets_disconnected_uses_prior():
    edges = [(0, 1, np.array([0.0, 100.0]), 0.9)]  # tile 2 disconnected
    prior = {0: (0.0, 0.0), 1: (0.0, 90.0), 2: (5.0, 300.0)}
    off = solve_global_offsets(3, edges, prior, min_confidence=0.2).to_frame()
    row2 = off[off["tile"] == 2].iloc[0]
    assert (row2["y"], row2["x"]) == (5.0, 300.0)


from workflow.lib.shared.stitching.stitch_well import find_neighbor_pairs, stitch_well
from workflow.lib.shared.stitching.place_cells import place_cells


@pytest.mark.unit
def test_place_cells_applies_offset():
    cells = pd.DataFrame({"tile": [0, 1], "i": [10.0, 20.0], "j": [5.0, 7.0],
                          "cell": [0, 1]})
    off = TileOffsets.from_frame(
        pd.DataFrame({"tile": [0, 1], "y": [0.0, 100.0], "x": [0.0, 50.0]})
    )
    out = place_cells(cells, off)
    np.testing.assert_allclose(out["gy"].to_numpy(), [10.0, 120.0])
    np.testing.assert_allclose(out["gx"].to_numpy(), [5.0, 57.0])


@pytest.mark.unit
def test_find_neighbor_pairs_grid():
    # 2x2 grid, tile side 100, 10% overlap -> right/down neighbors only.
    prior = {0: (0, 0), 1: (0, 90), 2: (90, 0), 3: (90, 90)}
    pairs = find_neighbor_pairs(prior, tile_shape=(100, 100), overlap_fraction=0.1)
    got = {(i, j) for i, j, _ in pairs}
    assert (0, 1) in got and (0, 2) in got and (1, 3) in got and (2, 3) in got
    assert (0, 3) not in got  # diagonal, no overlap
