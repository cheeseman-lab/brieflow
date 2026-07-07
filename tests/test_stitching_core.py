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
