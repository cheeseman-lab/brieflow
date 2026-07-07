# tests/test_image_stitch_merge.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.merge.image_stitch_merge import assign_subtiles, merge_subtiles


@pytest.mark.unit
def test_assign_subtiles_grid():
    cells = pd.DataFrame({"gy": [0.0, 250.0, 10.0], "gx": [0.0, 10.0, 250.0]})
    out = assign_subtiles(cells, subtile_size=(200, 200))
    # (0,0)->tile at grid (0,0); (250,10)->grid (1,0); (10,250)->grid (0,1)
    assert out["subtile"].tolist() == [0, 2, 1] or len(set(out["subtile"])) == 3
    # same-bucket cells share an id
    assert out.loc[0, "subtile"] != out.loc[1, "subtile"]


def _piecewise_two_modality(seed=0):
    """Two global frames related by a per-subtile-varying affine (rotation ramp)."""
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for st_r in range(2):
        for st_c in range(2):
            n = 200
            X = rng.uniform(0, 400, size=(n, 2)) + np.array([st_r * 400, st_c * 400])
            theta = np.deg2rad(1.0 + 0.3 * (st_r + st_c))  # varies per subtile
            scale = 0.27
            R = scale * np.array([[np.cos(theta), -np.sin(theta)],
                                  [np.sin(theta), np.cos(theta)]])
            Y = X @ R.T + np.array([40.0, -15.0])
            for k in range(n):
                rows.append((cid, X[k, 0], X[k, 1], Y[k, 0], Y[k, 1]))
                cid += 1
    df = pd.DataFrame(rows, columns=["cell", "phy", "phx", "sy", "sx"])
    ph = df[["cell", "phy", "phx"]].rename(columns={"phy": "gy", "phx": "gx"})
    ph["tile"] = 0; ph["well"] = "A1"; ph["plate"] = 1; ph["i"] = ph["gy"]; ph["j"] = ph["gx"]
    sbs = df[["cell", "sy", "sx"]].rename(columns={"sy": "gy", "sx": "gx"})
    sbs["tile"] = 0; sbs["well"] = "A1"; sbs["plate"] = 1; sbs["i"] = sbs["gy"]; sbs["j"] = sbs["gx"]
    return ph, sbs


@pytest.mark.unit
def test_merge_subtiles_recovers_matches():
    ph, sbs = _piecewise_two_modality()
    merged = merge_subtiles(
        ph, sbs, subtile_size=(400, 400), threshold=4,
        local_refinement="thin_plate_spline", warp_kwargs=None,
        evaluate_kwargs={"ransac_kwargs": {"random_state": 0}},
    )
    # most cells should match across the 4 subtiles despite the rotation ramp
    assert len(merged) > 0.7 * len(ph)
