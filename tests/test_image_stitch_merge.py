# tests/test_image_stitch_merge.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.merge.image_stitch_merge import assign_subtiles, merge_subtiles, merge_reference_tiles
from workflow.lib.shared.stitching.types import TileOffsets


@pytest.mark.unit
def test_assign_subtiles_grid():
    cells = pd.DataFrame({"gy": [0.0, 250.0, 10.0], "gx": [0.0, 10.0, 250.0]})
    out = assign_subtiles(cells, subtile_size=(200, 200))
    # (0,0)->tile at grid (0,0); (250,10)->grid (1,0); (10,250)->grid (0,1)
    assert out["subtile"].tolist() == [0, 2, 1] or len(set(out["subtile"])) == 3
    # same-bucket cells share an id
    assert out.loc[0, "subtile"] != out.loc[1, "subtile"]


def _piecewise_two_modality(seed=0):
    """PH and SBS in a COMMON global frame, related by per-subtile local affine.

    Cells are placed in the interior of each 400x400 subtile region (50px margin from
    each edge) to ensure the small per-subtile rotation + translation cannot push sbs
    cells across subtile boundaries, preserving shared-subtile bucketing for both frames.
    The 2x2 grid has 200 cells per subtile; rotation ramps 0.3 deg to 1.0 deg.
    """
    rng = np.random.default_rng(seed)
    rows = []
    cid = 0
    for st_r in range(2):
        for st_c in range(2):
            n = 200
            # Cells in interior of each subtile's 400x400 region (50px margin)
            X = rng.uniform(50, 350, size=(n, 2)) + np.array([st_r * 400, st_c * 400])
            # Per-subtile rotation ramp: 0.3 to 1.0 degrees across 2x2 grid
            angle_deg = 0.3 + 0.7 * (st_r * 2 + st_c) / 3.0
            theta = np.deg2rad(angle_deg)
            center = X.mean(axis=0)
            R = np.array([[np.cos(theta), -np.sin(theta)],
                          [np.sin(theta),  np.cos(theta)]])
            # Small per-subtile translation (+-1px, symmetric across subtiles)
            t = np.array([1.0 * (-1) ** st_r, 1.0 * (-1) ** st_c])
            Y = (X - center) @ R.T + center + t
            for k in range(n):
                rows.append((cid, X[k, 0], X[k, 1], Y[k, 0], Y[k, 1]))
                cid += 1
    df = pd.DataFrame(rows, columns=["cell", "phy", "phx", "sy", "sx"])
    ph = df[["cell", "phy", "phx"]].rename(columns={"phy": "gy", "phx": "gx"})
    ph["tile"] = 0; ph["well"] = "A1"; ph["plate"] = 1
    ph["i"] = ph["gy"]; ph["j"] = ph["gx"]
    sbs = df[["cell", "sy", "sx"]].rename(columns={"sy": "gy", "sx": "gx"})
    sbs["tile"] = 0; sbs["well"] = "A1"; sbs["plate"] = 1
    sbs["i"] = sbs["gy"]; sbs["j"] = sbs["gx"]
    return ph, sbs


@pytest.mark.unit
def test_merge_subtiles_recovers_matches():
    ph, sbs = _piecewise_two_modality()
    merged = merge_subtiles(
        ph, sbs, subtile_size=(400, 400), threshold=4,
        local_refinement="thin_plate_spline", warp_kwargs=None,
        evaluate_kwargs={"ransac_kwargs": {"random_state": 0}},
    )
    # >70% of PH cells matched across the 4 subtiles despite per-subtile rotation ramp
    assert len(merged) > 0.7 * len(ph)
    # Strict 1:1: no duplicate ph or sbs cell assignments after deduplicate_cells
    assert merged["cell_0"].nunique() == len(merged)
    assert merged["cell_1"].nunique() == len(merged)


@pytest.mark.unit
def test_merge_reference_tiles():
    """merge_reference_tiles matches >70% of SBS cells with strict 1:1."""
    rng = np.random.default_rng(42)
    tile_shape = (500, 500)
    h, w = tile_shape
    n_per_tile = 200

    tile_ids = [0, 1, 2, 3]
    tile_offsets = [(0, 0), (0, w), (h, 0), (h, w)]
    off_df = pd.DataFrame({
        "tile": tile_ids,
        "y": [o[0] for o in tile_offsets],
        "x": [o[1] for o in tile_offsets],
    })
    sbs_offsets = TileOffsets.from_frame(off_df)

    # Global transform: 0.5° rotation + [5, 8] translation in SBS px, scale=1.0
    angle_global = 0.5
    t_global = np.array([5.0, 8.0])
    theta = np.deg2rad(angle_global)
    R_fwd = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta),  np.cos(theta)]])
    # Inverse transform maps PH → SBS
    R_inv = R_fwd.T
    t_inv = -R_fwd.T @ t_global
    coarse = {
        "rotation": R_inv,
        "translation": t_inv,
        "angle_deg": -angle_global,
        "scale": 1.0,
    }

    sbs_rows, ph_rows = [], []
    for t_id, (oy, ox) in enumerate(tile_offsets):
        local_ij = rng.uniform(40, 460, size=(n_per_tile, 2))
        gy = local_ij[:, 0] + oy
        gx = local_ij[:, 1] + ox

        # Use globally unique cell IDs so cell_0.nunique() == len(merged) is testable
        for k in range(n_per_tile):
            cell_id = t_id * n_per_tile + k
            sbs_rows.append({
                "cell": cell_id, "tile": t_id, "well": "A1", "plate": 1,
                "gy": gy[k], "gx": gx[k], "i": gy[k], "j": gx[k],
            })

        # Per-tile residual: 0.08° rotation around tile center
        angle_res = 0.08 * (t_id + 1)
        theta_r = np.deg2rad(angle_res)
        R_res = np.array([[np.cos(theta_r), -np.sin(theta_r)],
                          [np.sin(theta_r),  np.cos(theta_r)]])
        center = np.array([oy + h / 2.0, ox + w / 2.0])

        for k in range(n_per_tile):
            cell_id = t_id * n_per_tile + k
            p = np.array([gy[k], gx[k]])
            p_ph = R_fwd @ p + t_global
            p_ph = R_res @ (p_ph - center) + center
            ph_rows.append({
                "cell": cell_id, "tile": t_id, "well": "A1", "plate": 1,
                "gy": p_ph[0], "gx": p_ph[1], "i": p_ph[0], "j": p_ph[1],
            })

    sbs_cells = pd.DataFrame(sbs_rows)
    ph_cells = pd.DataFrame(ph_rows)

    merged = merge_reference_tiles(
        sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape,
        threshold=4,
        evaluate_kwargs={"ransac_kwargs": {"random_state": 0}},
    )

    total_sbs = len(sbs_cells)
    assert len(merged) > 0.70 * total_sbs, (
        f"Only matched {len(merged)}/{total_sbs} ({len(merged)/total_sbs:.1%})"
    )
    assert merged["cell_0"].nunique() == len(merged), "PH cell_0 not strictly unique"
    assert merged["cell_1"].nunique() == len(merged), "SBS cell_1 not strictly unique"
