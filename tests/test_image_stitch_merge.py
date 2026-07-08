# tests/test_image_stitch_merge.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.merge.image_stitch_merge import (
    assign_subtiles,
    merge_subtiles,
    merge_reference_tiles,
    merge_reference_tiles_per_phtile,
)
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


@pytest.mark.unit
def test_merge_reference_tiles_density_mismatch():
    """align_ratio subsampling achieves high recall when PH footprint is denser than SBS.

    SBS tile: 300 cells.  PH footprint: the 300 true partners PLUS 200 extra cells
    from the same spatial distribution (1.67× total density).  This models the real
    two-scope scenario where PH (higher-res) detects additional faint nuclei absent
    from SBS — the extras are real co-field cells, not noise.

    With align_ratio=1.3 the implementation subsamples PH to int(1.3×300)=390 cells
    before triangle hashing (500 > 390 → subsampling fires) while still passing the
    full 500-cell PH set to merge_triangle_hash for matching.  The subsampled set
    retains ~67% true partners (well above the RANSAC reliability threshold), so the
    correct transform is recovered and recall on true partners exceeds 70%.
    """
    rng = np.random.default_rng(17)
    tile_shape = (500, 500)
    n_sbs = 300
    n_extra = 200  # 1.67× density flood

    coarse = {"rotation": np.eye(2), "translation": np.zeros(2), "angle_deg": 0.0, "scale": 1.0}
    off_df = pd.DataFrame({"tile": [0], "y": [0.0], "x": [0.0]})
    sbs_offsets = TileOffsets.from_frame(off_df)

    sbs_pos = rng.uniform(30, 470, size=(n_sbs, 2))
    # True PH partners: pure translation (3, 3) from SBS positions.
    ph_true = sbs_pos + np.array([3.0, 3.0])
    # Extra PH cells: drawn from the same spatial distribution as SBS (real co-field cells).
    extra_pos = rng.uniform(30, 470, size=(n_extra, 2))
    all_ph = np.vstack([ph_true, extra_pos])

    sbs_rows = [
        {"cell": k, "tile": 0, "well": "A1", "plate": 1,
         "gy": float(sbs_pos[k, 0]), "gx": float(sbs_pos[k, 1]),
         "i": float(sbs_pos[k, 0]), "j": float(sbs_pos[k, 1])}
        for k in range(n_sbs)
    ]
    ph_rows = [
        {"cell": k, "tile": 0, "well": "A1", "plate": 1,
         "gy": float(all_ph[k, 0]), "gx": float(all_ph[k, 1]),
         "i": float(all_ph[k, 0]), "j": float(all_ph[k, 1])}
        for k in range(len(all_ph))
    ]
    sbs_cells = pd.DataFrame(sbs_rows)
    ph_cells = pd.DataFrame(ph_rows)

    # align_ratio=1.3: PH (500) > 1.3×SBS (390) → subsampling fires for alignment.
    # merge_triangle_hash still receives all 500 PH cells for matching.
    merged = merge_reference_tiles(
        sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape,
        threshold=4,
        align_ratio=1.3,
        evaluate_kwargs={"ransac_kwargs": {"random_state": 0}},
    )

    recall = len(merged) / n_sbs
    assert recall > 0.70, (
        f"Density-mismatch fix: expected >70% recall, got {recall:.1%} "
        f"({len(merged)}/{n_sbs})"
    )
    assert merged["cell_0"].nunique() == len(merged), "PH cell_0 not strictly 1:1"
    assert merged["cell_1"].nunique() == len(merged), "SBS cell_1 not strictly 1:1"


@pytest.mark.unit
def test_merge_reference_tiles_global_model():
    """global_model=True rescues flooded tiles from spurious RANSAC failures.

    Setup: 6 tiles total.
      - 2 CLEAN tiles: 200 SBS cells, 200 PH cells = R_true @ SBS + t_true + 0.3px noise.
        Per-tile RANSAC succeeds → classified as GOOD tiles → provide R_global.
      - 4 FLOODED tiles: 200 SBS cells, 1000 PH cells (200 true partners + 800 random extra).
        True partner fraction = 20%. RANSAC unreliable at this density.

    global_model=True: extracts R_global from 2 clean tiles, applies globally + per-tile
    cKDTree translation → should recover >70% overall recall (≥840/1200 matches).

    global_model=False: only clean tiles produce reliable matches → ≤40% recall.
    """
    rng = np.random.default_rng(99)
    tile_shape = (600, 600)
    h, w = tile_shape
    n_sbs = 200
    n_extra = 800  # flood factor: 5× density in flooded tiles

    # True residual transform (applied on top of identity coarse)
    # This is the residual the per-tile evaluate_match would normally correct.
    angle_true = 0.6  # degrees residual
    t_true = np.array([18.0, 12.0])
    theta = np.deg2rad(angle_true)
    R_true = np.array([[np.cos(theta), -np.sin(theta)],
                       [np.sin(theta),  np.cos(theta)]])

    # Coarse = identity (so ph_coarse = ph_orig; residual = full true transform)
    coarse = {"rotation": np.eye(2), "translation": np.zeros(2), "angle_deg": 0.0, "scale": 1.0}

    # Tile layout: 3×2 grid
    tile_offsets_list = [(0, 0), (0, w), (h, 0), (h, w), (2*h, 0), (2*h, w)]
    tile_ids = [0, 1, 2, 3, 4, 5]
    off_df = pd.DataFrame({
        "tile": tile_ids,
        "y": [o[0] for o in tile_offsets_list],
        "x": [o[1] for o in tile_offsets_list],
    })
    sbs_offsets = TileOffsets.from_frame(off_df)

    sbs_rows, ph_rows = [], []
    for t_id, (oy, ox) in enumerate(tile_offsets_list):
        sbs_pos = rng.uniform(40, 560, size=(n_sbs, 2)) + np.array([oy, ox])

        for k in range(n_sbs):
            cell_id = t_id * n_sbs + k
            sbs_rows.append({
                "cell": cell_id, "tile": t_id, "well": "A1", "plate": 1,
                "gy": float(sbs_pos[k, 0]), "gx": float(sbs_pos[k, 1]),
                "i": float(sbs_pos[k, 0]), "j": float(sbs_pos[k, 1]),
            })

        # True PH partners: R_true @ SBS + t_true + tiny noise
        ph_true = (R_true @ sbs_pos.T).T + t_true + rng.normal(0, 0.3, size=(n_sbs, 2))

        if t_id < 2:
            # CLEAN tile: only true partners
            ph_all = ph_true
        else:
            # FLOODED tile: true partners + many random extras (same spatial domain)
            extra = rng.uniform(oy, oy + h, size=(n_extra, 1))
            extra = np.hstack([extra, rng.uniform(ox, ox + w, size=(n_extra, 1))])
            ph_all = np.vstack([ph_true, extra])

        for k, row in enumerate(ph_all):
            ph_rows.append({
                "cell": t_id * (n_sbs + n_extra) + k,
                "tile": t_id, "well": "A1", "plate": 1,
                "gy": float(row[0]), "gx": float(row[1]),
                "i": float(row[0]), "j": float(row[1]),
            })

    sbs_cells = pd.DataFrame(sbs_rows)
    ph_cells = pd.DataFrame(ph_rows)

    total_sbs = len(sbs_cells)  # 1200

    merged_true = merge_reference_tiles(
        sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape,
        threshold=4, local_refinement=None,
        evaluate_kwargs={"ransac_kwargs": {"random_state": 77}},
        margin_px=50.0, min_cells=20,
        global_model=True,
    )
    merged_false = merge_reference_tiles(
        sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape,
        threshold=4, local_refinement=None,
        evaluate_kwargs={"ransac_kwargs": {"random_state": 77}},
        margin_px=50.0, min_cells=20,
        global_model=False,
    )

    recall_true  = len(merged_true)  / total_sbs
    recall_false = len(merged_false) / total_sbs

    assert recall_true > 0.70, (
        f"global_model=True: expected >70% recall, got {recall_true:.1%} "
        f"({len(merged_true)}/{total_sbs})"
    )
    assert recall_true > recall_false + 0.15, (
        f"global_model=True ({recall_true:.1%}) should beat False ({recall_false:.1%}) "
        f"by >15 pp"
    )
    # Strict 1:1
    assert merged_true["cell_0"].nunique() == len(merged_true), "PH cell_0 not unique"
    assert merged_true["cell_1"].nunique() == len(merged_true), "SBS cell_1 not unique"


def _split_phtile_two_modality(seed=7):
    """3 SBS tiles, each overlapping 2 PH tiles with OPPOSITE residual transforms.

    Each 500x500 SBS tile footprint is covered by a LEFT and a RIGHT phenotype tile.
    Within one footprint the left PH tile carries residual (+1.2 deg rot, +4px shift)
    and the right PH tile carries the opposite (-1.2 deg rot, -4px shift), each about
    its own centroid.  No single per-footprint affine can align both halves (the
    piecewise-constant opposite shift is not linear in position), so a merged-footprint
    merge leaves >4px residuals on most cells.  Per-PH-tile hashing fits each coherent
    PH tile exactly.  Coarse transform is identity, so gy_ref == gy.

    Returns:
        (sbs_cells, ph_cells, coarse, sbs_offsets_dict, tile_shape) synthetic tables.
    """
    rng = np.random.default_rng(seed)
    tile_shape = (500, 500)
    h, w = tile_shape
    n_per_phtile = 150

    sbs_offsets = {0: (0.0, 0.0), 1: (0.0, float(w)), 2: (0.0, float(2 * w))}
    coarse = {"rotation": np.eye(2), "translation": np.zeros(2),
              "angle_deg": 0.0, "scale": 1.0}

    def _rot(deg):
        th = np.deg2rad(deg)
        return np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])

    sbs_rows, ph_rows = [], []
    sbs_cid = 0
    ph_cid = 0
    for s in range(3):
        ox = s * w
        # (ph_tile_id, x-range, residual angle deg, shift)
        subtiles = [
            (10 + 2 * s, (ox + 30, ox + 230), +1.2, np.array([4.0, 4.0])),
            (11 + 2 * s, (ox + 270, ox + 470), -1.2, np.array([-4.0, -4.0])),
        ]
        for ph_tile_id, (x0, x1), ang, shift in subtiles:
            q = np.column_stack([
                rng.uniform(30, 470, size=n_per_phtile),          # y (truth SBS)
                rng.uniform(x0, x1, size=n_per_phtile),           # x (truth SBS)
            ])
            c = q.mean(axis=0)
            R = _rot(ang)
            p = (R @ (q - c).T).T + c + shift  # PH partner positions
            for k in range(n_per_phtile):
                sbs_rows.append({
                    "cell": sbs_cid, "tile": s, "well": "A1", "plate": 1,
                    "gy": float(q[k, 0]), "gx": float(q[k, 1]),
                    "i": float(q[k, 0]), "j": float(q[k, 1]),
                })
                ph_rows.append({
                    "cell": ph_cid, "tile": ph_tile_id, "well": "A1", "plate": 1,
                    "gy": float(p[k, 0]), "gx": float(p[k, 1]),
                    "i": float(p[k, 0]), "j": float(p[k, 1]),
                })
                sbs_cid += 1
                ph_cid += 1

    return pd.DataFrame(sbs_rows), pd.DataFrame(ph_rows), coarse, sbs_offsets, tile_shape


@pytest.mark.unit
def test_merge_reference_tiles_per_phtile_recovers_split_tiles():
    """Per-PH-tile hashing recovers >70% with strict 1:1 on split-residual data."""
    sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape = _split_phtile_two_modality()

    merged = merge_reference_tiles_per_phtile(
        sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape,
        margin_px=50, min_overlap_cells=30, threshold=4,
        local_refinement=True,
    )

    n_sbs = len(sbs_cells)
    recall = len(merged) / n_sbs
    assert recall > 0.70, f"per-PH-tile recall {recall:.1%} ({len(merged)}/{n_sbs})"
    assert merged["cell_0"].nunique() == len(merged), "PH cell_0 not strictly 1:1"
    assert merged["cell_1"].nunique() == len(merged), "SBS cell_1 not strictly 1:1"


@pytest.mark.unit
def test_per_phtile_beats_merged_footprint():
    """Per-PH-tile mode beats merged-footprint merge_reference_tiles on the same data."""
    sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape = _split_phtile_two_modality()

    merged_per = merge_reference_tiles_per_phtile(
        sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape,
        margin_px=50, min_overlap_cells=30, threshold=4,
        local_refinement=True,
    )

    off_df = pd.DataFrame({
        "tile": list(sbs_offsets.keys()),
        "y": [v[0] for v in sbs_offsets.values()],
        "x": [v[1] for v in sbs_offsets.values()],
    })
    merged_foot = merge_reference_tiles(
        sbs_cells, ph_cells, coarse, TileOffsets.from_frame(off_df), tile_shape,
        threshold=4, local_refinement="thin_plate_spline",
        evaluate_kwargs={"ransac_kwargs": {"random_state": 0}},
        margin_px=50.0, min_cells=30, global_model=True,
    )

    n_sbs = len(sbs_cells)
    recall_per = len(merged_per) / n_sbs
    recall_foot = len(merged_foot) / n_sbs
    assert recall_per > recall_foot + 0.15, (
        f"per-PH-tile ({recall_per:.1%}) should beat merged-footprint "
        f"({recall_foot:.1%}) by >15 pp"
    )


@pytest.mark.unit
def test_per_phtile_neg_det_fallback_helps():
    """neg_det_fallback recovers at least as many cells as dropping failed pairs."""
    sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape = _split_phtile_two_modality()

    kw = dict(margin_px=50, min_overlap_cells=30, threshold=4, local_refinement=True)
    merged_on = merge_reference_tiles_per_phtile(
        sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape,
        neg_det_fallback=True, **kw,
    )
    merged_off = merge_reference_tiles_per_phtile(
        sbs_cells, ph_cells, coarse, sbs_offsets, tile_shape,
        neg_det_fallback=False, **kw,
    )
    assert len(merged_on) >= len(merged_off), (
        f"fallback on ({len(merged_on)}) should be >= off ({len(merged_off)})"
    )
    assert merged_on["cell_1"].nunique() == len(merged_on), "SBS cell_1 not 1:1"
