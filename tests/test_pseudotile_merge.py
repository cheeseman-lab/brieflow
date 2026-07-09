# tests/test_pseudotile_merge.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.merge.pseudotile_merge import (
    register_stage_frames,
    stage_coarse_transform,
    merge_pseudotiles,
)


def _grid_meta(n_rows, n_cols, spacing, dy=0.0, dx=0.0):
    """Build a regular grid metadata DataFrame with optional offset."""
    records = []
    tile = 0
    for r in range(n_rows):
        for c in range(n_cols):
            records.append(
                {"tile": tile, "y_pos": r * spacing + dy, "x_pos": c * spacing + dx}
            )
            tile += 1
    return pd.DataFrame(records)


@pytest.mark.unit
def test_register_stage_frames_recovers_known_translation():
    ph_meta = _grid_meta(3, 3, spacing=500.0)
    sbs_meta = _grid_meta(3, 3, spacing=500.0, dy=300.0, dx=645.0)

    result = register_stage_frames(sbs_meta, ph_meta)

    assert np.allclose(result["translation"], [300.0, 645.0], atol=1e-6)
    assert np.allclose(result["rotation"], np.eye(2))


@pytest.mark.unit
def test_register_stage_frames_y_then_x_ordering():
    ph_meta = _grid_meta(3, 3, spacing=500.0)
    sbs_meta = _grid_meta(3, 3, spacing=500.0, dy=100.0, dx=700.0)

    result = register_stage_frames(sbs_meta, ph_meta)

    assert np.isclose(result["translation"][0], 100.0, atol=1e-6), "index 0 must be dy"
    assert np.isclose(result["translation"][1], 700.0, atol=1e-6), "index 1 must be dx"


@pytest.mark.unit
def test_register_stage_frames_robust_to_extra_tiles():
    # ph_meta: 5x5 grid centred at (1000, 1000) + offset of (200, 400)
    ph_meta = _grid_meta(5, 5, spacing=500.0, dy=0.0, dx=0.0)
    # Shift ph_meta so its centre matches after applying known (dy, dx)
    # sbs: 3x3 subset — symmetric, so median == centre of 3x3 == (500, 500)
    # ph:  5x5 — median == centre of 5x5 == (1000, 1000)
    # We want translation = [300.0, 500.0]
    # => sbs_center - ph_center = [300.0, 500.0]
    # => sbs grid must be centred at (1300, 1500) relative to ph origin
    ph_meta_shifted = ph_meta.copy()
    # ph grid: y in {0,500,1000,1500,2000}, x in {0,500,1000,1500,2000}; median=(1000,1000)
    # sbs grid: y in {1300,1800,2300}, x in {1500,2000,2500}; median=(1800,2000)
    # translation = (1800-1000, 2000-1000) = (800, 1000) — just verify within atol=5
    sbs_meta = _grid_meta(3, 3, spacing=500.0, dy=1300.0, dx=1500.0)

    result = register_stage_frames(sbs_meta, ph_meta_shifted)

    expected = np.array([800.0, 1000.0])
    assert np.allclose(result["translation"], expected, atol=5.0)
    assert np.allclose(result["rotation"], np.eye(2))


# ---------------------------------------------------------------------------
# stage_coarse_transform tests
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_stage_coarse_transform_is_pure_scale_rotation():
    sbs_meta = _grid_meta(3, 3, spacing=1208.548)
    ph_meta = _grid_meta(3, 3, spacing=325.0, dy=200.0, dx=300.0)
    sbs_um_per_px = 1.208548
    ph_um_per_px = 0.325

    result = stage_coarse_transform(sbs_meta, ph_meta, sbs_um_per_px, ph_um_per_px)

    scale = ph_um_per_px / sbs_um_per_px
    assert np.isclose(result["rotation"][0, 0], scale), "diagonal [0,0] must equal scale"
    assert np.isclose(result["rotation"][1, 1], scale), "diagonal [1,1] must equal scale"
    assert np.isclose(result["rotation"][0, 1], 0.0), "off-diagonal [0,1] must be zero"
    assert np.isclose(result["rotation"][1, 0], 0.0), "off-diagonal [1,0] must be zero"
    assert result["angle_deg"] == 0.0
    assert np.isclose(result["scale"], scale)


@pytest.mark.unit
def test_stage_coarse_transform_projects_known_point():
    # SBS grid: 3x3, 1000-px spacing at 1.208548 µm/px → 1208.548 µm spacing; origin (0,0)
    # PH grid:  3x3,  325.0-µm spacing at 0.325 µm/px → 1000-px spacing; origin (200,300) µm
    sbs_um_per_px = 1.208548
    ph_um_per_px = 0.325
    sbs_meta = _grid_meta(3, 3, spacing=1208.548, dy=0.0, dx=0.0)
    ph_meta = _grid_meta(3, 3, spacing=325.0, dy=200.0, dx=300.0)

    result = stage_coarse_transform(sbs_meta, ph_meta, sbs_um_per_px, ph_um_per_px)

    # ---- closed-form from transform dict ----
    g_ph = np.array([500.0, 400.0])  # arbitrary PH global-px point
    ph_ref_via_dict = result["rotation"] @ g_ph + result["translation"]

    # ---- independent derivation ----
    # physical stage µm for this PH cell (y, x):
    scale = ph_um_per_px / sbs_um_per_px
    ph_y_min = float(ph_meta["y_pos"].min())
    ph_x_min = float(ph_meta["x_pos"].min())
    sbs_y_min = float(sbs_meta["y_pos"].min())
    sbs_x_min = float(sbs_meta["x_pos"].min())
    stage = register_stage_frames(sbs_meta, ph_meta)
    ty, tx = float(stage["translation"][0]), float(stage["translation"][1])
    # abs stage µm in PH frame → add t → SBS frame → subtract sbs_min → divide by sbs_um_per_px
    phys_y = g_ph[0] * ph_um_per_px + ph_y_min
    phys_x = g_ph[1] * ph_um_per_px + ph_x_min
    expected_y = (phys_y + ty - sbs_y_min) / sbs_um_per_px
    expected_x = (phys_x + tx - sbs_x_min) / sbs_um_per_px
    ph_ref_expected = np.array([expected_y, expected_x])

    assert np.allclose(ph_ref_via_dict, ph_ref_expected, atol=1e-6)
    # also equals scale·g_ph + const directly
    assert np.allclose(ph_ref_via_dict, scale * g_ph + result["translation"], atol=1e-9)


@pytest.mark.unit
def test_stage_coarse_transform_zero_offset_identity_translation():
    # Identical stage coords and equal pixel sizes → translation ≈ [0,0], scale == 1.0
    meta = _grid_meta(3, 3, spacing=500.0)
    um_per_px = 0.650

    result = stage_coarse_transform(meta, meta.copy(), um_per_px, um_per_px)

    assert np.allclose(result["translation"], [0.0, 0.0], atol=1e-9)
    assert np.isclose(result["scale"], 1.0)
    assert np.allclose(result["rotation"], np.eye(2))
    assert result["angle_deg"] == 0.0


# ---------------------------------------------------------------------------
# merge_pseudotiles tests
# ---------------------------------------------------------------------------


def _make_sbs_ph_synthetic():
    """Build a small SBS + PH cell pair for merge_pseudotiles tests.

    SBS: 2×2 tiles (step=360px, tile_shape=400×400, ~10% overlap), 120 cells per tile.
    PH: same 4 tiles; each PH tile's cells are the corresponding SBS cells rotated by a
    per-tile angle (drawn from ±1.5°) about that tile's centre.

    sbs_um_per_px = ph_um_per_px = 1.0; ph_meta = sbs_meta (identical stage frames)
    → stage_coarse_transform returns identity coarse (rotation=I, translation=(0,0)),
    so ph_ref = ph_global.  The per-tile rotations are small enough that
    evaluate_match/RANSAC converges; the purpose is to verify assignment-by-position
    and per-PH-tile hashing, not to stress RANSAC.
    """
    rng = np.random.default_rng(0)

    TILE_H, TILE_W = 400, 400
    STEP = 360  # 10% overlap (400 - 360 = 40 px per edge)
    N_CELLS = 120

    # SBS: 2×2 grid
    sbs_offsets = {0: (0, 0), 1: (0, STEP), 2: (STEP, 0), 3: (STEP, STEP)}

    # Stage metadata: y_pos/x_pos = tile offsets in µm (valid since um_per_px=1.0)
    sbs_meta = pd.DataFrame([
        {"tile": t, "y_pos": float(oy), "x_pos": float(ox)}
        for t, (oy, ox) in sbs_offsets.items()
    ])
    # Identical meta → stage_coarse_transform gives scale=1, translation=(0,0)
    ph_meta = sbs_meta.copy()

    # SBS cells: uniformly inside each tile (avoid the 20px edge for clean per-tile membership)
    sbs_rows = []
    for t, (oy, ox) in sbs_offsets.items():
        ly = rng.uniform(20, TILE_H - 20, N_CELLS)
        lx = rng.uniform(20, TILE_W - 20, N_CELLS)
        for k, (y, x) in enumerate(zip(ly, lx)):
            sbs_rows.append({"plate": 1, "well": "A1", "tile": t, "cell": k,
                              "gy": oy + y, "gx": ox + x})
    sbs_cells = pd.DataFrame(sbs_rows)

    # PH cells: same physical positions, each tile rotated by a different small angle.
    # Angles ≤ 1.5° keep RANSAC reliable; different per-tile = the key test variation.
    angles_deg = rng.uniform(-1.5, 1.5, 4)
    ph_rows = []
    for t, (oy, ox) in sbs_offsets.items():
        # Rotate about this PH tile's centre
        cy = oy + TILE_H / 2
        cx = ox + TILE_W / 2
        theta = np.deg2rad(float(angles_deg[t]))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        for row in sbs_cells[sbs_cells["tile"] == t].itertuples(index=False):
            dy, dx = row.gy - cy, row.gx - cx
            ph_rows.append({
                "plate": 1, "well": "A1", "tile": t, "cell": row.cell,
                "gy": cy + cos_t * dy - sin_t * dx,
                "gx": cx + sin_t * dy + cos_t * dx,
            })
    ph_cells = pd.DataFrame(ph_rows)

    return sbs_cells, ph_cells, sbs_meta, ph_meta, sbs_offsets, (TILE_H, TILE_W)


@pytest.mark.unit
def test_merge_pseudotiles_matches_under_per_tile_varying_rotation():
    """Physical-position assignment survives per-PH-tile-varying rotation.

    A single fitted global rotation fails when each PH tile carries a different residual
    (the production bug this feature fixes).  Here we verify that merge_pseudotiles —
    which assigns PH cells to SBS tile footprints by physical stage position and then hashes
    per-PH-tile — achieves a healthy match rate even though no single rotation aligns all
    four PH tiles simultaneously.
    """
    sbs_cells, ph_cells, sbs_meta, ph_meta, sbs_offsets, tile_shape = _make_sbs_ph_synthetic()

    result = merge_pseudotiles(
        sbs_cells,
        ph_cells,
        sbs_meta,
        ph_meta,
        sbs_offsets,
        tile_shape,
        sbs_um_per_px=1.0,
        ph_um_per_px=1.0,
        margin_um=50.0,       # small margin reduces cross-tile ambiguity
        min_overlap_cells=30,
        threshold=4.0,
        local_refinement=True,
    )

    total_sbs = len(sbs_cells)

    # (a) Healthy fraction of SBS cells matched
    assert len(result) / total_sbs >= 0.60, (
        f"Expected ≥60% SBS cells matched, got {len(result)}/{total_sbs}"
    )

    # (b) Strict 1:1: no SBS cell matched twice, no PH cell matched twice
    assert not result.duplicated(subset=["cell_1", "subtile"]).any(), \
        "Duplicate SBS cell in result (not strict 1:1 on SBS side)"
    assert not result.duplicated(subset=["cell_0", "ph_tile"]).any(), \
        "Duplicate PH cell in result (not strict 1:1 on PH side)"

    # (c) Median match distance well within threshold
    assert float(result["distance"].median()) < 4.0, \
        f"Median match distance too large: {result['distance'].median():.3f}"


@pytest.mark.unit
def test_merge_pseudotiles_returns_expected_schema():
    """merge_pseudotiles returns correct column schema on happy-path and empty inputs."""
    rng = np.random.default_rng(42)
    TILE_H, TILE_W = 400, 400

    sbs_offsets = {0: (0, 0)}
    sbs_meta = pd.DataFrame([{"tile": 0, "y_pos": 0.0, "x_pos": 0.0}])
    ph_meta = sbs_meta.copy()

    # Happy path: identical SBS and PH cells → should match nearly all
    n = 60
    gy = rng.uniform(20, TILE_H - 20, n)
    gx = rng.uniform(20, TILE_W - 20, n)
    sbs_cells = pd.DataFrame({
        "plate": 1, "well": "A1", "tile": 0,
        "cell": np.arange(n), "gy": gy, "gx": gx,
    })
    ph_cells = sbs_cells.copy()

    result = merge_pseudotiles(
        sbs_cells, ph_cells, sbs_meta, ph_meta,
        sbs_offsets, (TILE_H, TILE_W), 1.0, 1.0,
        margin_um=50.0, min_overlap_cells=10, threshold=2.0,
    )

    required = {"subtile", "ph_tile", "cell_0", "cell_1", "distance"}
    assert required.issubset(result.columns), \
        f"Missing columns: {required - set(result.columns)}"

    # Empty-safe: PH cells far outside SBS footprint → no overlap → empty result
    ph_far = ph_cells.copy()
    ph_far["gy"] = ph_far["gy"] + 1_000_000
    ph_far["gx"] = ph_far["gx"] + 1_000_000

    empty = merge_pseudotiles(
        sbs_cells, ph_far, sbs_meta, ph_meta,
        sbs_offsets, (TILE_H, TILE_W), 1.0, 1.0,
        margin_um=50.0, min_overlap_cells=10, threshold=2.0,
    )
    assert len(empty) == 0, "Expected empty result for non-overlapping inputs"
    assert "subtile" in empty.columns and "ph_tile" in empty.columns, \
        "Empty result must still carry subtile/ph_tile columns"
