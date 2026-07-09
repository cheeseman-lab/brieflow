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


def _make_multitile_synthetic():
    """Build a synthetic where each SBS footprint overlaps 4 PH tiles with distinct rotations.

    Physical region: [0,800]×[0,800] at unit scale (sbs_um_per_px = ph_um_per_px = 1.0).
    SBS: 2×2 tiles of 400×400.
    PH: 4×4 pseudo-tiles of 200×200; each of the 16 tiles is rotated by a different angle
        drawn from np.linspace(-8.0, 8.0, 16) degrees about its own centre.

    ph_meta = sbs_meta so stage_coarse_transform returns (scale=1, translation=(0,0)) and
    correspondence is purely by physical position. Each SBS tile footprint overlaps exactly
    4 PH tiles (e.g. SBS tile 0 sees PH tiles 0,1,4,5) whose angles span ~5.3°, making a
    single per-footprint rotation insufficient to align them all at threshold=4px.
    merge_pseudotiles hashes each PH tile independently and recovers high match rate;
    the naive merged-footprint RANSAC can only align ONE of the four groups and collapses.
    """
    rng = np.random.default_rng(0)

    SBS_H, SBS_W = 400, 400
    PH_H, PH_W = 200, 200
    N_PHYS = 2000

    # SBS 2×2 grid (step=400, no overlap)
    sbs_offsets = {0: (0, 0), 1: (0, 400), 2: (400, 0), 3: (400, 400)}
    sbs_meta = pd.DataFrame([
        {"tile": t, "y_pos": float(oy), "x_pos": float(ox)}
        for t, (oy, ox) in sbs_offsets.items()
    ])
    # ph_meta = sbs_meta → stage_coarse_transform returns identity (scale=1, translation=0)
    ph_meta = sbs_meta.copy()

    # Physical cell positions uniformly in [10, 790]×[10, 790]
    y_phys = rng.uniform(10, 790, N_PHYS)
    x_phys = rng.uniform(10, 790, N_PHYS)

    # SBS cells: gy/gx = physical position; tile = SBS quadrant (0–3)
    sbs_rows = []
    for k in range(N_PHYS):
        y, x = float(y_phys[k]), float(x_phys[k])
        t = min(int(y / SBS_H), 1) * 2 + min(int(x / SBS_W), 1)
        sbs_rows.append({"plate": 1, "well": "A1", "tile": t, "cell": k, "gy": y, "gx": x})
    sbs_cells = pd.DataFrame(sbs_rows)

    # PH cells: assign to 4×4 grid; each PH tile rotated by a distinct angle
    # Angles from -8° to +8° across 16 tiles — ~5.3° spread within each SBS footprint
    angles_deg = np.linspace(-8.0, 8.0, 16)
    ph_rows = []
    for k in range(N_PHYS):
        y, x = float(y_phys[k]), float(x_phys[k])
        ph_row = min(int(y / PH_H), 3)
        ph_col = min(int(x / PH_W), 3)
        ph_t = ph_row * 4 + ph_col

        # Rotate this cell about the PH tile's centre
        cy = ph_row * PH_H + PH_H / 2
        cx = ph_col * PH_W + PH_W / 2
        theta = np.deg2rad(float(angles_deg[ph_t]))
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dy, dx = y - cy, x - cx
        ph_rows.append({
            "plate": 1, "well": "A1", "tile": ph_t, "cell": k,
            "gy": cy + cos_t * dy - sin_t * dx,
            "gx": cx + sin_t * dy + cos_t * dx,
        })
    ph_cells = pd.DataFrame(ph_rows)

    return sbs_cells, ph_cells, sbs_meta, ph_meta, sbs_offsets, (SBS_H, SBS_W)


@pytest.mark.unit
def test_merge_pseudotiles_matches_under_per_tile_varying_rotation():
    """Per-PH-tile hashing survives the multi-tile-per-footprint scenario.

    Each SBS tile footprint overlaps 4 PH tiles each carrying a DIFFERENT rotation (angles
    span ~5.3° within one footprint). A single rotation fitted to the mixed footprint at
    threshold=4px can only align ONE of the four PH groups; merge_pseudotiles hashes each PH
    tile independently and achieves a high match rate.  The naive merge_reference_tiles
    (global_model=False, local_refinement=None) is shown to do materially worse, proving
    per-PH-tile hashing is what makes the difference.
    """
    sbs_cells, ph_cells, sbs_meta, ph_meta, sbs_offsets, tile_shape = _make_multitile_synthetic()
    total_sbs = len(sbs_cells)

    result = merge_pseudotiles(
        sbs_cells,
        ph_cells,
        sbs_meta,
        ph_meta,
        sbs_offsets,
        tile_shape,
        sbs_um_per_px=1.0,
        ph_um_per_px=1.0,
        margin_um=50.0,
        min_overlap_cells=30,
        threshold=4.0,
        local_refinement=True,
    )

    # (a) Healthy fraction of SBS cells matched
    assert len(result) / total_sbs >= 0.60, (
        f"merge_pseudotiles: expected ≥60% SBS cells matched, got {len(result)}/{total_sbs}"
    )

    # (b) Strict 1:1: no SBS cell matched twice (canonical 'site' key), no PH cell twice
    assert not result.duplicated(subset=["cell_1", "site"]).any(), \
        "Duplicate SBS cell in result (not strict 1:1 on SBS side)"
    assert not result.duplicated(subset=["cell_0", "ph_tile"]).any(), \
        "Duplicate PH cell in result (not strict 1:1 on PH side)"

    # (c) Median match distance well within threshold
    assert float(result["distance"].median()) < 4.0, \
        f"Median match distance too large: {result['distance'].median():.3f}"

    # (d) Contrast: naive merged-footprint RANSAC does materially worse WITHOUT per-tile TPS.
    # merge_reference_tiles mixes all PH tiles in each SBS footprint; a single RANSAC rotation
    # (no TPS) can only align ONE PH group at threshold=4px and misses the rest.
    # local_refinement=None here is intentional: TPS can hide the mixing by warping locally,
    # but the point is that the RANSAC seed itself fails when footprints carry multiple
    # distinct rotations — per-PH-tile hashing avoids that entirely.
    from workflow.lib.merge.image_stitch_merge import merge_reference_tiles
    from workflow.lib.shared.stitching.types import TileOffsets
    coarse = stage_coarse_transform(sbs_meta, ph_meta, 1.0, 1.0)
    # merge_reference_tiles requires TileOffsets (not a plain dict)
    offsets_frame = pd.DataFrame([
        {"tile": t, "y": float(v[0]), "x": float(v[1])}
        for t, v in sbs_offsets.items()
    ])
    sbs_offsets_obj = TileOffsets.from_frame(offsets_frame)
    naive = merge_reference_tiles(
        sbs_cells,
        ph_cells,
        coarse,
        sbs_offsets_obj,
        tile_shape=tile_shape,
        margin_px=50.0,
        threshold=4.0,
        local_refinement=None,
        global_model=False,
    )
    assert len(naive) < 0.6 * len(result), (
        f"Naive per-footprint RANSAC should do materially worse than per-PH-tile: "
        f"naive={len(naive)}, pseudotile={len(result)}"
    )


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

    required = {
        "cell_0", "i_0", "j_0", "site", "cell_1", "i_1", "j_1",
        "distance", "subtile", "ph_tile",
    }
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
