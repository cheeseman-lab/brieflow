"""Synthetic tests for the positions merge (`merge.approach: positions`).

A random cell field is imaged by a phenotype tile grid and a coarser SBS tile grid. Each
modality has its own camera scale, rotation and tile orientation, the tile stage positions
carry repeatability jitter, and the phenotype stage frame can be offset from the SBS one.
Every physical cell has an id, so a merge is scored exactly: a pair is correct when both
records come from the same physical cell, and no physical cell may appear twice.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.merge.positions_merge import (  # noqa: E402
    MERGE_COLUMNS,
    ORIENTATIONS,
    coarse_translation,
    orient_local,
    positions_merge,
    tile_pixel_to_um,
    um_to_tile_pixel,
)

TILE_PX = 300
PH_PX, SBS_PX = 0.6, 1.2
FIELD_UM = 1300.0


def make_screen(
    seed=0,
    n_cells=9000,
    orientation=(False, False, 0),
    ph_scale=1.0,
    sbs_scale=1.0,
    rotation_deg=0.0,
    overlap=0.1,
    offset_um=(0.0, 0.0),
    drop_ph_tiles=(),
    drop_sbs_tiles=(),
    radial_k1=0.0,
):
    """Simulate one well; return (inputs for positions_merge, physical-id lookups)."""
    rng = np.random.default_rng(seed)
    cells = _cell_field(rng, n=n_cells)
    ph = _image(
        rng,
        cells,
        PH_PX,
        ph_scale,
        rotation_deg,
        overlap,
        orientation,
        drop_ph_tiles,
        radial_k1,
    )
    sbs = _image(
        rng,
        cells,
        SBS_PX,
        sbs_scale,
        rotation_deg,
        overlap,
        orientation,
        drop_sbs_tiles,
        radial_k1,
    )
    ph[1]["x_pos"] += offset_um[0]
    ph[1]["y_pos"] += offset_um[1]
    return ph, sbs


def _cell_field(rng, n=9000, min_spacing=8.0):
    """Uniform random nuclei centroids with a minimum spacing (no lattice periodicity)."""
    points = rng.uniform(0, FIELD_UM, (n, 2))
    close = cKDTree(points).query_pairs(min_spacing, output_type="ndarray")
    return np.delete(points, np.unique(close[:, 1]), axis=0)


def _image(
    rng,
    cells,
    pixel_size,
    scale,
    rotation_deg,
    overlap,
    orientation,
    drop,
    radial_k1=0.0,
):
    """Image the cell field with one tile grid; return (info, metadata, physical ids)."""
    width_um = TILE_PX * pixel_size
    step = width_um * (1 - overlap)
    centers = np.arange(width_um / 2, FIELD_UM - width_um / 2 + step, step)
    theta = np.radians(rotation_deg)
    camera = (
        pixel_size
        * scale
        * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    )
    inverse = np.linalg.inv(camera)
    forward, offset = _orientation_affine(orientation)
    back = np.linalg.inv(forward)
    info, meta, ids = [], [], []
    tile = 0
    for cy in centers:
        for cx in centers:
            tile += 1
            if tile in drop:
                continue
            stage = np.array([cx, cy]) + rng.normal(0, 0.5, 2)
            u = (cells - stage) @ inverse.T
            for _ in range(6):
                rho2 = np.sum((u / ((TILE_PX - 1) / 2)) ** 2, axis=1)[:, None]
                u = (cells - stage - pixel_size * radial_k1 * rho2 * u) @ inverse.T
            inside = np.all(np.abs(u) < (TILE_PX - 1) / 2, axis=1)
            oriented = np.column_stack([u[inside, 1], u[inside, 0]]) + (TILE_PX - 1) / 2
            stored = (oriented - offset) @ back.T
            stored += rng.normal(0, 0.15, stored.shape)
            n = int(inside.sum())
            info.append(
                pd.DataFrame(
                    {
                        "plate": 1,
                        "well": "A1",
                        "tile": tile,
                        "cell": np.arange(1, n + 1),
                        "i": stored[:, 0],
                        "j": stored[:, 1],
                    }
                )
            )
            ids.append(
                pd.DataFrame(
                    {
                        "tile": tile,
                        "cell": np.arange(1, n + 1),
                        "pid": np.flatnonzero(inside),
                    }
                )
            )
            meta.append(
                {
                    "plate": 1,
                    "well": "A1",
                    "tile": tile,
                    "x_pos": cx,
                    "y_pos": cy,
                    "pixel_size_x": pixel_size,
                }
            )
    return (
        pd.concat(info, ignore_index=True),
        pd.DataFrame(meta),
        pd.concat(ids, ignore_index=True).set_index(["tile", "cell"])["pid"],
    )


def _orientation_affine(orientation):
    """Return (A, b) with orient_local((i, j)) = A @ (i, j) + b for square tiles."""
    probe_i = np.array([0.0, 1.0, 0.0])
    probe_j = np.array([0.0, 0.0, 1.0])
    oi, oj, _ = orient_local(probe_i, probe_j, (TILE_PX, TILE_PX), *orientation)
    b = np.array([oi[0], oj[0]])
    a = np.column_stack([[oi[1], oj[1]] - b, [oi[2], oj[2]] - b])
    return a, b


def run(ph, sbs, orientation=(False, False, 0), threshold=2):
    """Run the merge on a simulated well."""
    return run_full(ph, sbs, orientation, threshold)[:2]


def run_full(ph, sbs, orientation=(False, False, 0), threshold=2):
    """Run the merge and also return the placement."""
    flipud, fliplr, rot90 = orientation
    return positions_merge(
        ph[0],
        sbs[0],
        ph[1],
        sbs[1],
        (TILE_PX, TILE_PX),
        (TILE_PX, TILE_PX),
        threshold=threshold,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
    )


def score(merged, ph, sbs):
    """Return (precision, recall over physical cells seen by both, duplicate count)."""
    pid_0 = ph[2].loc[list(zip(merged["tile"], merged["cell_0"]))].values
    pid_1 = sbs[2].loc[list(zip(merged["site"], merged["cell_1"]))].values
    correct = pid_0 == pid_1
    both = np.intersect1d(ph[2].unique(), sbs[2].unique())
    duplicates = int(
        pd.Series(pid_0).duplicated().sum() + pd.Series(pid_1).duplicated().sum()
    )
    return correct.mean(), np.unique(pid_0[correct]).size / both.size, duplicates


def assert_good(merged, qc, ph, sbs):
    precision, recall, duplicates = score(merged, ph, sbs)
    assert precision >= 0.999, precision
    assert recall >= 0.99, recall
    assert duplicates == 0
    assert qc["status"].iloc[0] == "ok", qc.T


@pytest.mark.parametrize("orientation", ORIENTATIONS)
def test_every_orientation_is_recovered(orientation):
    ph, sbs = make_screen(seed=1, orientation=orientation)
    merged, qc = run(ph, sbs, orientation)
    assert_good(merged, qc, ph, sbs)
    assert not qc["phenotype_orientation_warning"].iloc[0]


def test_wrong_orientation_is_flagged_not_overridden():
    ph, sbs = make_screen(seed=2, orientation=(False, True, 0))
    merged, qc = run(ph, sbs, (False, False, 0))
    assert qc["phenotype_orientation_warning"].iloc[0]
    assert (
        qc["phenotype_best_orientation"].iloc[0] == "flipud=False,fliplr=True,rot90=0"
    )
    assert qc["status"].iloc[0] != "ok"
    assert len(merged) < 0.2 * len(ph[0])


@pytest.mark.parametrize(
    "ph_scale,sbs_scale", [(0.98, 1.02), (1.02, 0.98), (1.01, 1.01)]
)
def test_camera_scale_error(ph_scale, sbs_scale):
    ph, sbs = make_screen(seed=3, ph_scale=ph_scale, sbs_scale=sbs_scale)
    merged, qc = run(ph, sbs)
    assert_good(merged, qc, ph, sbs)
    assert qc["phenotype_camera_scale"].iloc[0] == pytest.approx(ph_scale, abs=0.002)


def test_camera_rotation():
    ph, sbs = make_screen(seed=4, rotation_deg=0.5)
    merged, qc = run(ph, sbs)
    assert_good(merged, qc, ph, sbs)
    assert qc["sbs_camera_rotation_deg"].iloc[0] == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("overlap", [0.05, 0.1, 0.2])
def test_overlap_cells_kept_once(overlap):
    ph, sbs = make_screen(seed=5, overlap=overlap)
    merged, qc = run(ph, sbs)
    assert_good(merged, qc, ph, sbs)
    assert qc["n_phenotype_kept"].iloc[0] == ph[2].nunique()
    assert qc["n_sbs_kept"].iloc[0] == sbs[2].nunique()


def test_missing_tiles_keep_neighbour_copies():
    ph, sbs = make_screen(
        seed=6, overlap=0.2, drop_ph_tiles=(8, 9), drop_sbs_tiles=(5,)
    )
    merged, qc = run(ph, sbs)
    assert_good(merged, qc, ph, sbs)
    assert qc["n_phenotype_kept"].iloc[0] == ph[2].nunique()


def test_stage_offset_between_modalities():
    ph, sbs = make_screen(seed=7, offset_um=(100.0, -60.0))
    merged, qc = run(ph, sbs)
    assert_good(merged, qc, ph, sbs)
    assert qc["translation_x_um"].iloc[0] == pytest.approx(-100.0, abs=2)
    assert qc["translation_y_um"].iloc[0] == pytest.approx(60.0, abs=2)


def test_coarse_translation_recovers_shift():
    rng = np.random.default_rng(8)
    points = rng.uniform(0, 2000, (4000, 2))
    shift, ratio = coarse_translation(points, points + [130.0, -70.0])
    assert np.allclose(shift, [130.0, -70.0], atol=10)
    assert ratio > 3


def test_orient_local_matches_image_transform():
    image = np.zeros((7, 5))
    image[1, 3] = 1
    for flipud, fliplr, rot90 in ORIENTATIONS + [(True, False, 3), (False, True, 2)]:
        moved = image
        if flipud:
            moved = np.flip(moved, axis=-2)
        if fliplr:
            moved = np.flip(moved, axis=-1)
        moved = np.rot90(moved, k=rot90, axes=(-2, -1))
        i, j, dims = orient_local(
            np.array([1.0]), np.array([3.0]), (7, 5), flipud, fliplr, rot90
        )
        assert moved.shape == dims
        assert moved[int(i[0]), int(j[0])] == 1


def test_output_schema_matches_fast_merge():
    ph, sbs = make_screen(seed=9)
    merged, qc = run(ph, sbs)
    assert list(merged.columns) == MERGE_COLUMNS
    assert len(qc) == 1
    assert (merged["distance"] < 2).all()


def test_too_few_cells_returns_empty():
    ph, sbs = make_screen(seed=10)
    empty = (ph[0].iloc[:2], ph[1], ph[2])
    merged, qc, placement = run_full(empty, sbs)
    assert placement is None
    assert merged.empty and list(merged.columns) == MERGE_COLUMNS
    assert qc["status"].iloc[0] == "insufficient_cells"


def test_sparse_tiles_fall_back_to_neighbours_or_global_model():
    ph, sbs = make_screen(seed=11, n_cells=350, rotation_deg=0.3, ph_scale=1.01)
    per_tile = ph[0].groupby("tile").size().reindex(ph[1]["tile"], fill_value=0)
    assert per_tile.median() <= 8 and (per_tile == 0).any()
    merged, qc = run(ph, sbs)
    precision, recall, duplicates = score(merged, ph, sbs)
    assert precision >= 0.999 and recall >= 0.99 and duplicates == 0
    fallback = (
        qc["phenotype_tiles_shift_neighbour"].iloc[0]
        + qc["phenotype_tiles_shift_global"].iloc[0]
    )
    assert fallback > 0


def test_tile_pixel_mapping_reproduces_matched_positions():
    ph, sbs = make_screen(seed=12, orientation=(True, False, 1), rotation_deg=0.4)
    merged, qc, placement = run_full(ph, sbs, (True, False, 1))
    rows = merged.sample(200, random_state=0)
    for _, row in rows.iterrows():
        p0 = tile_pixel_to_um(
            placement["phenotype"], row["tile"], [[row["i_0"], row["j_0"]]]
        )
        p1 = tile_pixel_to_um(placement["sbs"], row["site"], [[row["i_1"], row["j_1"]]])
        assert np.hypot(*(p0 - p1)[0]) < 2 * SBS_PX
        back = um_to_tile_pixel(placement["sbs"], row["site"], p1)
        assert np.allclose(back, [[row["i_1"], row["j_1"]]], atol=1e-3)


def test_radial_lens_distortion():
    ph, sbs = make_screen(seed=13, radial_k1=0.004)
    merged, qc = run(ph, sbs)
    assert_good(merged, qc, ph, sbs)
    assert qc["phenotype_radial_edge_um"].iloc[0] == pytest.approx(
        0.004 * PH_PX * (TILE_PX - 1) / 2, abs=0.15
    )
