"""Tests for the positions-merge image readouts (`lib/merge/positions_overlay.py`).

Synthetic DAPI and nuclei-label tiles are rendered from one random nucleus field at known
stage positions, so a correct placement must line seams and modalities up exactly, and a
shift injected into one tile must be read back as that shift.
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
import tifffile  # noqa: E402
from skimage.draw import disk  # noqa: E402

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.merge import positions_overlay  # noqa: E402
from lib.merge.positions_overlay import (  # noqa: E402
    cross_modality_overlay,
    read_plane,
    save_figures,
    seam_overlay,
    tile_image_paths,
    well_mosaic,
)

TILE = 200
PH_PX, SBS_PX = 0.5, 1.0


def nuclei(seed=0, extent=400.0, n=900):
    rng = np.random.default_rng(seed)
    return rng.uniform(0, extent, (n, 2))


def render(points, center, pixel_size, labels=False):
    """Draw nuclei (3 um radius) into a TILE x TILE image centred on `center` (um)."""
    image = np.zeros((TILE, TILE), dtype=np.uint16 if not labels else np.uint32)
    half = (TILE - 1) / 2
    for k, (x, y) in enumerate(points, start=1):
        i, j = (y - center[1]) / pixel_size + half, (x - center[0]) / pixel_size + half
        if -5 < i < TILE + 5 and -5 < j < TILE + 5:
            rr, cc = disk((i, j), 3.0 / pixel_size, shape=image.shape)
            image[rr, cc] = k if labels else 1000
    return image


def placement(centers, pixel_size, shifts=None):
    tiles = pd.DataFrame.from_dict(centers, orient="index", columns=["x_pos", "y_pos"])
    tiles.index.name = "tile"
    shifts = shifts or {}
    tiles["dx"] = [shifts.get(t, (0.0, 0.0))[0] for t in tiles.index]
    tiles["dy"] = [shifts.get(t, (0.0, 0.0))[1] for t in tiles.index]
    tiles["source"] = "own"
    return {
        "pixel_size": pixel_size,
        "dimensions": (TILE, TILE),
        "orientation": (False, False, 0),
        "matrix": np.eye(2) * pixel_size,
        "radial": np.zeros(2),
        "offset": np.zeros(2),
        "tiles": tiles,
    }


@pytest.fixture
def screen(tmp_path):
    """Two phenotype tiles overlapping by 20 % in x and one SBS tile covering both."""
    points = nuclei()
    ph_centers = {0: (150.0, 150.0), 1: (150.0 + 0.8 * TILE * PH_PX, 150.0)}
    sbs_centers = {0: (190.0, 150.0)}
    paths = {"phenotype": {}, "sbs": {}}
    for name, centers, px in (
        ("phenotype", ph_centers, PH_PX),
        ("sbs", sbs_centers, SBS_PX),
    ):
        for tile, center in centers.items():
            path = tmp_path / f"{name}_{tile}.tiff"
            tifffile.imwrite(path, render(points, center, px))
            paths[name][tile] = str(path)
    return paths, ph_centers, sbs_centers


def test_seam_strip_is_the_overlap_and_aligned(screen):
    paths, ph_centers, _ = screen
    place = placement(ph_centers, PH_PX)
    record, panel = seam_overlay(place, paths["phenotype"], {}, 0, 0, 1)
    ref, mov, _, mask = panel
    assert ref.shape[0] == TILE
    assert abs(ref.shape[1] - 0.2 * TILE) <= 2
    assert record["residual_px"] == pytest.approx(0, abs=0.5)
    assert record["colored_fraction"] < 0.1


def test_injected_shift_is_read_back(screen):
    paths, ph_centers, _ = screen
    place = placement(ph_centers, PH_PX, shifts={1: (3 * PH_PX, 0.0)})
    record, _ = seam_overlay(place, paths["phenotype"], {}, 0, 0, 1)
    assert record["residual_px"] == pytest.approx(3, abs=1)
    assert abs(record["residual_dx_px"]) == pytest.approx(3, abs=1)
    aligned, _ = seam_overlay(
        placement(ph_centers, PH_PX), paths["phenotype"], {}, 0, 0, 1
    )
    assert record["colored_fraction"] > aligned["colored_fraction"] + 0.2


def test_cross_modality_crop_covers_the_phenotype_tile(screen):
    paths, ph_centers, sbs_centers = screen
    place = {
        "phenotype": placement(ph_centers, PH_PX),
        "sbs": placement(sbs_centers, SBS_PX),
    }
    record, panel = cross_modality_overlay(
        place, paths, {}, {"phenotype": 0, "sbs": 0}, 0, 0
    )
    ref, _, _, _ = panel
    expected = TILE * PH_PX / SBS_PX
    assert abs(ref.shape[0] - expected) <= 2 and abs(ref.shape[1] - expected) <= 2
    assert record["residual_px"] == pytest.approx(0, abs=1)


def test_cross_modality_uses_labels_on_both_sides_when_an_image_is_missing(
    screen, tmp_path
):
    paths, ph_centers, sbs_centers = screen
    points = nuclei()
    labels = {"phenotype": {}, "sbs": {}}
    for name, centers, px in (
        ("phenotype", ph_centers, PH_PX),
        ("sbs", sbs_centers, SBS_PX),
    ):
        for tile, center in centers.items():
            path = tmp_path / f"{name}_{tile}_labels.tiff"
            tifffile.imwrite(path, render(points, center, px, labels=True))
            labels[name][tile] = str(path)
    place = {
        "phenotype": placement(ph_centers, PH_PX),
        "sbs": placement(sbs_centers, SBS_PX),
    }
    images = {"phenotype": paths["phenotype"], "sbs": {}}
    record, panel = cross_modality_overlay(place, images, labels, {}, 0, 0)
    ref, mov, _, _ = panel
    assert set(np.unique(ref)) <= {0.0, 1.0}
    assert record["residual_px"] == pytest.approx(0, abs=1)


def test_mosaic_size_is_bounded(screen, monkeypatch, tmp_path):
    points = nuclei()
    labels = tmp_path / "labels_0.tiff"
    tifffile.imwrite(labels, render(points, (150.0, 150.0), PH_PX, labels=True))
    place = placement({0: (150.0, 150.0)}, PH_PX)
    monkeypatch.setattr(positions_overlay, "MOSAIC_MAX_PX", 64)
    image, um_per_px = well_mosaic(place, {0: str(labels)})
    assert max(image.shape[:2]) <= 65
    assert um_per_px >= PH_PX
    assert image.max() == 1


def test_read_plane_and_paths(tmp_path):
    stack = np.arange(3 * 8 * 8, dtype=np.uint16).reshape(3, 8, 8)
    path = tmp_path / "P-1_W-A1_T-5__aligned.tiff"
    tifffile.imwrite(path, stack)
    assert np.array_equal(read_plane(path, channel=2), stack[2])
    assert np.array_equal(read_plane(path, channel=1, step=2), stack[1, ::2, ::2])
    template = str(tmp_path / "P-{plate}_W-{well}_T-{tile}__aligned.tiff")
    assert tile_image_paths(template, [5, 6], 1, "A1") == {5: str(path)}


def test_save_figures_writes_placeholders(tmp_path):
    out = [tmp_path / f"{k}.png" for k in ("seams", "cross", "mosaic")]
    save_figures({}, out)
    assert all(p.exists() and p.stat().st_size > 0 for p in out)


def test_summary_warns_on_micrometres_not_pixels():
    from lib.merge.positions_overlay import summarize_image_qc

    records = pd.DataFrame(
        {
            "kind": ["seam_phenotype"] * 3 + ["cross_modality"],
            "residual_px": [5.0, 6.0, 5.0, 0.0],
            "residual_um": [0.8, 1.0, 0.8, 0.0],
            "colored_fraction": [0.1, 0.1, 0.1, np.nan],
        }
    )
    assert not summarize_image_qc(records)["image_qc_warning"]
    records["residual_um"] = [8.0, 9.0, 8.0, 0.0]
    assert summarize_image_qc(records)["image_qc_warning"]
