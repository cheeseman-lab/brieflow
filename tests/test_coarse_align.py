# tests/test_coarse_align.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from skimage.transform import rotate as sk_rotate

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.merge.coarse_align import (
    build_coarse_mosaic,
    register_coarse,
    coarse_align_dapi,
)
from workflow.lib.shared.stitching.types import TileOffsets


def _make_blob_image(shape, n_blobs=25, seed=0):
    rng = np.random.default_rng(seed)
    image = np.zeros(shape, dtype=np.float32)
    h, w = shape
    for _ in range(n_blobs):
        cy = rng.integers(20, h - 20)
        cx = rng.integers(20, w - 20)
        sigma = rng.uniform(4, 10)
        Y, X = np.ogrid[:h, :w]
        image += np.exp(-((Y - cy) ** 2 + (X - cx) ** 2) / (2 * sigma ** 2))
    return image


@pytest.mark.unit
def test_build_coarse_mosaic_two_tiles():
    tile_shape = (200, 200)
    src_um = 0.65
    tgt_um = 8.0
    scale = src_um / tgt_um

    planes = {
        0: _make_blob_image(tile_shape, seed=0),
        1: _make_blob_image(tile_shape, seed=1),
    }
    off_df = pd.DataFrame({"tile": [0, 1], "y": [0.0, 0.0], "x": [0.0, 200.0]})
    offsets = TileOffsets.from_frame(off_df)

    mosaic = build_coarse_mosaic(planes, offsets, src_um, tgt_um)

    coarse_tile_w = int(round(200 * scale))
    coarse_offset_x1 = int(round(200 * scale))
    expected_w = coarse_offset_x1 + coarse_tile_w
    expected_h = int(round(200 * scale))

    assert mosaic.shape == (expected_h, expected_w), (
        f"Expected ({expected_h}, {expected_w}), got {mosaic.shape}"
    )
    assert mosaic[:, :coarse_tile_w].mean() > 0, "Left tile region is empty"
    assert mosaic[:, coarse_offset_x1:].mean() > 0, "Right tile region is empty"


@pytest.mark.unit
def test_register_coarse_recovers_rotation():
    from skimage.transform import rotate as sk_rotate

    shape = (256, 256)
    ref = _make_blob_image(shape, n_blobs=30, seed=7)
    known_angle = 1.5
    mov = sk_rotate(ref, known_angle, preserve_range=True).astype(np.float32)

    best_angle, shift = register_coarse(
        ref, mov, max_rotation_deg=3.0, rotation_step_deg=0.25
    )

    assert abs(best_angle - (-known_angle)) < 0.3, (
        f"Expected angle ≈ {-known_angle}°, got {best_angle}°"
    )
    assert np.abs(shift).max() < 2.0, f"Shift too large: {shift}"


@pytest.mark.unit
def test_coarse_align_dapi_identical_images():
    tile_shape = (200, 200)
    src_um = 0.65
    tgt_um = 8.0

    planes = {
        0: _make_blob_image(tile_shape, seed=0),
        1: _make_blob_image(tile_shape, seed=1),
    }
    off_df = pd.DataFrame({"tile": [0, 1], "y": [0.0, 0.0], "x": [0.0, 200.0]})
    offsets = TileOffsets.from_frame(off_df)

    result = coarse_align_dapi(
        sbs_planes=planes,
        ph_planes=planes,
        sbs_offsets=offsets,
        ph_offsets=offsets,
        sbs_um_per_px=src_um,
        ph_um_per_px=src_um,
        target_um_per_px=tgt_um,
        max_rotation_deg=3.0,
    )

    assert set(result.keys()) >= {"rotation", "translation", "angle_deg", "scale"}
    assert abs(result["scale"] - 1.0) < 1e-6
    assert abs(result["angle_deg"]) < 1.0, (
        f"Expected angle ≈ 0°, got {result['angle_deg']}°"
    )
    assert result["rotation"].shape == (2, 2)
    assert result["translation"].shape == (2,)


@pytest.mark.unit
def test_coarse_align_dapi_applies_rotation_about_center():
    """Coarse transform R@ph+t must reproduce the center-rotation geometry exactly.

    BUG (pre-fix): coarse_align_dapi returned translation = shift_coarse * coarse_to_sbs.
    skimage.transform.rotate operates about the IMAGE CENTER, so the correct translation
    is (shift_coarse + (I-R) @ center_coarse) * coarse_to_sbs.  Without the center
    compensation, the application R@ph+t is off by (I-R)@center_coarse*coarse_to_sbs
    — for a 512x512 image at 2.5° that's ~15.8 px.

    Setup: PH mosaic = SBS mosaic rotated 2.5° about image center (no shift).
    Same pixel sizes → scale=1, coarse_to_sbs=1.
    Assert that applying the returned transform maps PH global coords → SBS global
    coords to < 1 px.  Without the fix the error is ~15.8 px (will FAIL RED).
    """
    shape = (512, 512)
    known_angle_deg = 2.5

    ref = _make_blob_image(shape, n_blobs=60, seed=42)
    mov = sk_rotate(ref, known_angle_deg, preserve_range=True).astype(np.float32)

    # Same pixel size for both → scale=1, coarse_to_sbs=1, coarse pixel ≡ global px
    um_per_px = 8.0  # = target, so src_scale = tgt_scale = 1
    off_df = pd.DataFrame({"tile": [0], "y": [0.0], "x": [0.0]})
    offsets = TileOffsets.from_frame(off_df)

    result = coarse_align_dapi(
        sbs_planes={0: ref},
        ph_planes={0: mov},
        sbs_offsets=offsets,
        ph_offsets=offsets,
        sbs_um_per_px=um_per_px,
        ph_um_per_px=um_per_px,
        target_um_per_px=um_per_px,
        max_rotation_deg=3.0,
    )

    R_hat = np.asarray(result["rotation"], dtype=np.float64)
    t_hat = np.asarray(result["translation"], dtype=np.float64)

    # Ground-truth: feature at p_ref in SBS corresponds to p_ph in PH where
    #   p_ph = R(+known_angle) @ (p_ref - center) + center
    # Inverse: p_ref = R(-known_angle) @ (p_ph - center) + center
    center = np.array([shape[0] / 2.0, shape[1] / 2.0])
    theta_inv = np.deg2rad(-known_angle_deg)
    R_true = np.array([[np.cos(theta_inv), -np.sin(theta_inv)],
                       [np.sin(theta_inv),  np.cos(theta_inv)]])

    rng = np.random.default_rng(99)
    ph_pts = rng.uniform(30, shape[0] - 30, size=(100, 2))
    sbs_pts_true = (R_true @ (ph_pts - center).T).T + center

    sbs_pts_hat = (R_hat @ ph_pts.T).T + t_hat

    residuals = np.linalg.norm(sbs_pts_hat - sbs_pts_true, axis=1)
    # Without fix: error ≈ |(I-R)@center| = 2.5° * 256 * sqrt(2) ≈ 15.8 px
    assert residuals.max() < 1.0, (
        f"Rotation-center bug: max residual {residuals.max():.2f} px "
        f"(expected < 1 px after fix; pre-fix error is ~"
        f"{float(np.linalg.norm((np.eye(2) - R_true) @ center)):.1f} px)"
    )
