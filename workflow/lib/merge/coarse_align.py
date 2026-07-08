"""Hash-free coarse DAPI overlay alignment between SBS and phenotype modalities (tier 2)."""

from __future__ import annotations

import numpy as np
from scipy.fft import fft2, ifft2
from skimage.registration import phase_cross_correlation
from skimage.transform import rescale, rotate

from workflow.lib.shared.stitching.types import TileOffsets


def build_coarse_mosaic(
    planes: dict[int, np.ndarray],
    offsets: TileOffsets,
    src_um_per_px: float,
    target_um_per_px: float = 8.0,
) -> np.ndarray:
    """Build a downsampled coarse mosaic from per-tile 2D DAPI planes.

    Args:
        planes: Mapping from tile index to 2D float DAPI plane.
        offsets: TileOffsets with per-tile (y, x) positions in source pixels.
        src_um_per_px: Physical pixel size of the source images (µm/px).
        target_um_per_px: Physical pixel size of the output mosaic (µm/px).

    Returns:
        2D float32 mosaic at target_um_per_px resolution.
    """
    scale = src_um_per_px / target_um_per_px
    off_frame = offsets.to_frame().set_index("tile")

    sample = next(iter(planes.values()))
    tile_h, tile_w = sample.shape
    coarse_h = max(1, int(round(tile_h * scale)))
    coarse_w = max(1, int(round(tile_w * scale)))

    canvas_h, canvas_w = 0, 0
    for tile_idx in planes:
        if tile_idx not in off_frame.index:
            continue
        oy = int(round(float(off_frame.loc[tile_idx, "y"]) * scale))
        ox = int(round(float(off_frame.loc[tile_idx, "x"]) * scale))
        canvas_h = max(canvas_h, oy + coarse_h)
        canvas_w = max(canvas_w, ox + coarse_w)

    canvas = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for tile_idx, plane in planes.items():
        if tile_idx not in off_frame.index:
            continue
        downsampled = rescale(
            plane.astype(np.float32), scale, anti_aliasing=True, channel_axis=None
        )
        oy = int(round(float(off_frame.loc[tile_idx, "y"]) * scale))
        ox = int(round(float(off_frame.loc[tile_idx, "x"]) * scale))
        dh, dw = downsampled.shape
        ch = min(dh, canvas_h - oy)
        cw = min(dw, canvas_w - ox)
        if ch > 0 and cw > 0:
            canvas[oy : oy + ch, ox : ox + cw] = downsampled[:ch, :cw]

    return canvas


def register_coarse(
    mosaic_ref: np.ndarray,
    mosaic_mov: np.ndarray,
    max_rotation_deg: float = 3.0,
    rotation_step_deg: float = 0.25,
) -> tuple[float, np.ndarray]:
    """Find the rotation + shift that aligns mosaic_mov onto mosaic_ref.

    Tests candidate rotations in [-max_rotation_deg, +max_rotation_deg] using
    normalized cross-correlation (NCC) peak as quality metric; picks the angle
    with the highest NCC peak, then computes sub-pixel shift via
    phase_cross_correlation at the best angle.

    Args:
        mosaic_ref: 2D reference mosaic (SBS).
        mosaic_mov: 2D moving mosaic (phenotype).
        max_rotation_deg: Half-range of rotation search in degrees.
        rotation_step_deg: Angular step size in degrees.

    Returns:
        Tuple (best_angle_deg, shift_yx) where shift_yx is in coarse-grid pixels.
        A positive best_angle_deg means mosaic_mov is rotated CCW by that angle
        to best align with mosaic_ref.
    """
    h = max(mosaic_ref.shape[0], mosaic_mov.shape[0])
    w = max(mosaic_ref.shape[1], mosaic_mov.shape[1])

    def _pad(arr: np.ndarray) -> np.ndarray:
        out = np.zeros((h, w), dtype=np.float32)
        out[: arr.shape[0], : arr.shape[1]] = arr
        return out

    def _norm(arr: np.ndarray) -> np.ndarray:
        mu = float(arr.mean())
        sigma = float(arr.std())
        return (arr - mu) / (sigma + 1e-8)

    ref = _norm(_pad(mosaic_ref.astype(np.float32)))
    mov_padded = _pad(mosaic_mov.astype(np.float32))

    F_ref = fft2(ref)
    ref_norm = float(np.linalg.norm(ref))

    angles = np.arange(
        -max_rotation_deg,
        max_rotation_deg + rotation_step_deg * 0.5,
        rotation_step_deg,
    )

    best_angle = 0.0
    best_ncc = -np.inf

    for angle in angles:
        rotated = _norm(rotate(mov_padded, angle, preserve_range=True).astype(np.float32))
        xps = F_ref * np.conj(fft2(rotated))
        rot_norm = float(np.linalg.norm(rotated))
        ncc_peak = float(np.real(ifft2(xps)).max()) / (ref_norm * rot_norm + 1e-12)
        if ncc_peak > best_ncc:
            best_ncc = ncc_peak
            best_angle = float(angle)

    best_rotated = _norm(
        rotate(mov_padded, best_angle, preserve_range=True).astype(np.float32)
    )
    shift, _, _ = phase_cross_correlation(
        ref, best_rotated, upsample_factor=4, normalization=None
    )

    return best_angle, np.asarray(shift, dtype=float)


def coarse_align_dapi(
    sbs_planes: dict[int, np.ndarray],
    ph_planes: dict[int, np.ndarray],
    sbs_offsets: TileOffsets,
    ph_offsets: TileOffsets,
    sbs_um_per_px: float,
    ph_um_per_px: float,
    target_um_per_px: float = 8.0,
    max_rotation_deg: float = 3.0,
) -> dict:
    """Coarse DAPI overlay alignment between SBS (reference) and phenotype (moving).

    Builds downsampled mosaics for both modalities at a common target resolution,
    registers them with an NCC-guided rotation grid search + phase cross-correlation,
    and converts the result to a transform mapping PH global pixel coords →
    SBS global pixel coords.

    Args:
        sbs_planes: Tile → 2D DAPI plane for SBS modality.
        ph_planes: Tile → 2D DAPI plane for phenotype modality.
        sbs_offsets: Per-tile pixel offsets for SBS.
        ph_offsets: Per-tile pixel offsets for phenotype.
        sbs_um_per_px: SBS pixel size (µm/px).
        ph_um_per_px: Phenotype pixel size (µm/px).
        target_um_per_px: Common downsampled resolution for coarse mosaics (µm/px).
        max_rotation_deg: Half-range for rotation search (degrees).

    Returns:
        Dict with keys:
            rotation (np.ndarray, shape 2×2): Combined rotation+scale matrix mapping
                PH pixels to SBS pixels (scale * R where R is the 2D rotation matrix).
            translation (np.ndarray, shape (2,)): Translation in SBS pixels.
            angle_deg (float): Recovered rotation in degrees applied to PH to reach SBS.
            scale (float): ph_um_per_px / sbs_um_per_px.
    """
    mosaic_sbs = build_coarse_mosaic(sbs_planes, sbs_offsets, sbs_um_per_px, target_um_per_px)
    mosaic_ph = build_coarse_mosaic(ph_planes, ph_offsets, ph_um_per_px, target_um_per_px)

    angle_deg, shift_coarse = register_coarse(
        mosaic_sbs, mosaic_ph, max_rotation_deg=max_rotation_deg
    )

    theta = np.deg2rad(angle_deg)
    R = np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]],
        dtype=np.float64,
    )

    scale = ph_um_per_px / sbs_um_per_px
    coarse_to_sbs = target_um_per_px / sbs_um_per_px

    # skimage.transform.rotate defaults to rotating about the image center, but the
    # application in image_stitch_merge.py uses R @ ph + t (rotation about the ORIGIN).
    # Fold the center-compensation term into the translation so that applying R @ ph + t
    # correctly reproduces the center-rotation measured by register_coarse.
    padded_h = max(mosaic_sbs.shape[0], mosaic_ph.shape[0])
    padded_w = max(mosaic_sbs.shape[1], mosaic_ph.shape[1])
    center_coarse = np.array([padded_h / 2.0, padded_w / 2.0])
    center_compensation = (np.eye(2) - R) @ center_coarse
    translation = (np.asarray(shift_coarse, dtype=np.float64) + center_compensation) * coarse_to_sbs

    return {
        "rotation": scale * R,
        "translation": translation,
        "angle_deg": angle_deg,
        "scale": scale,
    }
