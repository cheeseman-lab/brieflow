"""Pairwise tile-overlap registration via FFT cross-correlation."""

from __future__ import annotations

import numpy as np
from scipy.ndimage import shift as ndi_shift
from skimage.registration import phase_cross_correlation


def _zncc(a: np.ndarray, b: np.ndarray) -> float:
    """Zero-normalized cross-correlation of two equal-shape arrays."""
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    a -= a.mean()
    b -= b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def _overlap_strips(ref, mov, expected_shift, overlap_fraction):
    """Return the overlapping sub-windows of ref and mov given the prior shift."""
    h, w = ref.shape
    dy, dx = expected_shift
    # Horizontal neighbor (dx dominant): take right strip of ref, left strip of mov.
    if abs(dx) >= abs(dy):
        ow = max(8, int(round(w * overlap_fraction)))
        ref_s = ref[:, w - ow:]
        mov_s = mov[:, :ow]
    else:
        oh = max(8, int(round(h * overlap_fraction)))
        ref_s = ref[h - oh:, :]
        mov_s = mov[:oh, :]
    return ref_s, mov_s


def register_pair(ref, mov, expected_shift, overlap_fraction=0.1, max_shift=40.0):
    """Register neighbor tile ``mov`` against ``ref`` from image content.

    Args:
        ref: 2D reference tile plane.
        mov: 2D moving neighbor tile plane.
        expected_shift: (dy, dx) stage-prior displacement of mov vs ref, px.
        overlap_fraction: fraction of tile used as the overlap strip.
        max_shift: reject residuals larger than this (px) as unreliable.

    Returns:
        (shift_yx, confidence): refined full (dy, dx) displacement and its ZNCC.
    """
    ref_s, mov_s = _overlap_strips(ref, mov, expected_shift, overlap_fraction)
    residual, _, _ = phase_cross_correlation(
        ref_s, mov_s, upsample_factor=10, normalization=None
    )
    if np.linalg.norm(residual) > max_shift:
        residual = np.zeros(2)
    shift_yx = np.asarray(expected_shift, dtype=float) + residual
    aligned = ndi_shift(mov_s, shift=residual, order=1)
    conf = _zncc(ref_s, aligned)
    return shift_yx, conf
