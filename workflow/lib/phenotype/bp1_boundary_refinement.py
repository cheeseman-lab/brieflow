"""Configurable refinement of existing 53BP1 object boundaries.

Refinement must be enabled explicitly by the caller. It only removes pixels
from existing labels; it cannot discover objects absent from the input.
Only 53BP1 intensities enter the cutoff formula. Other object masks can affect
which pixels in the supplied baseline estimate 53BP1 background.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.measure import regionprops

BOUNDARY_COLUMNS = [
    "object_id",
    "parent_nucleus_id",
    "original_area",
    "refined_area",
    "removed_fraction",
    "background",
    "background_robust_sd",
    "background_fallback",
    "cutoff",
    "original_mean",
    "refined_mean",
    "removed_mean",
    "excess_signal_retained",
    "bright_quartile_retained",
    "vanished",
    "refinement_status",
]


@dataclass(frozen=True)
class BoundarySpec:
    """Boundary settings in reference-image pixels and raw-intensity units."""

    method: str = "relative_contrast"
    threshold: float = 0.2
    smoothing_sigma: float = 0.7
    ring_inner: int = 2
    ring_outer: int = 8
    minimum_background_pixels: int = 30

    def __post_init__(self):
        """Reject invalid methods, scales and thresholds."""
        if self.method not in ("erosion", "background_z", "relative_contrast"):
            raise ValueError("Unknown boundary refinement method")
        if not np.isfinite(self.threshold) or self.threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        if self.method == "relative_contrast" and self.threshold >= 1:
            raise ValueError("relative contrast threshold must be below one")
        if self.method == "erosion" and int(self.threshold) != self.threshold:
            raise ValueError("erosion radius must be an integer")
        if not np.isfinite(self.smoothing_sigma) or self.smoothing_sigma < 0:
            raise ValueError("Invalid smoothing sigma")
        if not 0 <= self.ring_inner < self.ring_outer:
            raise ValueError("Invalid background annulus")
        if any(int(v) != v for v in (self.ring_inner, self.ring_outer)):
            raise ValueError("Background annulus radii must be integers")
        if (
            self.minimum_background_pixels < 1
            or int(self.minimum_background_pixels) != self.minimum_background_pixels
        ):
            raise ValueError("minimum_background_pixels must be a positive integer")


def robust_background(values):
    """Return a median and MAD-derived spread, with a one-count noise floor."""
    center = float(np.median(values))
    spread = max(1.0, float(1.4826 * np.median(abs(values - center))))
    return center, spread


def retain_peak_component(candidate, smoothed, original):
    """Retain the connected component containing the original brightest pixel.

    Only holes fully enclosed by this component are filled, inside the original
    mask. This is perimeter refinement, not carving out internal dark texture.
    An erosion that removes the peak or all pixels is an explicit lost object.
    """
    peak = np.unravel_index(
        np.argmax(np.where(original, smoothed, -np.inf)), original.shape
    )
    labels, _ = ndi.label(candidate, structure=np.ones((3, 3), bool))
    label = labels[peak]
    if label == 0:
        return np.zeros_like(original)
    return ndi.binary_fill_holes(labels == label) & original


def refine_one(mask, intensity, smooth, background, spread, spec):
    """Return one candidate footprint and its raw-intensity threshold."""
    if spec.method == "erosion":
        radius = int(spec.threshold)
        y, x = np.ogrid[-radius : radius + 1, -radius : radius + 1]
        candidate = ndi.binary_erosion(
            mask, structure=(x * x + y * y <= radius * radius)
        )
        cutoff = np.nan
    else:
        peak90 = float(np.percentile(smooth[mask], 90))
        cutoff = (
            background + spec.threshold * spread
            if spec.method == "background_z"
            else background + spec.threshold * max(0.0, peak90 - background)
        )
        candidate = mask & (smooth >= cutoff)
    return retain_peak_component(candidate, smooth, mask), cutoff


def refine_boundaries(image, labels, nuclei, baseline, spec):
    """Refine labels using their own parent and local unenhanced 53BP1 signal.

    Background comes from a 2–8 px annulus intersected with the supplied depleted
    parent. If too few pixels remain, the entire depleted parent is used. All
    methods use the same frozen original baseline for threshold estimation.
    Gaussian smoothing guides boundaries only; measurements use raw intensity.
    Labels and parents are preserved; removed objects are reported, not hidden.
    If a parent has no usable background, preserve its objects and mark them
    background_unavailable instead of inventing a threshold or failing a tile.
    Empty inputs return a table with a stable schema.
    """
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("Expected a two-dimensional intensity image")
    if any(np.asarray(a).shape != image.shape for a in (labels, nuclei, baseline)):
        raise ValueError("All arrays must share YX shape")
    if not np.isfinite(image).all():
        raise ValueError("Nonfinite image")
    if np.any((baseline > 0) & ((baseline != nuclei) | (labels > 0))):
        raise ValueError("Baseline must preserve parent IDs and exclude objects")
    output = np.zeros_like(labels)
    records = []
    # Work per parent so a neighboring nucleus cannot change smoothing or background.
    for parent in regionprops(nuclei):
        y0, x0, y1, x1 = parent.bbox
        pad = spec.ring_outer + 3
        sl = np.s_[
            max(0, y0 - pad) : min(image.shape[0], y1 + pad),
            max(0, x0 - pad) : min(image.shape[1], x1 + pad),
        ]
        domain = nuclei[sl] == parent.label
        raw = image[sl]
        sigma = spec.smoothing_sigma
        weights = ndi.gaussian_filter(domain.astype(float), sigma)
        smoothed = ndi.gaussian_filter(np.where(domain, raw, 0.0), sigma) / np.maximum(
            weights, 1e-12
        )
        local_labels = np.where(domain, labels[sl], 0)
        bodies = regionprops(local_labels)
        if not bodies:
            continue
        parent_baseline = baseline[sl] == parent.label
        if not parent_baseline.any():
            # Keep identity and expose the failure. Downstream baseline QC still
            # rejects measurements without a sufficiently large depleted parent.
            for body in bodies:
                mask = local_labels == body.label
                output[sl][mask] = body.label
                records.append(
                    dict(
                        object_id=int(body.label),
                        parent_nucleus_id=int(parent.label),
                        original_area=int(body.area),
                        refined_area=int(body.area),
                        removed_fraction=0.0,
                        background=np.nan,
                        background_robust_sd=np.nan,
                        background_fallback=True,
                        cutoff=np.nan,
                        original_mean=float(raw[mask].mean()),
                        refined_mean=float(raw[mask].mean()),
                        removed_mean=np.nan,
                        excess_signal_retained=np.nan,
                        bright_quartile_retained=1.0,
                        vanished=False,
                        refinement_status="background_unavailable",
                    )
                )
            continue
        for body in bodies:
            # Avoid repeated full-parent arrays for each small focus.
            by0, bx0, by1, bx1 = body.bbox
            small = np.s_[
                max(0, by0 - pad) : min(domain.shape[0], by1 + pad),
                max(0, bx0 - pad) : min(domain.shape[1], bx1 + pad),
            ]
            mask = local_labels[small] == body.label
            distance = ndi.distance_transform_edt(~mask)
            ring = (
                (distance > spec.ring_inner)
                & (distance <= spec.ring_outer)
                & parent_baseline[small]
            )
            fallback = ring.sum() < spec.minimum_background_pixels
            bg_values = raw[parent_baseline] if fallback else raw[small][ring]
            bg, spread = robust_background(bg_values)
            new, cutoff = refine_one(
                mask, raw[small], smoothed[small], bg, spread, spec
            )
            target = output[sl][small]
            assert not np.any(target[new] > 0)
            target[new] = body.label
            removed = mask & ~new
            excess = np.maximum(raw[small] - bg, 0)
            original_excess = float(excess[mask].sum())
            top = mask & (smoothed[small] >= np.percentile(smoothed[small][mask], 75))
            records.append(
                dict(
                    object_id=int(body.label),
                    parent_nucleus_id=int(parent.label),
                    original_area=int(mask.sum()),
                    refined_area=int(new.sum()),
                    removed_fraction=float(removed.sum() / mask.sum()),
                    background=bg,
                    background_robust_sd=spread,
                    background_fallback=fallback,
                    cutoff=cutoff,
                    original_mean=float(raw[small][mask].mean()),
                    refined_mean=float(raw[small][new].mean()) if new.any() else np.nan,
                    removed_mean=float(raw[small][removed].mean())
                    if removed.any()
                    else np.nan,
                    excess_signal_retained=float(excess[new].sum() / original_excess)
                    if original_excess > 0
                    else np.nan,
                    bright_quartile_retained=float((top & new).sum() / top.sum()),
                    vanished=not new.any(),
                    refinement_status="removed"
                    if not new.any()
                    else ("refined" if removed.any() else "unchanged"),
                )
            )
    assert not np.any((output > 0) & (output != labels))
    return output, pd.DataFrame(records, columns=BOUNDARY_COLUMNS)
