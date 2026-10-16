"""Experimental cross-round registration and nucleolar exclusion for MN calls.

Registration uses DAPI only; marker overlap is never used to fit the transform.
All exclusion operations preserve the original 53BP1 object IDs and footprints.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage import measure, filters, morphology
from skimage.registration import phase_cross_correlation

from lib.phenotype.nuclear_bodies import (
    NuclearObjectSpec,
    _smooth_inside,
    segment_nuclear_objects,
    contact_table,
    validate_labels,
)


def _registration_image(image):
    image = np.asarray(image, dtype=np.float32)
    if image.ndim != 2 or not np.isfinite(image).all() or np.ptp(image) == 0:
        raise ValueError("Registration requires a finite, nonconstant 2D DAPI image")
    lo, hi = np.percentile(image, [1, 99.5])
    if hi <= lo:
        raise ValueError("Insufficient DAPI contrast for registration")
    return ndi.gaussian_filter(np.clip((image - lo) / (hi - lo), 0, 1), 1)


def align_round(reference_dapi, moving_dapi, moving_marker, max_shift=128):
    """Translate one imaging round to reference DAPI, retaining valid support.

    Subpixel interpolation is linear for intensities. No label masks are warped.
    Invalid edge bands are returned explicitly, never treated as negative signal.
    Regional residuals diagnose departures from a single global translation.
    """
    ref, mov = _registration_image(reference_dapi), _registration_image(moving_dapi)
    marker = np.asarray(moving_marker, dtype=np.float32)
    if (
        ref.shape != mov.shape
        or marker.shape != ref.shape
        or not np.isfinite(marker).all()
    ):
        raise ValueError("Round images must share a finite 2D shape")
    shift, error, _ = phase_cross_correlation(
        ref, mov, upsample_factor=10, normalization=None
    )
    if not np.isfinite(shift).all() or np.linalg.norm(shift) > max_shift:
        raise ValueError(f"Unacceptable DAPI registration shift: {shift}")
    aligned = ndi.shift(
        marker, shift, order=1, mode="constant", cval=0, prefilter=False
    )
    support = (
        ndi.shift(
            np.ones(ref.shape, np.float32),
            shift,
            order=1,
            mode="constant",
            cval=0,
            prefilter=False,
        )
        > 0.999
    )
    registered_dapi = ndi.shift(
        np.asarray(moving_dapi, np.float32),
        shift,
        order=1,
        mode="constant",
        cval=0,
        prefilter=False,
    )
    registered = ndi.shift(
        mov, shift, order=1, mode="constant", cval=0, prefilter=False
    )
    residuals = []
    patch = min(512, *ref.shape)
    for y in np.unique(np.linspace(0, ref.shape[0] - patch, 3).astype(int)):
        for x in np.unique(np.linspace(0, ref.shape[1] - patch, 3).astype(int)):
            sl = np.s_[y : y + patch, x : x + patch]
            if support[sl].mean() < 0.98 or ref[sl].std() < 0.02:
                continue
            residual, _, _ = phase_cross_correlation(
                ref[sl], registered[sl], upsample_factor=10, normalization=None
            )
            residuals.append(residual.tolist())
    stats = {
        "shift_yx_to_reference": shift.tolist(),
        "phase_error": float(error),
        "dapi_correlation_before": float(np.corrcoef(ref[support], mov[support])[0, 1]),
        "dapi_correlation_after": float(
            np.corrcoef(ref[support], registered[support])[0, 1]
        ),
        "valid_fraction": float(support.mean()),
        "regional_residual_yx": residuals,
        "regional_residual_max_px": float(np.linalg.norm(residuals, axis=1).max())
        if residuals
        else None,
        "regional_residual_median_px": float(
            np.median(np.linalg.norm(residuals, axis=1))
        )
        if residuals
        else None,
    }
    return aligned, registered_dapi, support, stats


def common_nuclei(nuclei, *supports):
    """Drop entire nuclei lacking complete support in any required round."""
    nuclei = validate_labels(nuclei)
    valid = np.ones(nuclei.shape, bool)
    for support in supports:
        if np.asarray(support).shape != nuclei.shape:
            raise ValueError("Registration support must match nucleus shape")
        valid &= support
    missing = np.unique(nuclei[~valid])
    missing = missing[missing > 0]
    return np.where(np.isin(nuclei, missing), 0, nuclei), missing


@dataclass(frozen=True)
class NucleolusSpec:
    """Pol II-depleted interiors, guarded against low-signal nuclear rims.

    feature_diameter, adaptive_window, and deviations apply only to dark_spots.
    relative_depletion uses Otsu capped at depletion_fraction times nuclear P80.
    All distances and sizes are in pixels of the registered reference image.
    """

    feature_diameter: int = 40
    diameter_min: float = 5
    diameter_max: float = 60
    adaptive_window: int = 60
    deviations: float = 2
    smoothing_scale: float = 2
    boundary_margin: int = 3
    max_parent_fraction: float = 0.4
    method: str = "relative_depletion"
    depletion_fraction: float = 0.65

    def __post_init__(self):
        """Reject invalid spatial and contrast settings before processing images."""
        if not 0 < self.diameter_min <= self.diameter_max < np.inf:
            raise ValueError("Nucleolar diameters must be finite, positive, ordered")
        if not np.isfinite(self.smoothing_scale) or self.smoothing_scale < 0:
            raise ValueError("smoothing_scale must be finite and nonnegative")
        if not isinstance(self.boundary_margin, int) or self.boundary_margin < 0:
            raise ValueError("boundary_margin must be a nonnegative integer")
        if not 0 < self.max_parent_fraction <= 1:
            raise ValueError("max_parent_fraction must lie in (0,1]")
        if not 0 < self.depletion_fraction < 1:
            raise ValueError("depletion_fraction must lie in (0,1)")
        if self.method not in {"relative_depletion", "dark_spots"}:
            raise ValueError("Unknown nucleolus method")


def segment_nucleoli(polii, nuclei, spec=None):
    """Detect interior Pol II-depleted pockets without using RNP or MN labels.

    These are nucleolar proxies, not an independent nucleolar ground truth.
    Shape/size guards and parent-relative contrast are documented in the output.
    """
    spec = (
        NucleolusSpec(**spec) if isinstance(spec, dict) else (spec or NucleolusSpec())
    )
    if not isinstance(spec.boundary_margin, int) or spec.boundary_margin < 0:
        raise ValueError("boundary_margin must be a nonnegative integer")
    if not 0 < spec.max_parent_fraction <= 1:
        raise ValueError("max_parent_fraction must lie in (0,1]")
    nuclei = validate_labels(nuclei)
    if spec.method == "relative_depletion":
        return _segment_polii_depletion(polii, nuclei, spec)
    if spec.method != "dark_spots":
        raise ValueError("Unknown nucleolus method")
    inner = nuclei.copy()
    for r in measure.regionprops(nuclei):
        roi = nuclei[r.slice] == r.label
        distance = ndi.distance_transform_edt(np.pad(roi, 1))[1:-1, 1:-1]
        inner[r.slice][roi & (distance <= spec.boundary_margin)] = 0
    detection = NuclearObjectSpec(
        feature_diameter=spec.feature_diameter,
        diameter_min=spec.diameter_min,
        diameter_max=spec.diameter_max,
        adaptive_window=spec.adaptive_window,
        threshold_smoothing_scale=spec.smoothing_scale,
        declump_smoothing_size=5,
        seed_min_distance=3,
        deviations=spec.deviations,
    )
    labels, parents, debug = segment_nuclear_objects(
        -np.asarray(polii, dtype=float), inner, detection
    )
    rows, rejected = [], []
    parent_areas = dict(zip(*np.unique(nuclei[nuclei > 0], return_counts=True)))
    parent_lookup = parents.set_index("object_id").parent_nucleus_id.to_dict()
    edge_lookup = parents.set_index("object_id").touches_nuclear_boundary.to_dict()
    for r in measure.regionprops(labels):
        parent = parent_lookup[r.label]
        fraction = r.area / parent_areas[parent]
        reason = (
            "inner_boundary"
            if edge_lookup[r.label]
            else "too_large_fraction"
            if fraction > spec.max_parent_fraction
            else "accepted"
        )
        rows.append(
            {
                "object_id": r.label,
                "parent_nucleus_id": parent,
                "area": r.area,
                "parent_fraction": fraction,
                "status": reason,
            }
        )
        if reason != "accepted":
            rejected.append(r.label)
    labels[np.isin(labels, rejected)] = 0
    return (
        labels,
        pd.DataFrame(
            rows,
            columns=[
                "object_id",
                "parent_nucleus_id",
                "area",
                "parent_fraction",
                "status",
            ],
        ),
        debug,
    )


def _segment_polii_depletion(polii, nuclei, spec):
    """Select substantial interior low-Pol-II regions; do not split dark pockets."""
    polii = np.asarray(polii, dtype=float)
    if polii.shape != nuclei.shape or not np.isfinite(polii).all():
        raise ValueError("Pol II image must match nuclei and be finite")
    if not 0 < spec.depletion_fraction < 1:
        raise ValueError("depletion_fraction must lie in (0,1)")
    labels = np.zeros(nuclei.shape, np.int32)
    threshold_image = np.full(nuclei.shape, np.nan)
    rows = []
    next_id = 1
    for parent in measure.regionprops(nuclei):
        sl = parent.slice
        roi = nuclei[sl] == parent.label
        distance = ndi.distance_transform_edt(np.pad(roi, 1))[1:-1, 1:-1]
        inner = roi & (distance > spec.boundary_margin)
        if inner.sum() < 20:
            continue
        smooth = _smooth_inside(polii[sl], roi, spec.smoothing_scale / 1.3488)
        values = smooth[inner]
        if (
            np.ptp(values) <= max(1, abs(values.mean())) * 1e-10
            or np.percentile(values, 80) <= 0
        ):
            continue
        cutoff = min(
            filters.threshold_otsu(values),
            spec.depletion_fraction * np.percentile(values, 80),
        )
        threshold_image[sl][roi] = cutoff
        low = inner & (smooth < cutoff)
        # Opening removes single-pixel necks connecting a pocket to a dark rim.
        low = ndi.binary_opening(low, structure=morphology.disk(1))
        low = ndi.binary_fill_holes(low) & inner
        components = measure.label(low, connectivity=2)
        boundary = inner & ~ndi.binary_erosion(inner)
        for r in measure.regionprops(components):
            selected = components == r.label
            diameter = np.sqrt(4 * r.area / np.pi)
            status = "accepted"
            if np.any(selected & boundary):
                status = "inner_boundary"
            elif not spec.diameter_min <= diameter <= spec.diameter_max:
                status = "size"
            elif r.area / roi.sum() > spec.max_parent_fraction:
                status = "too_large_fraction"
            oid = next_id if status == "accepted" else 0
            rows.append(
                {
                    "object_id": oid,
                    "parent_nucleus_id": parent.label,
                    "area": r.area,
                    "parent_fraction": r.area / roi.sum(),
                    "status": status,
                    "cutoff": cutoff,
                }
            )
            if oid:
                labels[sl][selected] = oid
                next_id += 1
    return (
        labels,
        pd.DataFrame(
            rows,
            columns=[
                "object_id",
                "parent_nucleus_id",
                "area",
                "parent_fraction",
                "status",
                "cutoff",
            ],
        ),
        {"threshold": threshold_image},
    )


def exclude_nucleolar_mn(mn, nucleoli, nuclei, connectivity=2):
    """Exclude entire MN objects touching any same-parent nucleolar object."""
    pairs = contact_table(mn, nucleoli, nuclei, connectivity)
    reject = np.isin(mn, pairs.a_id.unique()) & (mn > 0)
    return np.where(reject, 0, mn), np.where(reject, mn, 0), pairs
