"""Experimental, nucleus-restricted body segmentation and contact relationships.

Adapted from the supplied CellProfiler settings, not a bitwise CellProfiler port.
Detection images are separate from quantitative measurement images.
All coordinates and size parameters are in pixels; arrays are strictly 2D.
"""

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from scipy.interpolate import RegularGridInterpolator
from skimage import feature, measure, morphology, segmentation


@dataclass(frozen=True)
class NuclearObjectSpec:
    """Parameters for one channel; seed distance is an explicit adaptation."""

    feature_diameter: int = 16
    diameter_min: float = 3
    diameter_max: float = 30
    adaptive_window: int = 30
    threshold_smoothing_scale: float = 1.3
    declump_smoothing_size: float = 5
    seed_min_distance: int = 2
    trim_fraction: float = 0.05
    deviations: float = 4
    threshold_correction: float = 1
    fill_holes: bool = True

    def __post_init__(self):
        """Reject invalid settings before processing any image."""
        if not all(np.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Parameters must be finite")
        for key in ("feature_diameter", "adaptive_window", "seed_min_distance"):
            if not isinstance(getattr(self, key), int) or getattr(self, key) < 1:
                raise ValueError(f"{key} must be a positive integer")
        if not 0 < self.diameter_min <= self.diameter_max:
            raise ValueError("Require 0 < diameter_min <= diameter_max")
        if not 0 <= self.trim_fraction < 0.5:
            raise ValueError("trim_fraction must be in [0, 0.5)")
        if (
            min(
                self.threshold_smoothing_scale,
                self.declump_smoothing_size,
                self.deviations,
            )
            < 0
        ):
            raise ValueError("Smoothing and deviations must be nonnegative")
        if self.threshold_correction <= 0:
            raise ValueError("threshold_correction must be positive")


def validate_labels(labels, shape=None):
    """Validate a 2D nonnegative integer label image without relabeling."""
    labels = np.asarray(labels)
    if labels.ndim != 2 or labels.dtype.kind not in "iu":
        raise ValueError("Expected a 2D integer label image")
    if np.any(labels < 0) or (shape is not None and labels.shape != shape):
        raise ValueError("Negative labels or mismatched shape")
    return labels


def object_parents(labels, nuclei):
    """Require every object pixel to belong to exactly one nonzero nucleus."""
    nuclei = validate_labels(nuclei)
    labels = validate_labels(labels, nuclei.shape)
    occupied = labels > 0
    pairs = np.unique(np.column_stack((labels[occupied], nuclei[occupied])), axis=0)
    if pairs.size and (
        np.any(pairs[:, 1] == 0) or len(np.unique(pairs[:, 0])) != len(pairs)
    ):
        raise ValueError("Object outside nuclei or spanning multiple nucleus labels")
    return {int(a): int(b) for a, b in pairs}


def _smooth_inside(image, mask, sigma):
    if sigma == 0:
        return np.where(mask, image, 0.0)
    weights = ndi.gaussian_filter(mask.astype(float), sigma, mode="constant")
    total = ndi.gaussian_filter(np.where(mask, image, 0.0), sigma, mode="constant")
    return np.divide(
        total, weights, out=np.zeros_like(total), where=mask & (weights > 0)
    )


def _enhance_inside(image, mask, diameter):
    # Ignore excluded pixels in both morphology passes, including adjacent nuclei.
    footprint = morphology.disk(diameter / 2)
    eroded = ndi.minimum_filter(np.where(mask, image, np.inf), footprint=footprint)
    opened = ndi.maximum_filter(np.where(mask, eroded, -np.inf), footprint=footprint)
    return np.where(mask, np.maximum(image - opened, 0), 0)


def _robust_threshold(values, spec):
    values = np.sort(values)
    cut = int(len(values) * spec.trim_fraction)
    kept = values[cut : len(values) - cut] if cut else values
    return float(kept.mean() + spec.deviations * kept.std())


def _adaptive_threshold(image, mask, spec):
    # Per-nucleus guide and linearly interpolated block statistics are deliberate
    # differences from CellProfiler's image-wide guide and spline interpolation.
    guide = _robust_threshold(image[mask], spec) * spec.threshold_correction
    ny, nx = image.shape
    edges_y = np.arange(0, ny, spec.adaptive_window).tolist() + [ny]
    edges_x = np.arange(0, nx, spec.adaptive_window).tolist() + [nx]
    grid = np.full((len(edges_y) - 1, len(edges_x) - 1), guide)
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            sl = np.s_[edges_y[y] : edges_y[y + 1], edges_x[x] : edges_x[x + 1]]
            values = image[sl][mask[sl]]
            if len(values) >= 3:
                grid[y, x] = _robust_threshold(values, spec) * spec.threshold_correction
    cy = (np.array(edges_y[:-1]) + np.array(edges_y[1:]) - 1) / 2
    cx = (np.array(edges_x[:-1]) + np.array(edges_x[1:]) - 1) / 2
    yy, xx = np.meshgrid(
        np.clip(np.arange(ny), cy[0], cy[-1]),
        np.clip(np.arange(nx), cx[0], cx[-1]),
        indexing="ij",
    )
    threshold = RegularGridInterpolator(
        (cy, cx), grid, bounds_error=False, fill_value=None
    )((yy, xx))
    return np.clip(threshold, 0.7 * guide, 1.5 * guide)


def segment_nuclear_objects(channel, nuclei, spec, max_objects=None):
    """Enhance, threshold and watershed separately inside each primary nucleus.

    Args:
        channel: Corrected, registered 2D intensity image.
        nuclei: Integer primary labels; retained as parents.
        spec: NuclearObjectSpec or a dict of its parameters.
        max_objects: Optional hard failure limit, never converted to empty data.

    Returns:
        Labels, parent mapping table and enhanced/threshold diagnostic images.
    """
    nuclei = validate_labels(nuclei)
    channel = np.asarray(channel, dtype=float)
    if channel.shape != nuclei.shape or not np.isfinite(channel).all():
        raise ValueError("Channel must match nuclei and contain only finite values")
    spec = NuclearObjectSpec(**spec) if isinstance(spec, dict) else spec
    if not isinstance(spec, NuclearObjectSpec):
        raise TypeError("spec must be NuclearObjectSpec or dict")
    if max_objects is not None and (
        not isinstance(max_objects, int) or max_objects < 1
    ):
        raise ValueError("max_objects must be None or a positive integer")
    labels = np.zeros(nuclei.shape, np.int32)
    enhanced = np.zeros(nuclei.shape, float)
    thresholds = np.full(nuclei.shape, np.nan)
    rows = []
    candidate_rows = []
    candidate_columns = [
        "parent_nucleus_id",
        "candidate_id",
        "area",
        "equivalent_diameter",
        "status",
        "object_id",
    ]
    values = channel[nuclei > 0]
    if not len(values) or np.ptp(values) == 0:
        return (
            labels,
            pd.DataFrame(
                columns=["object_id", "parent_nucleus_id", "touches_nuclear_boundary"]
            ),
            {
                "enhanced": enhanced,
                "threshold": thresholds,
                "candidates": pd.DataFrame(columns=candidate_columns),
            },
        )
    scaled = (channel - values.min()) / np.ptp(values)
    # regionprops avoids allocation proportional to the highest sparse nucleus ID.
    ids, inverse = np.unique(nuclei, return_inverse=True)
    dense = inverse.reshape(nuclei.shape) + 1
    dense[nuclei == 0] = 0
    for nucleus in measure.regionprops(dense):
        parent_id = int(ids[nucleus.label - 1])
        sl = nucleus.slice
        roi = nuclei[sl] == parent_id
        local = _enhance_inside(scaled[sl], roi, spec.feature_diameter)
        enhanced[sl][roi] = local[roi]
        if not np.any(local[roi] > 0):
            continue
        threshold = _adaptive_threshold(local, roi, spec)
        thresholds[sl][roi] = threshold[roi]
        blurred = _smooth_inside(local, roi, spec.threshold_smoothing_scale / 1.3488)
        binary = roi & (blurred > threshold) & (blurred > 0)
        if spec.fill_holes:
            binary = ndi.binary_fill_holes(binary) & roi
        components = measure.label(binary, connectivity=2)
        if not components.any():
            continue
        peaks_image = _smooth_inside(local, roi, spec.declump_smoothing_size / 2.35)
        peaks = feature.peak_local_max(
            peaks_image,
            min_distance=spec.seed_min_distance,
            labels=components,
            exclude_border=False,
        )
        markers = np.zeros(roi.shape, np.int32)
        for marker_id, (y, x) in enumerate(peaks, 1):
            markers[y, x] = marker_id
        # Guarantee a seed for each component, even for flat plateaus.
        for component in measure.regionprops(components):
            coords = component.coords
            if not markers[coords[:, 0], coords[:, 1]].any():
                y, x = coords[np.argmax(peaks_image[coords[:, 0], coords[:, 1]])]
                markers[y, x] = int(markers.max()) + 1
        separated = segmentation.watershed(
            -peaks_image, markers, mask=binary, connectivity=2
        )
        boundary = roi & ~ndi.binary_erosion(roi, border_value=0)
        for region in measure.regionprops(separated):
            candidate = separated == region.label
            if spec.fill_holes:
                holes = ndi.binary_fill_holes(candidate) & roi & (separated == 0)
                candidate |= holes
            diameter = np.sqrt(4 * candidate.sum() / np.pi)
            status = (
                "below_min"
                if diameter < spec.diameter_min
                else "above_max"
                if diameter > spec.diameter_max
                else "accepted"
            )
            candidate_rows.append(
                {
                    "parent_nucleus_id": parent_id,
                    "candidate_id": int(region.label),
                    "area": int(candidate.sum()),
                    "equivalent_diameter": diameter,
                    "status": status,
                    "object_id": len(rows) + 1 if status == "accepted" else 0,
                }
            )
            if status != "accepted":
                continue
            object_id = len(rows) + 1
            if max_objects is not None and object_id > max_objects:
                raise RuntimeError("Object limit exceeded; tile is failed, not empty")
            labels[sl][candidate] = object_id
            rows.append(
                {
                    "object_id": object_id,
                    "parent_nucleus_id": parent_id,
                    "touches_nuclear_boundary": bool(np.any(candidate & boundary)),
                }
            )
    columns = ["object_id", "parent_nucleus_id", "touches_nuclear_boundary"]
    return (
        labels,
        pd.DataFrame(rows, columns=columns),
        {
            "enhanced": enhanced,
            "threshold": thresholds,
            "candidates": pd.DataFrame(candidate_rows, columns=candidate_columns),
        },
    )


def contact_table(a, b, nuclei, connectivity=2):
    """Return all same-nucleus overlaps/contacts, without choosing one partner.

    Args:
        a: First object labels.
        b: Second object labels.
        nuclei: Shared primary labels.
        connectivity: 0 for overlap only, 1 for overlap/edges, 2 adds corners.

    Returns:
        Pair table with raw overlap and adjacent-pixel-pair counts.
        Neither input mask is expanded in the output.
    """
    pa, pb = object_parents(a, nuclei), object_parents(b, nuclei)
    if connectivity not in (0, 1, 2):
        raise ValueError("connectivity must be 0, 1 or 2")
    offsets = [(0, 0)]
    if connectivity:
        offsets += [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 2:
        offsets += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    counts = {}
    h, w = nuclei.shape
    for dy, dx in offsets:
        sa = np.s_[max(0, -dy) : min(h, h - dy), max(0, -dx) : min(w, w - dx)]
        sb = np.s_[max(0, dy) : min(h, h + dy), max(0, dx) : min(w, w + dx)]
        aa, bb = a[sa], b[sb]
        keep = (aa > 0) & (bb > 0) & (nuclei[sa] == nuclei[sb])
        pairs, ns = np.unique(
            np.column_stack((aa[keep], bb[keep])), axis=0, return_counts=True
        )
        for (aid, bid), n in zip(pairs, ns):
            key = (int(aid), int(bid))
            entry = counts.setdefault(key, [0, 0])
            entry[0 if (dy, dx) == (0, 0) else 1] += int(n)
    area_a = dict(zip(*np.unique(a[a > 0], return_counts=True)))
    area_b = dict(zip(*np.unique(b[b > 0], return_counts=True)))
    rows = []
    for (aid, bid), (overlap, adjacent) in sorted(counts.items()):
        assert pa[aid] == pb[bid]
        rows.append(
            [
                aid,
                bid,
                pa[aid],
                overlap,
                adjacent,
                overlap / area_a[aid],
                overlap / area_b[bid],
            ]
        )
    return pd.DataFrame(
        rows,
        columns=[
            "a_id",
            "b_id",
            "parent_nucleus_id",
            "overlap_px",
            "adjacent_pixel_pairs",
            "fraction_a",
            "fraction_b",
        ],
    )


def derive_mn_bodies(
    bp1, rnp, nuclei, connectivity=2, reporter=None, reporter_min_overlap=None
):
    """Select whole 53BP1 bodies touching RNP and subtract base objects from nuclei.

    Reporter is optional and never part of the MN definition.
    Without reporter, the baseline is explicitly named nuclei_minus_bp1_rnp.
    """
    pairs = contact_table(bp1, rnp, nuclei, connectivity)
    mn = np.where(np.isin(bp1, pairs.a_id.unique()), bp1, 0)
    masks = {"bp1": bp1, "rnp": rnp, "mn": mn}
    remove = (bp1 > 0) | (rnp > 0)
    reporter_pairs = pd.DataFrame()
    baseline_name = "nuclei_minus_bp1_rnp"
    if reporter is not None:
        object_parents(reporter, nuclei)
        if reporter_min_overlap is None or not 0 < reporter_min_overlap <= 1:
            raise ValueError("Explicit reporter_min_overlap in (0,1] is required")
        reporter_pairs = contact_table(reporter, mn, nuclei, connectivity=0)
        reporter_pairs["qualifies"] = reporter_pairs.fraction_a >= reporter_min_overlap
        selected = reporter_pairs.loc[reporter_pairs.qualifies, "a_id"].unique()
        masks["reporter"] = reporter
        masks["reporter_in_mn"] = np.where(np.isin(reporter, selected), reporter, 0)
        remove |= reporter > 0
        baseline_name = "nuclei_depleted"
    masks[baseline_name] = np.where(remove, 0, nuclei)
    return masks, pairs, reporter_pairs


def measure_body_features(image, channel_names, nuclei, masks, metadata=None):
    """Measure corrected intensities and produce unique per-nucleus summaries.

    Empty baseline compartments keep a row with area zero and NaN intensities.
    Object intensity ratios use the matching parent's baseline, without epsilons.
    """
    nuclei = validate_labels(nuclei)
    image = np.asarray(image)
    if (
        image.ndim != 3
        or image.shape[1:] != nuclei.shape
        or image.shape[0] != len(channel_names)
    ):
        raise ValueError("Expected CYX image and one name per channel")
    if len(set(channel_names)) != len(channel_names) or not np.isfinite(image).all():
        raise ValueError("Channel names must be unique; intensities must be finite")
    metadata = metadata or {}
    if set(metadata) - {"plate", "well", "tile"}:
        raise ValueError("Metadata keys must be plate, well and tile")
    parent_ids = np.unique(nuclei[nuclei > 0])
    rows = []
    baseline_names = [k for k in masks if k.startswith("nuclei_")]
    if len(baseline_names) != 1:
        raise ValueError("Exactly one depleted/baseline mask is required")
    baseline_name = baseline_names[0]
    for name, labels in masks.items():
        parents = object_parents(labels, nuclei)
        obj_ids = parent_ids if name == baseline_name else sorted(parents)
        # Measure within bounding boxes instead of scanning the entire tile once
        # per body. Dense temporary IDs also handle sparse original labels safely.
        ids, inverse = np.unique(labels, return_inverse=True)
        dense = inverse.reshape(labels.shape) + 1
        dense[labels == 0] = 0
        slices = {int(ids[r.label - 1]): r.slice for r in measure.regionprops(dense)}
        for oid in obj_ids:
            sl = slices.get(oid, (slice(0, 0), slice(0, 0)))
            selected = labels[sl] == oid
            parent = int(oid) if name == baseline_name else parents[oid]
            y, x = np.nonzero(selected)
            y, x = y + sl[0].start, x + sl[1].start
            row = {
                **metadata,
                "object_type": name,
                "object_id": int(oid),
                "parent_nucleus_id": parent,
                "area": len(y),
                "i": float(y.mean()) if len(y) else np.nan,
                "j": float(x.mean()) if len(x) else np.nan,
            }
            for ch, channel_name in enumerate(channel_names):
                vals = image[ch][sl][selected]
                for stat in ("mean", "median", "sum"):
                    row[f"{channel_name}_{stat}"] = (
                        float(getattr(np, stat)(vals)) if len(vals) else np.nan
                    )
            rows.append(row)
    columns = list(metadata) + [
        "object_type",
        "object_id",
        "parent_nucleus_id",
        "area",
        "i",
        "j",
    ]
    columns += [
        f"{ch}_{stat}" for ch in channel_names for stat in ("mean", "median", "sum")
    ]
    objects = pd.DataFrame(rows, columns=columns)
    baseline = objects[objects.object_type == baseline_name].set_index(
        "parent_nucleus_id"
    )
    for channel_name in channel_names:
        denominator = objects.parent_nucleus_id.map(baseline[f"{channel_name}_mean"])
        objects[f"{channel_name}_baseline_valid"] = denominator.notna() & (
            denominator > 0
        )
        objects[f"{channel_name}_enrichment"] = objects[
            f"{channel_name}_mean"
        ] / denominator.where(denominator > 0)
    summary = pd.DataFrame({"label": parent_ids})
    for name in masks:
        sub = objects[objects.object_type == name]
        if name != baseline_name:
            summary[f"nucleus_n_{name}"] = (
                summary.label.map(sub.groupby("parent_nucleus_id").size())
                .fillna(0)
                .astype(int)
            )
        summary[f"nucleus_{name}_area"] = summary.label.map(
            sub.groupby("parent_nucleus_id").area.sum()
        ).fillna(0)
    for ch in channel_names:
        summary[f"nucleus_{baseline_name}_{ch}_mean"] = summary.label.map(
            baseline[f"{ch}_mean"]
        )
    for key, value in metadata.items():
        summary.insert(0, key, value)
    return objects, summary
