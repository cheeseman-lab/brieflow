"""Reporter relationships for existing MN objects; never reselect MN identity."""

import numpy as np
import pandas as pd
from skimage.measure import regionprops

from lib.phenotype.nuclear_bodies import contact_table, object_parents


def filter_reporter_candidates(
    image, candidates, nuclei, minimum_mean_enrichment=2, minimum_peak_enrichment=3
):
    """Keep bright puncta relative to their own nuclear median FLAG intensity.

    Candidate size/shape segmentation is upstream. The median is measured from
    the entire parent nucleus, independent of MN status and chromatin readouts.
    No fixed number of spots is required or imposed on a nucleus.
    """
    image = np.asarray(image)
    if image.shape != nuclei.shape or not np.isfinite(image).all():
        raise ValueError("Reporter intensities must be finite and match nuclei")
    if not all(
        np.isfinite(x) and x > 0
        for x in [minimum_mean_enrichment, minimum_peak_enrichment]
    ):
        raise ValueError("Reporter brightness thresholds must be finite and positive")
    parents = object_parents(candidates, nuclei)
    medians = {
        r.label: float(np.median(image[r.slice][nuclei[r.slice] == r.label]))
        for r in regionprops(nuclei)
    }
    columns = [
        "reporter_id",
        "parent_nucleus_id",
        "area",
        "mean",
        "max",
        "nuclear_median",
        "enrichment",
        "peak_enrichment",
        "cy",
        "cx",
        "keep",
    ]
    rows = []
    for r in regionprops(candidates, intensity_image=image):
        nid = parents[r.label]
        median = medians[nid]
        mean, peak = float(r.mean_intensity), float(r.max_intensity)
        mean_ratio = mean / median if median > 0 else np.nan
        peak_ratio = peak / median if median > 0 else np.nan
        keep = bool(
            mean_ratio >= minimum_mean_enrichment
            and peak_ratio >= minimum_peak_enrichment
        )
        rows.append(
            [
                r.label,
                nid,
                r.area,
                mean,
                peak,
                median,
                mean_ratio,
                peak_ratio,
                *r.centroid,
                keep,
            ]
        )
    table = pd.DataFrame(rows, columns=columns)
    return np.where(
        np.isin(candidates, table.loc[table.keep.astype(bool), "reporter_id"]),
        candidates,
        0,
    ), table


def classify_mn_reporter(
    mn,
    reporter,
    nuclei,
    minimum_overlap_pixels=1,
    minimum_reporter_fraction=0.0,
    *,
    strict_fraction=False,
):
    """Partition whole MN bodies by same-parent reporter intersection.

    The default requires positive pixel overlap, not adjacency or dilation.
    Optional reporter-area fraction is evaluated per reporter/MN pair. Set
    strict_fraction=True to require > rather than the legacy >= cutoff. Multiple
    reporter partners are retained; each MN is counted once. This function
    expects a fully observed domain; missing imaging support is handled by the
    caller and must not be encoded as reporter-negative observations.
    """
    if not isinstance(minimum_overlap_pixels, int) or minimum_overlap_pixels < 1:
        raise ValueError("minimum_overlap_pixels must be a positive integer")
    if (
        not np.isfinite(minimum_reporter_fraction)
        or not 0 <= minimum_reporter_fraction <= 1
    ):
        raise ValueError("minimum_reporter_fraction must lie in [0,1]")
    parents = object_parents(mn, nuclei)
    pairs = contact_table(mn, reporter, nuclei, connectivity=0)
    fraction_ok = (
        pairs.fraction_b > minimum_reporter_fraction
        if strict_fraction
        else pairs.fraction_b >= minimum_reporter_fraction
    )
    pairs["qualifies"] = (pairs.overlap_px >= minimum_overlap_pixels) & fraction_ok
    selected = pairs[pairs.qualifies]
    ids = sorted(parents)
    records = pd.DataFrame(
        {"mn_id": ids, "parent_nucleus_id": [parents[i] for i in ids]}
    )
    records["with_reporter"] = records.mn_id.isin(selected.a_id)
    records["n_reporter_partners"] = (
        records.mn_id.map(selected.groupby("a_id").b_id.nunique()).fillna(0).astype(int)
    )
    records["overlap_px"] = (
        records.mn_id.map(pairs.groupby("a_id").overlap_px.sum()).fillna(0).astype(int)
    )
    # Non-overlapping reporter labels make this sum a union, without duplicates.
    areas = dict(zip(*np.unique(mn[mn > 0], return_counts=True)))
    records["MN_coverage_fraction"] = records.overlap_px / records.mn_id.map(areas)
    positive = np.where(np.isin(mn, selected.a_id.unique()), mn, 0)
    negative = np.where((mn > 0) & (positive == 0), mn, 0)
    return (
        {"mn_with_reporter": positive, "mn_without_reporter": negative},
        pairs,
        records,
    )
