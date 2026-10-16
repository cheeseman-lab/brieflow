"""Bright IF artifact masks and explicit histone/DNA normalization helpers."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi


@dataclass(frozen=True)
class BrightArtifactSpec:
    """Native-image intensity, contrast and footprint settings for bright debris."""

    seed_intensity: float = 30000
    seed_local_ratio: float = 4
    background_sigma: float = 10
    grow_intensity: float = 12000
    grow_local_ratio: float = 1.8
    saturation_value: float = 65535
    saturated_local_ratio: float = 2
    halo_pixels: int = 1


def bright_if_artifacts(image, spec=BrightArtifactSpec()):
    """Find extreme puncta in a native IF image, inside and outside nuclei.

    Grow high-intensity/high-contrast seeds within a lower intensity/contrast
    support. Saturated seeds use a lower local-ratio threshold because broad
    clipped cores underestimate their true peaks. A small explicit halo covers
    the footprint edge; the caller must transform this mask with the same
    registration as the intensity image. Nuclear/marker biology is not an input.
    """
    im = np.asarray(image, dtype=float)
    if im.ndim != 2 or not np.isfinite(im).all() or (im < 0).any():
        raise ValueError("A finite nonnegative 2D image is required")
    if not isinstance(spec.halo_pixels, int) or spec.halo_pixels < 0:
        raise ValueError("halo_pixels must be a nonnegative integer")
    vals = [
        spec.seed_intensity,
        spec.seed_local_ratio,
        spec.background_sigma,
        spec.grow_intensity,
        spec.grow_local_ratio,
        spec.saturation_value,
        spec.saturated_local_ratio,
    ]
    if not all(np.isfinite(v) and v > 0 for v in vals):
        raise ValueError("Artifact thresholds must be finite and positive")
    if (
        spec.grow_intensity > spec.seed_intensity
        or spec.grow_local_ratio > spec.seed_local_ratio
    ):
        raise ValueError("Growth thresholds cannot exceed seed thresholds")
    background = ndi.gaussian_filter(im, spec.background_sigma)
    ratio = np.divide(im, background, out=np.zeros_like(im), where=background > 0)
    seeds = ((im >= spec.seed_intensity) & (ratio >= spec.seed_local_ratio)) | (
        (im >= spec.saturation_value) & (ratio >= spec.saturated_local_ratio)
    )
    support = (im >= spec.grow_intensity) & (ratio >= spec.grow_local_ratio)
    grown = ndi.binary_propagation(
        seeds, structure=np.ones((3, 3)), mask=support | seeds
    )
    if spec.halo_pixels:
        grown = ndi.binary_dilation(
            grown, structure=np.ones((3, 3)), iterations=spec.halo_pixels
        )
    return grown, {"seed": seeds, "support": support, "local_ratio": ratio}


def safe_ratio(numerator, denominator):
    """Ratios only for finite nonnegative numerators and positive denominators."""
    a, b = np.broadcast_arrays(
        np.asarray(numerator, float), np.asarray(denominator, float)
    )
    return np.divide(
        a,
        b,
        out=np.full(a.shape, np.nan),
        where=np.isfinite(a) & (a >= 0) & np.isfinite(b) & (b > 0),
    )


def histone_normalizations(marker_body, marker_baseline, dapi_body, dapi_baseline):
    """All means must describe the same MN and matched baseline footprints.

    per_DNA is an IF/DAPI intensity ratio, not a calibrated modification density.
    DNA_adjusted compares that ratio between MN and depleted nuclear reference.
    """
    nuclear = safe_ratio(marker_body, marker_baseline)
    per_dna = safe_ratio(marker_body, dapi_body)
    adjusted = safe_ratio(per_dna, safe_ratio(marker_baseline, dapi_baseline))
    return {"nuclear": nuclear, "per_DNA": per_dna, "DNA_adjusted": adjusted}


def label_intensity_statistics(labels, image):
    """Measure intensity and evenness without intensity bins or background fitting.

    Homogeneity is Shannon entropy of pixel intensity proportions divided by
    log(pixel count): 1 for uniform positive signal, 0 for one nonzero pixel.
    Effective fraction is exp(entropy)/pixel count. Both ignore pixel ordering;
    neither is a spatial texture measure. Additive background raises evenness.
    Empty, single-pixel and zero-signal ROIs have undefined homogeneity (NaN).
    Arrays are indexed by label ID; index zero is deliberately undefined.
    """
    labels, image = np.asarray(labels), np.asarray(image)
    if labels.shape != image.shape or labels.ndim != 2:
        raise ValueError("Labels and image must be matching 2D arrays")
    if not np.issubdtype(labels.dtype, np.integer) or (labels < 0).any():
        raise ValueError("Labels must be nonnegative integers")
    mask = labels > 0
    ids, values = labels[mask], image[mask].astype(float)
    if not np.isfinite(values).all() or (values < 0).any():
        raise ValueError("ROI intensities must be finite and nonnegative")
    size = int(labels.max()) + 1
    area = np.bincount(ids, minlength=size)
    total = np.bincount(ids, weights=values, minlength=size)
    mean = safe_ratio(total, area)
    # Center first to avoid cancellation for nearly uniform, bright images.
    variance = safe_ratio(
        np.bincount(ids, weights=(values - mean[ids]) ** 2, minlength=size), area
    )
    sd = np.sqrt(variance)
    probabilities = np.divide(
        values, total[ids], out=np.zeros_like(values), where=total[ids] > 0
    )
    positive = probabilities > 0
    entropy = -np.bincount(
        ids[positive],
        weights=probabilities[positive] * np.log(probabilities[positive]),
        minlength=size,
    )
    usable = (area > 1) & (total > 0)
    homogeneity = np.full(size, np.nan)
    effective_fraction = np.full(size, np.nan)
    homogeneity[usable] = np.clip(entropy[usable] / np.log(area[usable]), 0, 1)
    effective_fraction[usable] = np.clip(np.exp(entropy[usable]) / area[usable], 0, 1)
    total = total.astype(float)
    total[area == 0] = np.nan
    return dict(
        area=area,
        sum=total,
        mean=mean,
        sd=sd,
        cv=safe_ratio(sd, mean),
        homogeneity=homogeneity,
        effective_fraction=effective_fraction,
    )
