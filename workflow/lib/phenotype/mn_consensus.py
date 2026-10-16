"""Select consensus MN bodies, with a configurable bright-and-large exception."""

from dataclasses import dataclass

import numpy as np

from lib.phenotype.nuclear_bodies import measure_body_features, validate_labels


@dataclass(frozen=True)
class MNConsensusSpec:
    """Pilot settings; area is in pixels at the reference-image resolution.

    Enrichment is mean unenhanced 53BP1 over a body divided by mean 53BP1 in
    the supplied depleted parent nucleus. Both lower bounds are inclusive.
    These thresholds require recalibration for a different pixel size/assay.
    """

    minimum_enrichment: float = 5.0
    minimum_area_px: float = 70.0

    def __post_init__(self):
        """Require finite, positive rescue thresholds."""
        for name in ("minimum_enrichment", "minimum_area_px"):
            value = getattr(self, name)
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")


def derive_consensus_mn(bp1, mn_a, mn_c, nuclei, bp1_image, baseline, spec=None):
    """Return consensus, rescued and optimized masks plus per-body decisions.

    Each A/C mask must contain whole, identically labeled objects from ``bp1``.
    Select every shared object; otherwise select a single-round object only
    when BOTH 53BP1 enrichment and area pass the configured bounds. A bright,
    large body with no RNP support is never rescued. Invalid/empty baselines
    cannot authorize rescue, but do not remove a shared detection.

    The caller supplies registered rounds and a common observation domain.
    Baseline labels retain nucleus IDs and must exclude all BP1 bodies. For
    this pilot the frozen baseline also excludes both RNP masks and nucleolar
    proxies. Keep the baseline independent of the final MN selection.
    Pol II/H2AK119ub intensities and nucleolar contacts are not gating inputs.
    Original object IDs and full pixel footprints are preserved.
    """
    spec = (
        MNConsensusSpec(**spec) if isinstance(spec, dict) else spec or MNConsensusSpec()
    )
    bp1, mn_a, mn_c, nuclei, baseline = (
        validate_labels(labels) for labels in (bp1, mn_a, mn_c, nuclei, baseline)
    )
    if any(labels.shape != bp1.shape for labels in (mn_a, mn_c, nuclei, baseline)):
        raise ValueError("All masks must share the reference image shape")
    if np.any((baseline > 0) & (baseline != nuclei)):
        raise ValueError("Baseline must retain the corresponding nucleus IDs")
    if np.any((baseline > 0) & (bp1 > 0)):
        raise ValueError("Baseline must exclude all 53BP1 bodies")
    round_ids = []
    for labels in (mn_a, mn_c):
        ids = np.unique(labels[labels > 0])
        if not np.array_equal(labels, np.where(np.isin(bp1, ids), bp1, 0)):
            raise ValueError(
                "Round masks must retain whole original 53BP1 objects and IDs"
            )
        round_ids.append(ids)
    features, _ = measure_body_features(
        np.asarray(bp1_image)[None],
        ["BP1"],
        nuclei,
        {"bp1": bp1, "nuclei_depleted": baseline},
    )
    decisions = features[features.object_type == "bp1"].copy()
    # Empty observation domains can otherwise inherit object dtype from pandas.
    decisions["BP1_enrichment"] = decisions.BP1_enrichment.astype(float)
    decisions["in_A"] = decisions.object_id.isin(round_ids[0])
    decisions["in_C"] = decisions.object_id.isin(round_ids[1])
    decisions["consensus"] = decisions.in_A & decisions.in_C
    decisions["passes_brightness"] = np.isfinite(decisions.BP1_enrichment) & (
        decisions.BP1_enrichment >= spec.minimum_enrichment
    )
    decisions["passes_area"] = decisions.area >= spec.minimum_area_px
    single = decisions.in_A ^ decisions.in_C
    decisions["rescued"] = single & decisions.passes_brightness & decisions.passes_area
    decisions["optimized"] = decisions.consensus | decisions.rescued
    decisions["decision_reason"] = np.select(
        [
            decisions.consensus,
            decisions.rescued & decisions.in_A,
            decisions.rescued & decisions.in_C,
            ~(decisions.in_A | decisions.in_C),
            ~np.isfinite(decisions.BP1_enrichment),
            ~decisions.passes_brightness & ~decisions.passes_area,
            ~decisions.passes_brightness,
        ],
        [
            "both_rounds",
            "bright_large_A_only",
            "bright_large_C_only",
            "neither_round",
            "invalid_baseline",
            "below_brightness_and_area",
            "below_brightness",
        ],
        default="below_area",
    )
    masks = {
        name: np.where(
            np.isin(bp1, decisions.loc[decisions[name], "object_id"]), bp1, 0
        )
        for name in ("consensus", "rescued", "optimized")
    }
    return masks, decisions
