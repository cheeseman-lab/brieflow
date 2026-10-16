"""Operational reporter damage groups and complementary Pol II balance scores."""

import numpy as np
import pandas as pd

from lib.phenotype.nuclear_bodies import contact_table, object_parents


def polii_balance(ser5, ser2):
    """Relative enrichment shares, not calibrated molecular fractions.

    Inputs must be normalized on the same object type and parent reference.
    The two outputs sum to one and contain exactly one independent quantity.
    """
    a, b = np.broadcast_arrays(np.asarray(ser5, float), np.asarray(ser2, float))
    total = a + b
    valid = np.isfinite(a) & np.isfinite(b) & (a >= 0) & (b >= 0) & (total > 0)
    initiation = np.divide(a, total, out=np.full(a.shape, np.nan), where=valid)
    elongation = np.divide(b, total, out=np.full(a.shape, np.nan), where=valid)
    return initiation, elongation


def polii_level_balance_scores(ser5, ser2):
    """Compare exploratory scores of positive, normalized Ser5/Ser2 intensity.

    No pseudocounts or clipping. Invalid or nonpositive inputs return NaN.
    The sum is a sum of relative enrichments, not a calibrated Pol II amount.
    Only S5_times_sum scales quadratically under common rescaling; the proposed
    abundance-weighted alternatives scale linearly. Share/log-ratio are scale
    invariant. Bounded_balance refers to its weight in [0,2], not a bounded
    overall score. Ratio-weighted scores remain sensitive to small Ser2.
    """
    a, b = np.broadcast_arrays(np.asarray(ser5, float), np.asarray(ser2, float))
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    x, y = a[valid], b[valid]
    total = x + y
    values = {
        "Ser5_share": x / total,
        "PolII_mean_enrichment": total / 2,
        "Ser5_Ser2_log2_ratio": np.log2(x / y),
        "S5_times_sum": x * total,
        "S5_times_sum_over_S2": x * total / y,
        "S5_bounded_balance": 2 * x * x / total,
        "S5_weighted_geometric": x**0.75 * y**0.25,
        "S5_minus_S2": x - y,
    }
    result = {}
    for key, value in values.items():
        out = np.full(a.shape, np.nan)
        out[valid] = value
        result[key] = out
    return result


def classify_reporter_damage(reporters, bp1, mn, nuclei, minimum_fraction=0.2):
    """Separate no-contact reporters from reporters overlapping an accepted MN.

    Contact includes shared pixels and edge/corner adjacency, never a gap.
    MN association requires strictly greater than minimum_fraction of a full
    individual reporter inside one individual MN. Intermediate cases remain
    separate, rather than becoming false negatives in the no-contact group.
    """
    if not np.isfinite(minimum_fraction) or not 0 <= minimum_fraction <= 1:
        raise ValueError("minimum_fraction must lie in [0,1]")
    parents = object_parents(reporters, nuclei)
    bp_pairs = contact_table(reporters, bp1, nuclei, connectivity=2)
    mn_pairs = contact_table(reporters, mn, nuclei, connectivity=2)
    ids = sorted(parents)
    data = pd.DataFrame(
        dict(reporter_id=ids, parent_nucleus_id=[parents[i] for i in ids])
    )
    data["has_BP1_contact"] = data.reporter_id.isin(bp_pairs.a_id)
    data["has_MN_contact"] = data.reporter_id.isin(mn_pairs.a_id)
    data["max_MN_reporter_fraction"] = data.reporter_id.map(
        mn_pairs.groupby("a_id").fraction_a.max()
    ).fillna(0)
    positive = mn_pairs[
        (mn_pairs.overlap_px > 0) & (mn_pairs.fraction_a > minimum_fraction)
    ]
    data["MN_associated"] = data.reporter_id.isin(positive.a_id)
    data["no_detected_BP1_contact"] = ~data.has_BP1_contact & ~data.has_MN_contact
    data["damage_group"] = np.select(
        [data.MN_associated, data.no_detected_BP1_contact],
        ["Reporter with MN", "No detected 53BP1 contact"],
        default="Other 53BP1/MN contact",
    )
    data["associated_MN_ids"] = data.reporter_id.map(
        positive.groupby("a_id").b_id.apply(lambda s: ",".join(map(str, sorted(s))))
    ).fillna("")
    return data, dict(BP1=bp_pairs, MN=mn_pairs)
