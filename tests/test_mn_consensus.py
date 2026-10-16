"""Check scientific selection invariants and preservation of object identity."""

import numpy as np
import pytest

from lib.phenotype.mn_consensus import MNConsensusSpec, derive_consensus_mn


def fixture():
    nuclei = np.ones((20, 80), np.int32)
    nuclei[:, 40:] = 2
    bp = np.zeros_like(nuclei)
    image = np.full(nuclei.shape, 10.0)
    # Shared weak, A bright+large, C bright+large, neither bright+large,
    # A bright+small, C weak+large, A exactly at both inclusive boundaries.
    for i, (side, intensity) in enumerate(
        [(3, 15), (4, 70), (4, 70), (4, 70), (2, 70), (4, 30), (4, 50)]
    ):
        x = i * 10 + 2
        bp[3 : 3 + side, x : x + side] = (i + 1) * 10
        image[3 : 3 + side, x : x + side] = intensity
    a = np.where(np.isin(bp, [10, 20, 50, 70]), bp, 0)
    c = np.where(np.isin(bp, [10, 30, 60]), bp, 0)
    baseline = np.where(bp > 0, 0, nuclei)
    return bp, a, c, nuclei, image, baseline


def test_consensus_and_rescue_keep_ids_and_require_both_size_and_intensity():
    bp, a, c, n, image, baseline = fixture()
    masks, table = derive_consensus_mn(
        bp, a, c, n, image, baseline, MNConsensusSpec(5, 16)
    )
    np.testing.assert_array_equal(masks["consensus"], np.where(bp == 10, bp, 0))
    np.testing.assert_array_equal(
        masks["optimized"], np.where(np.isin(bp, [10, 20, 30, 70]), bp, 0)
    )
    assert not np.any((masks["consensus"] > 0) & (masks["rescued"] > 0))
    assert table.set_index("object_id").loc[40, "decision_reason"] == "neither_round"
    assert table.set_index("object_id").loc[50, "decision_reason"] == "below_area"
    assert table.set_index("object_id").loc[60, "decision_reason"] == "below_brightness"
    assert table.set_index("object_id").loc[70, "rescued"]


def test_missing_denominator_keeps_consensus_but_cannot_rescue():
    bp, a, c, n, image, baseline = fixture()
    baseline[:] = 0
    masks, _ = derive_consensus_mn(bp, a, c, n, image, baseline, MNConsensusSpec(5, 16))
    assert not masks["rescued"].any()
    np.testing.assert_array_equal(masks["optimized"], np.where(bp == 10, bp, 0))


def test_partial_or_changed_round_objects_are_rejected():
    bp, a, c, n, image, baseline = fixture()
    a[3, 2] = 0
    with pytest.raises(ValueError, match="whole original"):
        derive_consensus_mn(bp, a, c, n, image, baseline)


def test_baseline_cannot_include_a_body_or_wrong_parent_id():
    bp, a, c, n, image, baseline = fixture()
    baseline[3, 2] = 1
    with pytest.raises(ValueError, match="exclude all"):
        derive_consensus_mn(bp, a, c, n, image, baseline)
    baseline[3, 2] = 0
    baseline[0, 0] = 2
    with pytest.raises(ValueError, match="nucleus IDs"):
        derive_consensus_mn(bp, a, c, n, image, baseline)


@pytest.mark.parametrize(
    "spec",
    [
        dict(minimum_area_px=0),
        dict(minimum_enrichment=np.nan),
        dict(minimum_area_px=np.inf),
    ],
)
def test_invalid_thresholds(spec):
    with pytest.raises(ValueError):
        MNConsensusSpec(**spec)


@pytest.mark.parametrize("has_nuclei", [True, False])
def test_empty_field_preserves_empty_schema(has_nuclei):
    zeros = np.zeros((10, 10), np.int32)
    nuclei = np.ones_like(zeros) if has_nuclei else zeros.copy()
    masks, decisions = derive_consensus_mn(
        zeros, zeros, zeros, nuclei, np.ones_like(zeros), nuclei
    )
    assert decisions.empty and "decision_reason" in decisions
    assert all(not mask.any() for mask in masks.values())
