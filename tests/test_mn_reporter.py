import numpy as np
import pytest

from lib.phenotype.mn_reporter import classify_mn_reporter
from lib.phenotype.mn_reporter import filter_reporter_candidates


def test_overlap_is_required_and_whole_mn_ids_are_preserved():
    n = np.ones((12, 16), np.int32)
    mn = np.zeros_like(n)
    mn[2:5, 2:5] = 11
    mn[7:10, 2:5] = 22
    mn[2:5, 10:13] = 33
    r = np.zeros_like(n)
    r[3, 3] = 1
    r[4, 4] = 2
    r[7:10, 5:7] = 3
    masks, pairs, rows = classify_mn_reporter(mn, r, n)
    np.testing.assert_array_equal(masks["mn_with_reporter"], np.where(mn == 11, mn, 0))
    np.testing.assert_array_equal(
        masks["mn_without_reporter"], np.where(mn != 11, mn, 0)
    )
    assert rows.set_index("mn_id").loc[11, "n_reporter_partners"] == 2
    assert len(pairs) == 2 and pairs.overlap_px.sum() == 2
    np.testing.assert_array_equal(
        masks["mn_with_reporter"] + masks["mn_without_reporter"], mn
    )


def test_fraction_is_reporter_area_and_threshold_is_inclusive():
    n = np.ones((8, 8), np.int32)
    mn = np.zeros_like(n)
    r = np.zeros_like(n)
    mn[1:3, 1:3] = 7
    r[2:4, 2:4] = 8
    masks, pairs, _ = classify_mn_reporter(mn, r, n, minimum_reporter_fraction=0.25)
    assert masks["mn_with_reporter"].max() == 7 and pairs.fraction_b.item() == 0.25
    masks, _, _ = classify_mn_reporter(mn, r, n, minimum_overlap_pixels=2)
    assert not masks["mn_with_reporter"].any()


def test_strict_cutoff_rejects_exact_boundary_and_does_not_sum_across_mn():
    n = np.ones((8, 8), np.int32)
    mn = np.zeros_like(n)
    r = np.zeros_like(n)
    r[2:4, 2:4] = 8
    mn[2, 2] = 11
    mn[3, 3] = 12
    # Each MN occupies 25% of the reporter. They cannot combine to pass 40%.
    for threshold in [0.25, 0.4]:
        masks, pairs, _ = classify_mn_reporter(
            mn, r, n, minimum_reporter_fraction=threshold, strict_fraction=True
        )
        assert not masks["mn_with_reporter"].any()
        assert (pairs.fraction_b == 0.25).all()
        assert (pairs.fraction_a == 1).all()
    masks, _, rows = classify_mn_reporter(mn, r, n, strict_fraction=True)
    assert rows.with_reporter.sum() == 2
    np.testing.assert_array_equal(masks["mn_with_reporter"], mn)


def test_invalid_parent_or_empty_reporters():
    n = np.ones((8, 8), np.int32)
    mn = np.zeros_like(n)
    r = np.zeros_like(n)
    mn[1:3, 1:3] = 7
    masks, _, rows = classify_mn_reporter(mn, r, n)
    assert not rows.with_reporter.any()
    np.testing.assert_array_equal(masks["mn_without_reporter"], mn)
    n[2, 2] = 2
    r[1:3, 1:3] = 8
    with pytest.raises(ValueError):
        classify_mn_reporter(mn, r, n)


def test_empty_field_and_invalid_cutoffs():
    zero = np.zeros((5, 5), np.int32)
    masks, pairs, rows = classify_mn_reporter(zero, zero, zero)
    assert rows.empty and pairs.empty and not masks["mn_with_reporter"].any()
    with pytest.raises(ValueError):
        classify_mn_reporter(zero, zero, zero, 0)
    with pytest.raises(ValueError):
        classify_mn_reporter(zero, zero, zero, minimum_reporter_fraction=2)


def test_reporter_filter_keeps_three_bright_spots_without_capping_counts():
    n = np.ones((20, 20), np.int32)
    c = np.zeros_like(n)
    im = np.full(n.shape, 10.0)
    for i, x in enumerate([2, 7, 12], 1):
        c[2:4, x : x + 2] = i
        im[2:4, x : x + 2] = 30
    c[10:12, 2:4] = 9
    im[10:12, 2:4] = 15
    labels, table = filter_reporter_candidates(im, c, n)
    assert np.unique(labels).tolist() == [0, 1, 2, 3]
    assert table.keep.sum() == 3
    blank, _ = filter_reporter_candidates(np.zeros_like(im), c, n)
    assert not blank.any()


def test_reporter_filter_uses_only_corresponding_nucleus():
    n = np.zeros((10, 10), np.int32)
    n[2:8, 2:8] = 5
    c = np.zeros_like(n)
    c[4:6, 4:6] = 7
    im = np.full(n.shape, 10.0)
    im[c > 0] = 30
    a, _ = filter_reporter_candidates(im, c, n)
    im[n == 0] = 1e9
    b, _ = filter_reporter_candidates(im, c, n)
    np.testing.assert_array_equal(a, b)
