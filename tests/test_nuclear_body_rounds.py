"""Cross-round scientific invariants for registration and nucleolar exclusion."""

import numpy as np
import pytest
from scipy import ndimage as ndi
from lib.phenotype.nuclear_body_rounds import (
    align_round,
    common_nuclei,
    segment_nucleoli,
    exclude_nucleolar_mn,
)


def test_registration_shift_sign_and_marker_transform():
    yy, xx = np.mgrid[:128, :128]
    reference = np.exp(-((yy - 45) ** 2 + (xx - 50) ** 2) / 100) + 0.6 * np.exp(
        -((yy - 80) ** 2 + (xx - 85) ** 2) / 150
    )
    marker = np.exp(-((yy - 45) ** 2 + (xx - 50) ** 2) / 6)
    shift = (7, -4)
    moving = ndi.shift(reference, shift, order=1)
    moving_marker = ndi.shift(marker, shift, order=1)
    aligned, _, valid, stats = align_round(reference, moving, moving_marker)
    np.testing.assert_allclose(stats["shift_yx_to_reference"], [-7, 4], atol=0.2)
    assert np.unravel_index(np.argmax(aligned), aligned.shape) == (45, 50)
    assert not valid[-1].any()
    assert stats["dapi_correlation_after"] > 0.99
    assert stats["dapi_correlation_after"] > stats["dapi_correlation_before"]


def test_registration_rejects_blank_dapi():
    with pytest.raises(ValueError):
        align_round(np.zeros((32, 32)), np.zeros((32, 32)), np.zeros((32, 32)))


def test_missing_round_support_excludes_entire_nucleus():
    n = np.zeros((12, 15), np.int32)
    n[2:5, 2:6] = 3
    n[7:10, 7:11] = 9
    valid = np.ones(n.shape, bool)
    valid[2, 2] = False
    result, ids = common_nuclei(n, valid)
    assert ids.tolist() == [3] and not np.any(result == 3)
    np.testing.assert_array_equal(result == 9, n == 9)


def test_nucleolus_contact_removes_whole_mn_without_trimming_or_relabeling():
    n = np.ones((20, 20), np.int32)
    mn = np.zeros_like(n)
    mn[3:7, 3:7] = 11
    mn[12:16, 12:16] = 35
    no = np.zeros_like(n)
    no[7:10, 7:10] = 4
    kept, rejected, pairs = exclude_nucleolar_mn(mn, no, n, connectivity=2)
    np.testing.assert_array_equal(rejected == 11, mn == 11)
    np.testing.assert_array_equal(kept == 35, mn == 35)
    assert not np.any(kept == 11) and pairs.a_id.tolist() == [11]
    overlap_only, _, _ = exclude_nucleolar_mn(mn, no, n, connectivity=0)
    np.testing.assert_array_equal(overlap_only, mn)


def test_nucleolar_contact_cannot_cross_nuclear_parents():
    n = np.ones((10, 10), np.int32)
    n[:, 5:] = 2
    mn = np.zeros_like(n)
    mn[3, 4] = 11
    no = np.zeros_like(n)
    no[3, 5] = 9
    kept, rejected, _ = exclude_nucleolar_mn(mn, no, n)
    np.testing.assert_array_equal(kept, mn)
    assert not rejected.any()


def test_dark_pocket_detection_ignores_outside_polii():
    n = np.zeros((96, 96), np.int32)
    n[10:86, 10:86] = 2
    yy, xx = np.mgrid[:96, :96]
    polii = np.full(n.shape, 100.0)
    polii[(yy - 48) ** 2 + (xx - 48) ** 2 < 9**2] = 10
    first, _, _ = segment_nucleoli(polii, n)
    assert first[48, 48] > 0 and not first[n == 0].any()
    changed = polii.copy()
    changed[n == 0] = 1e9
    second, _, _ = segment_nucleoli(changed, n)
    np.testing.assert_array_equal(first, second)
    uniform, _, _ = segment_nucleoli(np.full(n.shape, 10.0), n)
    assert not uniform.any()
