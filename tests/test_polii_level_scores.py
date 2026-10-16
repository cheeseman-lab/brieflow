import numpy as np

from lib.phenotype.reporter_polii import polii_level_balance_scores


def test_common_scale_and_reference_values():
    a = polii_level_balance_scores([1, 0.2, 2], [1, 0.05, 0.5])
    b = polii_level_balance_scores([10, 2, 20], [10, 0.5, 5])
    for key in a:
        degree = (
            0
            if key in ["Ser5_share", "Ser5_Ser2_log2_ratio"]
            else 2
            if key == "S5_times_sum"
            else 1
        )
        np.testing.assert_allclose(b[key], a[key] * 10**degree)
    assert a["S5_times_sum"][0] == 2
    assert a["S5_times_sum_over_S2"][0] == 2
    assert a["S5_bounded_balance"][0] == 1
    np.testing.assert_allclose(a["S5_bounded_balance"][1:], [0.32, 3.2])


def test_bounded_weight_and_small_denominator_failure_mode():
    s = polii_level_balance_scores([0.2, 0.2, 2], [0.05, 0.001, 0.5])
    assert s["S5_times_sum_over_S2"][1] > 40
    assert s["S5_bounded_balance"][1] < 0.4
    assert np.all(s["S5_bounded_balance"] <= 2 * np.array([0.2, 0.2, 2]))
    assert np.all(s["S5_weighted_geometric"][:2] < 1)
    np.testing.assert_allclose(s["S5_minus_S2"], [0.15, 0.199, 1.5])


def test_invalid_inputs_are_explicitly_missing():
    scores = polii_level_balance_scores([0, -1, np.nan, 1], [1, 1, 1, 0])
    assert all(np.isnan(v).all() for v in scores.values())
