"""Intensity evenness and sums on irregular object masks."""

import numpy as np
import pytest

from lib.phenotype.chromatin_quantification import label_intensity_statistics


def test_uniform_concentrated_and_missing_signal():
    labels = np.array([[1, 1, 2, 2], [3, 3, 4, 0]])
    image = np.array([[5.0, 5.0, 10.0, 0.0], [0.0, 0.0, 7.0, np.nan]])
    s = label_intensity_statistics(labels, image)
    np.testing.assert_allclose(s["homogeneity"][1:3], [1, 0])
    np.testing.assert_allclose(s["effective_fraction"][1:3], [1, 0.5])
    np.testing.assert_allclose(s["cv"][1:3], [0, 1])
    assert np.isnan(s["homogeneity"][[0, 3, 4]]).all()
    np.testing.assert_allclose(s["sum"][1:], s["mean"][1:] * s["area"][1:])
    assert np.isnan(s["cv"][3])


def test_evenness_scale_invariance_and_background_bias():
    labels = np.array([[1, 1, 1], [1, 1, 0]])
    image = np.array([[0.0, 2.0, 4.0], [9.0, 20.0, 9999.0]])
    a = label_intensity_statistics(labels, image)
    b = label_intensity_statistics(labels, image * 1000)
    c = label_intensity_statistics(labels, image + 100)
    for key in ["cv", "homogeneity", "effective_fraction"]:
        np.testing.assert_allclose(a[key], b[key])
    assert c["homogeneity"][1] > a["homogeneity"][1]
    assert c["cv"][1] < a["cv"][1]
    np.testing.assert_allclose(a["sd"][1], np.std([0, 2, 4, 9, 20]))


def test_negative_roi_intensity_rejected():
    with pytest.raises(ValueError, match="nonnegative"):
        label_intensity_statistics(np.ones((2, 2), int), -np.ones((2, 2)))
