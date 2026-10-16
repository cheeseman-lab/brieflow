import numpy as np
import pytest

from lib.phenotype.chromatin_quantification import (
    BrightArtifactSpec,
    bright_if_artifacts,
    histone_normalizations,
)


def test_bright_punctum_grows_halo_but_uniform_bright_stain_does_not():
    im = np.full((120, 120), 5000.0)
    im[50:70, 50:70] = 15000
    im[57:63, 57:63] = 60000
    mask, debug = bright_if_artifacts(im)
    assert mask[60, 60] and debug["seed"][60, 60]
    assert mask[56, 56] and not mask[0, 0]
    assert not bright_if_artifacts(np.full_like(im, 45000))[0].any()
    assert not bright_if_artifacts(np.full_like(im, 65535))[0].any()


def test_growth_requires_a_seed_and_parameters_are_validated():
    im = np.full((80, 80), 500.0)
    im[35:45, 35:45] = 18000
    assert not bright_if_artifacts(im)[0].any()
    with pytest.raises(ValueError):
        bright_if_artifacts(im, BrightArtifactSpec(halo_pixels=-1))


def test_normalization_uses_matched_means_and_excludes_zero_denominators():
    q = histone_normalizations([20, 20, 20], [10, 10, 0], [40, 0, 40], [10, 10, 10])
    np.testing.assert_allclose(q["nuclear"][:2], [2, 2])
    assert q["per_DNA"][0] == 0.5
    assert q["DNA_adjusted"][0] == 0.5  # 2x histone / 4x DAPI enrichment
    assert np.isnan(q["DNA_adjusted"][1:]).all()
    assert np.isnan(q["per_DNA"][1])


def test_same_ROI_mean_and_integrated_DNA_ratios_agree():
    q = histone_normalizations(24, 8, 12, 4)
    assert q["per_DNA"] == (24 * 17) / (12 * 17)
    assert q["DNA_adjusted"] == 1
