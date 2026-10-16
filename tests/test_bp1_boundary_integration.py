"""Boundary-stage contracts: raw masks, rescue rules, missing data and replay."""

import numpy as np
import pytest

from lib.phenotype.bp1_boundary_refinement import BoundarySpec, refine_boundaries
from lib.phenotype.mn_pipeline import boundary_settings, define_mn, refine_components


def scene():
    nuclei = np.ones((40, 40), dtype=np.int32)
    bp1 = np.zeros_like(nuclei)
    bp1[10:19, 10:19] = 7
    image = np.full(nuclei.shape, 10.0)
    image[bp1 > 0] = 13
    image[12:17, 12:17] = 1000
    rnp = np.zeros_like(nuclei)
    rnp[13:16, 13:16] = 1
    components = dict(
        nuclei_primary=nuclei,
        nuclei_definition=nuclei,
        nuclei_observed=nuclei,
        bp1=bp1,
        rnp_primary=rnp,
        rnp_fifth=np.zeros_like(nuclei),
        reporter=np.zeros_like(nuclei),
        nucleoli=np.zeros_like(nuclei),
    )
    images = dict(BP1=image, H3K9me2_artifact=np.zeros_like(nuclei, bool))
    settings = dict(
        bp1_boundary=dict(
            enabled=True, method="relative_contrast", threshold=0.30, smoothing_sigma=0
        ),
        consensus=dict(minimum_enrichment=5, minimum_area_px=70),
        reporter=dict(minimum_overlap_fraction=0.2),
    )
    return images, components, settings


def test_refined_area_is_used_for_rescue_and_original_masks_remain_available():
    images, old, config = scene()
    assert define_mn(images, old, config)[1].optimized.tolist() == [True]
    new, audit, info = refine_components(images, old, config)
    assert np.count_nonzero(new["bp1"]) == 25
    assert np.count_nonzero(old["bp1"]) == 81
    np.testing.assert_array_equal(new["bp1_untrimmed"], old["bp1"])
    for key in old:
        if key != "bp1":
            np.testing.assert_array_equal(new[key], old[key])
    decision = define_mn(images, new, config)[1].iloc[0]
    assert decision.in_A and not decision.in_C
    assert not decision.optimized and decision.decision_reason == "below_area"
    assert audit.object_id.tolist() == [7]
    assert info["enabled"] and info["background_unavailable_count"] == 0


def test_omitted_or_disabled_setting_is_an_identity_stage():
    images, masks, config = scene()
    for boundary in [None, {"enabled": False}]:
        settings = {k: v for k, v in config.items() if k != "bp1_boundary"}
        if boundary is not None:
            settings["bp1_boundary"] = boundary
        new, audit, info = refine_components(images, masks, settings)
        np.testing.assert_array_equal(new["bp1"], masks["bp1"])
        assert audit.empty and "refinement_status" in audit.columns
        assert not info["enabled"]


def test_refinement_cannot_accidentally_run_twice():
    images, masks, config = scene()
    new, _, _ = refine_components(images, masks, config)
    with pytest.raises(ValueError, match="already refined"):
        refine_components(images, new, config)


def test_missing_background_preserves_object_with_audit_and_empty_input_has_schema():
    images, masks, config = scene()
    masks["nucleoli"] = masks["nuclei_definition"].copy()
    new, audit, info = refine_components(images, masks, config)
    np.testing.assert_array_equal(new["bp1"], masks["bp1"])
    assert audit.refinement_status.tolist() == ["background_unavailable"]
    assert info["background_unavailable_count"] == 1
    masks["bp1"][:] = 0
    _, audit, info = refine_components(images, masks, config)
    assert audit.empty and "object_id" in audit.columns
    assert info["n_refined_objects"] == 0


def test_missing_reporter_domain_does_not_erase_definition_domain():
    images, masks, config = scene()
    masks["nuclei_observed"] = np.zeros_like(masks["nuclei_observed"])
    new, _, _ = refine_components(images, masks, config)
    assert np.any(new["bp1"])
    np.testing.assert_array_equal(new["nuclei_definition"], masks["nuclei_definition"])
    assert not np.any(new["nuclei_observed"])


@pytest.mark.parametrize(
    "config",
    [
        {"enabled": "true"},
        {"enabled": True},
        {"enabled": True, "threshold": 1},
        {"enabled": True, "threshold": 0.3, "minimum_background_pixels": 0},
    ],
)
def test_invalid_boundary_config_fails_early(config):
    with pytest.raises(ValueError):
        boundary_settings({"bp1_boundary": config})


def test_threshold_increase_only_removes_pixels():
    images, masks, _ = scene()
    baseline = np.where(masks["bp1"] > 0, 0, masks["nuclei_definition"])
    previous = masks["bp1"]
    for threshold in [0.12, 0.2, 0.3, 0.4]:
        current, _ = refine_boundaries(
            images["BP1"],
            masks["bp1"],
            masks["nuclei_definition"],
            baseline,
            BoundarySpec(threshold=threshold),
        )
        assert np.all((current == 0) | (current == previous))
        previous = current
