"""Biological invariants at the accepted pipeline stage boundaries."""

import numpy as np
import pandas as pd

from lib.phenotype.mn_pipeline import define_mn, measure_mn, parent_summary, MARKERS


def test_saturation_is_detected_before_illumination_correction():
    from lib.phenotype.mn_pipeline import register_rounds, ROLE_ROUND

    rounds = {}
    names = []
    for rnd in ["primary", "round2", "round3", "round4", "round5"]:
        roles = {
            role: role
            for role, value in ROLE_ROUND.items()
            if value == rnd and role != "DAPI"
        }
        rounds[rnd] = dict(dapi="DAPI_" + rnd, markers=roles)
        names.extend(["DAPI_" + rnd] + list(roles))
    image = np.full((len(names), 64, 64), 100.0, dtype=np.float32)
    dapi = np.random.default_rng(1).uniform(500, 2000, size=(64, 64))
    for rnd in rounds:
        image[names.index(rounds[rnd]["dapi"])] = dapi
    raw = image.copy()
    raw[names.index("BP1"), 20, 20] = 65535
    settings = dict(
        channel_names=names,
        rounds=rounds,
        saturation_value=65535,
        objects={key: {} for key in ["bp1", "rnp_primary", "rnp_fifth", "reporter"]},
        nucleoli={},
        H3K9me2_artifacts={},
        registration=dict(
            max_shift_px=128,
            minimum_dapi_correlation=0.95,
            maximum_regional_residual_px=2,
        ),
    )
    aligned, _ = register_rounds(image, settings, raw)
    assert aligned["BP1"][20, 20] == 100
    assert aligned["BP1_saturated"][20, 20]


def fixture():
    n = np.ones((20, 20), np.int32)
    bp = np.zeros_like(n)
    bp[3:6, 3:6] = 1
    bp[10:13, 10:13] = 2
    a = np.zeros_like(n)
    a[3:5, 3:5] = 1
    a[10:12, 10:12] = 2
    c = np.zeros_like(n)
    c[3:5, 3:5] = 1
    rep = np.zeros_like(n)
    rep[3:5, 3:5] = 1
    masks = dict(
        nuclei_primary=n,
        nuclei_definition=n,
        nuclei_observed=n,
        bp1=bp,
        rnp_primary=a,
        rnp_fifth=c,
        reporter=rep,
        nucleoli=np.zeros_like(n),
    )
    images = {m: np.full(n.shape, 10.0, float) for m in MARKERS}
    images["BP1"][bp > 0] = 100.0
    for m in MARKERS:
        images[m + "_saturated"] = np.zeros_like(n, bool)
    for rnd in ["primary", "round2", "round3", "round4", "round5"]:
        images[rnd + "_valid"] = np.ones_like(n, bool)
    images["H3K9me2_artifact"] = np.zeros_like(n, bool)
    config = dict(
        consensus=dict(minimum_enrichment=5, minimum_area_px=8),
        reporter=dict(minimum_overlap_fraction=0.2),
        measurement=dict(minimum_baseline_pixels=20, minimum_baseline_fraction=0.2),
    )
    return images, masks, config


def test_identity_frozen_before_reporter_baseline_and_nucleolus_not_a_veto():
    images, masks, config = fixture()
    masks["nucleoli"][3:6, 3:6] = 1
    result, decisions, _, members, _, _ = define_mn(images, masks, config)
    assert members.mn_id.tolist() == [1, 2]
    assert decisions.rescued.tolist() == [False, True]
    assert members.with_reporter.tolist() == [True, False]
    prior = result["mn"].copy()
    # Add a reporter outside BP1: it changes measurement baseline, not rescue.
    masks["reporter"][15:18, 15:18] = 2
    updated = define_mn(images, masks, config)[0]
    np.testing.assert_array_equal(prior, updated["mn"])
    np.testing.assert_array_equal(result["rescue_baseline"], updated["rescue_baseline"])
    assert np.count_nonzero(updated["measurement_baseline"]) < np.count_nonzero(
        result["measurement_baseline"]
    )


def test_artifact_and_missing_round_qc_are_marker_specific():
    images, components, config = fixture()
    images["H3K9me2_artifact"][4, 4] = True
    masks, _, _, members, _, _ = define_mn(images, components, config)
    images["round4_valid"][0, 0] = False
    data, _ = measure_mn(images, components, masks, members, config)
    assert data.H3K9me2_qc_clean.tolist() == [False, True]
    assert data.BP1_qc_clean.all() and data.PolII_Ser5Ph_qc_clean.all()
    assert not data.PolII_Ser2Ph_qc_clean.any()
    assert not data.PolII_eligible.any()
    assert np.allclose(data.PolII_level, 1)
    assert np.allclose(data.PolII_balance, 0)


def test_zero_objects_and_missing_parents_are_distinguished():
    images, components, config = fixture()
    components["bp1"][:] = 0
    components["nuclei_primary"][18:, :] = 2
    components["nuclei_observed"] = np.where(components["nuclei_primary"] == 1, 1, 0)
    components["nuclei_definition"] = components["nuclei_observed"]
    components["reporter"][:] = 0
    masks, _, _, members, _, _ = define_mn(images, components, config)
    data, _ = measure_mn(images, components, masks, members, config)
    assert data.empty
    summary = parent_summary(data, components).set_index("label")
    assert summary.loc[1, "mn_all_count"] == 0
    assert pd.isna(summary.loc[2, "mn_all_count"])
