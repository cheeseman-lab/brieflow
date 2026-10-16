"""Scientific invariants for the experimental nuclear-body adapter."""

import numpy as np
import pandas as pd
import pytest

from lib.phenotype.nuclear_bodies import (
    NuclearObjectSpec,
    contact_table,
    derive_mn_bodies,
    measure_body_features,
    object_parents,
    segment_nuclear_objects,
)
from lib.phenotype.nuclear_body_io import (
    identify_nuclear_objects,
    derive_nuclear_objects,
)


def scene():
    nuclei = np.full((12, 16), 7, np.int32)
    a = np.zeros_like(nuclei)
    b = np.zeros_like(nuclei)
    a[3:6, 2:5] = 11
    a[3:6, 8:11] = 93
    b[4, 5:8] = 4
    return nuclei, a, b


def test_bridge_keeps_two_mn_bodies_and_original_pixels():
    nuclei, a, b = scene()
    masks, pairs, _ = derive_mn_bodies(a, b, nuclei, connectivity=1)
    assert set(pairs.a_id) == {11, 93}
    assert set(pairs.b_id) == {4}
    np.testing.assert_array_equal(masks["mn"], a)
    assert (pairs.overlap_px == 0).all()
    assert (pairs.adjacent_pixel_pairs > 0).all()
    assert not np.any(masks["nuclei_minus_bp1_rnp"][(a > 0) | (b > 0)])


@pytest.mark.parametrize("mode,expected", [(0, False), (1, False), (2, True)])
def test_diagonal_contact_is_configurable(mode, expected):
    nuclei = np.ones((4, 4), np.int32)
    a, b = np.zeros_like(nuclei), np.zeros_like(nuclei)
    a[1, 1], b[2, 2] = 5, 9
    assert (not contact_table(a, b, nuclei, mode).empty) == expected


def test_overlap_fractions_use_each_objects_area():
    nuclei, a, b = scene()
    b[:] = 0
    b[4, 3] = 1
    pairs = contact_table(a, b, nuclei, 0)
    assert pairs.iloc[0].fraction_a == pytest.approx(1 / 9)
    assert pairs.iloc[0].fraction_b == 1


def test_neighbors_in_different_nuclei_never_contact():
    nuclei, a, b = scene()
    nuclei[:, 5:] = 8
    pairs = contact_table(a, b, nuclei, 2)
    assert set(pairs.a_id) == {93}


def test_one_pixel_gap_is_not_touching():
    nuclei = np.ones((5, 5), np.int32)
    a, b = np.zeros_like(nuclei), np.zeros_like(nuclei)
    a[2, 1], b[2, 3] = 1, 2
    assert contact_table(a, b, nuclei, 2).empty


def test_invalid_parentage_fails_instead_of_centroid_assignment():
    nuclei, a, _ = scene()
    nuclei[3, 2] = 0
    with pytest.raises(ValueError, match="outside"):
        object_parents(a, nuclei)


def test_ring_parent_does_not_depend_on_centroid():
    n = np.zeros((9, 9), np.int32)
    n[1:8, 1:8] = 42
    n[3:6, 3:6] = 0
    a = np.where(n > 0, 6, 0)
    assert object_parents(a, n) == {6: 42}


def test_reporter_is_optional_and_not_required_for_mn():
    n, a, b = scene()
    reporter = np.zeros_like(n)
    reporter[4, 3] = 17
    masks, _, rp = derive_mn_bodies(
        a, b, n, reporter=reporter, reporter_min_overlap=0.5
    )
    assert set(np.unique(masks["mn"])) == {0, 11, 93}
    np.testing.assert_array_equal(masks["reporter_in_mn"], reporter)
    assert rp.qualifies.all()
    assert "nuclei_depleted" in masks


def test_empty_baseline_and_zero_object_nucleus_remain_in_summary():
    n = np.array([[3, 3, 0, 19], [3, 3, 0, 19]], np.int32)
    a = np.where(n == 3, 8, 0)
    b = np.zeros_like(n)
    masks, _, _ = derive_mn_bodies(a, b, n)
    objects, summary = measure_body_features(
        np.ones((1, *n.shape)), ["marker"], n, masks
    )
    assert len(summary) == 2 and set(summary.label) == {3, 19}
    baseline = objects.query(
        "object_type == 'nuclei_minus_bp1_rnp' and parent_nucleus_id == 3"
    ).iloc[0]
    assert baseline.area == 0 and np.isnan(baseline.marker_mean)
    body = objects.query("object_type == 'bp1'").iloc[0]
    assert np.isnan(body.marker_enrichment)
    assert summary.loc[summary.label == 19, "nucleus_n_bp1"].item() == 0


def test_no_nuclei_produces_schema_correct_empty_results():
    n = np.zeros((8, 8), np.int32)
    labels, parents, _ = segment_nuclear_objects(
        np.zeros_like(n), n, NuclearObjectSpec()
    )
    assert not labels.any() and parents.empty
    masks, _, _ = derive_mn_bodies(labels, labels, n)
    objects, summary = measure_body_features(np.zeros((1, 8, 8)), ["x"], n, masks)
    assert objects.empty and summary.empty
    assert "x_mean" in objects and "nucleus_n_mn" in summary


def test_cropped_measurements_preserve_sparse_labels_and_global_coordinates():
    n = np.zeros((30, 40), np.uint32)
    n[10:20, 20:35] = 4_000_000_000
    a = np.zeros_like(n)
    a[12:15, 23:28] = 3_000_000_000
    b = np.zeros_like(n)
    image = np.arange(n.size).reshape(n.shape)[None]
    masks, _, _ = derive_mn_bodies(a, b, n)
    table, _ = measure_body_features(image, ["marker"], n, masks)
    row = table.query("object_type == 'bp1'").iloc[0]
    assert row.object_id == 3_000_000_000
    assert row.parent_nucleus_id == 4_000_000_000
    assert row.i == 13 and row.j == 25
    assert row.marker_mean == image[0][a > 0].mean()
    assert row.marker_median == np.median(image[0][a > 0])
    assert row.marker_sum == image[0][a > 0].sum()


def test_size_diagnostics_identify_rejected_candidates():
    n = np.ones((50, 50), np.int32)
    yy, xx = np.mgrid[:50, :50]
    image = np.exp(-((yy - 25) ** 2 + (xx - 25) ** 2) / 12)
    small, _, debug = segment_nuclear_objects(
        image, n, NuclearObjectSpec(diameter_min=0.1, diameter_max=1)
    )
    assert not small.any()
    assert len(debug["candidates"]) > 0
    assert (debug["candidates"].status == "above_max").all()


def test_detection_is_invariant_to_extranuclear_staining():
    n = np.zeros((64, 64), np.int32)
    n[8:56, 8:56] = 10000
    y, x = np.mgrid[:64, :64]
    image = 5 + 80 * np.exp(-((y - 30) ** 2 + (x - 30) ** 2) / 8)
    spec = NuclearObjectSpec(deviations=2)
    a, _, _ = segment_nuclear_objects(image, n, spec)
    changed = image.copy()
    changed[n == 0] = 1e8
    b, _, _ = segment_nuclear_objects(changed, n, spec)
    assert a.max() > 0
    np.testing.assert_array_equal(a, b)
    assert np.all(a[n == 0] == 0)
    assert set(object_parents(a, n).values()) == {10000}


def test_invalid_images_and_parameters_fail():
    with pytest.raises(ValueError):
        NuclearObjectSpec(diameter_min=10, diameter_max=2)
    with pytest.raises(ValueError):
        segment_nuclear_objects(np.full((5, 5), np.nan), np.ones((5, 5), int), {})


def test_limit_exceeded_is_failure_not_empty_result():
    n = np.zeros((64, 96), np.int32)
    n[5:59, 5:91] = 1
    y, x = np.mgrid[:64, :96]
    image = 5 + 80 * np.exp(-((y - 30) ** 2 + (x - 25) ** 2) / 8)
    image += 80 * np.exp(-((y - 30) ** 2 + (x - 70) ** 2) / 8)
    with pytest.raises(RuntimeError, match="limit exceeded"):
        segment_nuclear_objects(
            image, n, NuclearObjectSpec(deviations=2), max_objects=1
        )


def test_three_dimensional_labels_are_rejected():
    with pytest.raises(ValueError, match="2D"):
        segment_nuclear_objects(np.zeros((2, 8, 8)), np.ones((2, 8, 8), int), {})


def test_file_adapters_roundtrip(tmp_path):
    from tifffile import imwrite

    n = np.zeros((64, 64), np.int32)
    n[5:59, 5:59] = 7
    y, x = np.mgrid[:64, :64]
    c = 10 + 100 * np.exp(-((y - 30) ** 2 + (x - 30) ** 2) / 8)
    image = np.stack([c, c]).astype(np.float32)
    imwrite(
        tmp_path / "image.tiff",
        image,
        metadata={"axes": "CYX"},
        photometric="minisblack",
    )
    imwrite(tmp_path / "nuclei.tiff", n)
    cfg = {
        "channel_names": ["bp", "r"],
        "objects": {
            k: {"channel": ch, "params": {"deviations": 2}}
            for k, ch in [("bp1", "bp"), ("rnp", "r")]
        },
    }
    identify_nuclear_objects(
        tmp_path / "image.tiff",
        tmp_path / "nuclei.tiff",
        cfg,
        tmp_path / "base.npz",
        tmp_path / "parents.tsv",
    )
    outputs = [
        tmp_path / name
        for name in [
            "masks.npz",
            "pairs.tsv",
            "rep.tsv",
            "objects.tsv",
            "summary.tsv",
            "manifest.json",
        ]
    ]
    derive_nuclear_objects(
        tmp_path / "image.tiff",
        tmp_path / "nuclei.tiff",
        tmp_path / "base.npz",
        tmp_path / "parents.tsv",
        cfg,
        outputs,
        {"plate": "1", "well": "A1", "tile": "1"},
    )
    summary = pd.read_csv(outputs[4], sep="\t")
    assert len(summary) == 1 and summary.nucleus_n_mn.item() >= 1
    assert all(path.exists() for path in outputs)
