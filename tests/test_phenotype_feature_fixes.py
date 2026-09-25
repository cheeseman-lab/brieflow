"""Regression tests for phenotype segmentation and feature-extraction defects.

Each test pins one defect:

1. `measure_colocalization` with a float threshold raised `NameError` (`ifinstance`).
2. `identify_cytoplasm_cellpose` returned `None` for unreconciled masks, which the caller
   then crashed on; it now raises a `ValueError` that names the `reconcile` setting.
3. A list-valued `foci_channel` raised `TypeError` in the secondary-object features,
   while the primary phenotype path accepts an int or a list.
4. Nuclei centroids for secondary objects were keyed by row index, so
   `nearest_nucleus_id` was a row number instead of a nucleus label.
5. `segment_cellpose`'s default `cellpose_kwargs` held `None` thresholds that reached
   Cellpose, and `pop` mutated that shared default and the caller's dict.
6. cp_measure dropped every measurement after the first failing one in a group.
7. cp_measure column names carried a triple underscore.
8. The secondary-object compute module required plotting dependencies to import.
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))


def _two_cells():
    """Return nuclei and cell masks for two cells whose labels match."""
    cells = np.zeros((40, 40), dtype=int)
    cells[2:18, 2:18] = 1
    cells[22:38, 22:38] = 2
    nuclei = np.zeros_like(cells)
    nuclei[6:14, 6:14] = 1
    nuclei[26:34, 26:34] = 2
    return nuclei, cells


def test_colocalization_accepts_float_threshold():
    from lib.external.cp_emulator import measure_colocalization

    rng = np.random.default_rng(0)
    a = rng.random(100)
    b = a + 0.1 * rng.random(100)
    result = measure_colocalization(a, b, threshold=0.15)
    assert len(result) == 7
    assert np.isfinite(result[0])


def test_cytoplasm_reconciled_masks():
    from lib.phenotype.identify_cytoplasm_cellpose import identify_cytoplasm_cellpose

    nuclei, cells = _two_cells()
    cytoplasms = identify_cytoplasm_cellpose(nuclei, cells)
    assert set(np.unique(cytoplasms)) == {0, 1, 2}
    assert not np.any(cytoplasms[nuclei > 0])


def test_cytoplasm_unreconciled_masks_raise_clear_error():
    from lib.phenotype.identify_cytoplasm_cellpose import identify_cytoplasm_cellpose

    nuclei, cells = _two_cells()
    nuclei[30:32, 5:7] = 3
    with pytest.raises(ValueError, match="reconcile"):
        identify_cytoplasm_cellpose(nuclei, cells)


@pytest.mark.parametrize("foci_channel", [1, [1], [0, 1]])
def test_second_obj_foci_channel_int_or_list(foci_channel):
    from lib.phenotype.extract_phenotype_second_objs import (
        extract_phenotype_second_objs,
    )

    rng = np.random.default_rng(0)
    data = rng.integers(0, 50, size=(2, 40, 40)).astype(np.uint16)
    data[:, 8:10, 8:10] = 4000
    _, second_objs = _two_cells()
    df = extract_phenotype_second_objs(
        data,
        second_objs,
        wildcards={"plate": 1, "well": "A1", "tile": 0},
        foci_channel=foci_channel,
        channel_names=["dapi", "gfp"],
    )
    channels = [foci_channel] if isinstance(foci_channel, int) else foci_channel
    names = ["dapi", "gfp"]
    for ch in channels:
        assert f"second_obj_{names[ch]}_foci_count" in df.columns


def test_nuclei_centroids_keyed_by_nucleus_label():
    from lib.phenotype.segment_secondary_object import (
        _postprocess_secondary_objects,
        nuclei_centroids_from_table,
    )

    phenotype_info = pd.DataFrame(
        {"cell": [7, 3], "i": [10.0, 30.0], "j": [10.0, 30.0]}
    )
    centroids = nuclei_centroids_from_table(phenotype_info)
    assert centroids == {7: (10.0, 10.0), 3: (30.0, 30.0)}

    _, cells = _two_cells()
    second_objs = np.zeros_like(cells)
    second_objs[28:32, 28:32] = 1
    for nuclei_centroids in (centroids, phenotype_info):
        _, table, _ = _postprocess_secondary_objects(
            second_objs,
            cells,
            None,
            second_obj_min_size=0,
            second_obj_max_size=1000,
            size_filter_method="area",
            max_objects_per_cell=10,
            overlap_threshold=0.1,
            nuclei_centroids=nuclei_centroids,
            max_total_objects=None,
        )
        mapping = table["second_obj_cell_mapping"]
        assert mapping["nearest_nucleus_id"].tolist() == [3]


class _FakeModel:
    """Stand-in for a CellposeModel that records the eval kwargs."""

    calls = []

    def eval(self, image, diameter=None, **kwargs):
        type(self).calls.append(kwargs)
        mask = np.zeros(image.shape[-2:], dtype=int)
        mask[10:20, 10:20] = 1
        return mask, None, None


def test_segment_cellpose_default_kwargs(monkeypatch):
    import lib.shared.segment_cellpose as sc

    monkeypatch.setattr(sc, "create_cellpose_model", lambda *a, **k: _FakeModel())
    _FakeModel.calls = []
    data = np.random.default_rng(0).integers(1, 100, size=(2, 32, 32)).astype(float)

    sc.segment_cellpose(data, 0, 1, 10, 20, cells=False)
    sc.segment_cellpose(data, 0, 1, 10, 20, cells=True, reconcile=False)
    assert _FakeModel.calls == [
        {"flow_threshold": 0.4, "cellprob_threshold": 0},
        {"flow_threshold": 0.4, "cellprob_threshold": 0},
        {"flow_threshold": 0.4, "cellprob_threshold": 0, "channels": [2, 3]},
    ]

    _FakeModel.calls = []
    kwargs = dict(
        flow_threshold=0.5,
        cellprob_threshold=-1,
        nuclei_flow_threshold=0.3,
        cell_cellprob_threshold=2,
    )
    sc.segment_cellpose(data, 0, 1, 10, 20, cellpose_kwargs=kwargs, reconcile=False)
    assert _FakeModel.calls == [
        {"flow_threshold": 0.3, "cellprob_threshold": -1},
        {"flow_threshold": 0.5, "cellprob_threshold": 2, "channels": [2, 3]},
    ]
    assert kwargs["nuclei_flow_threshold"] == 0.3
    assert kwargs["cell_cellprob_threshold"] == 2


def test_cp_measure_failure_keeps_other_measurements(monkeypatch):
    pytest.importorskip("cp_measure")
    import lib.phenotype.extract_phenotype_cp_measure as cpm

    def broken(mask, image):
        raise RuntimeError("boom")

    def working(mask, image):
        return {"Value": np.ones(len(np.unique(mask)) - 1)}

    monkeypatch.setattr(
        cpm, "get_core_measurements", lambda: {"broken": broken, "working": working}
    )
    _, cells = _two_cells()
    with pytest.warns(UserWarning, match="broken"):
        features = cpm.get_single_object_features(
            np.ones(cells.shape), cells, "cell_DAPI__"
        )
    assert [k.endswith("Value") for k in features] == [True]


def test_cp_measure_column_names():
    pytest.importorskip("cp_measure")
    from lib.phenotype.extract_phenotype_cp_measure import extract_phenotype_cp_measure

    nuclei, cells = _two_cells()
    rng = np.random.default_rng(0)
    data = rng.random((2, 40, 40))
    df = extract_phenotype_cp_measure(
        data, nuclei, cells, channel_names=["DAPI", "GFP"]
    )
    assert not df.empty
    assert not any("___" in c for c in df.columns)
    assert any(c.startswith("nucleus_DAPI__") for c in df.columns)
    assert any(c.startswith("cell_neighbor__") for c in df.columns)


def test_second_obj_module_imports_without_plotting_deps():
    code = (
        "import sys; sys.modules['microfilm'] = None; "
        "sys.modules['microfilm.microplot'] = None; "
        "import lib.phenotype.segment_secondary_object"
    )
    subprocess.run([sys.executable, "-c", code], cwd=_WORKFLOW, check=True)
