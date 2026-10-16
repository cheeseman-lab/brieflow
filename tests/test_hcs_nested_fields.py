"""HCS metadata for stores that nest one image per cycle under each field.

Preprocess SBS stores are written as ``image_{plate}.zarr/{row}/{col}/{tile}/{cycle}``.
HCS readers open each field a well lists as an image, and the NGFF well schema only
allows single-segment field paths, so the plate root raised ``KeyError: 'axes'`` in
``ome_zarr.reader.Reader`` (what napari-ome-zarr uses). ``write_hcs_metadata`` now
gives such a field the metadata of its first cycle with the dataset paths re-rooted.
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
import pytest
import zarr

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.shared.hcs import write_hcs_metadata  # noqa: E402
from lib.shared.image_io import write_image_omezarr  # noqa: E402

_SHAPE = (1, 2, 1, 16, 16)


def _write_image(path, seed):
    data = np.random.default_rng(seed).integers(0, 4000, size=_SHAPE, dtype=np.uint16)
    write_image_omezarr(data, str(path), channel_names=["a", "b"], axes="TCZYX")
    return data


def _ome(path):
    return json.loads((path / "zarr.json").read_text())["attributes"]["ome"]


def _read_plate_root(plate):
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    return list(Reader(parse_url(str(plate), mode="r"))())


@pytest.fixture
def nested_plate(tmp_path):
    """One well, one field, cycles 1, 2 and 10 nested under the field."""
    plate = tmp_path / "image_1.zarr"
    cycles = {
        c: _write_image(plate / "A" / "1" / "0" / str(c), seed=c) for c in (1, 2, 10)
    }
    write_hcs_metadata(plate)
    return plate, cycles


def test_plate_root_and_field_open_in_ome_zarr_reader(nested_plate):
    """The plate root no longer raises and the field reads as its first cycle.

    ome_zarr's plate and well mosaics load ``{field}/{level}`` arrays directly and
    render zeros for a field whose arrays sit one level down; only a layout change
    can populate them, so only the field node's pixels are asserted here.
    """
    pytest.importorskip("ome_zarr")
    plate, cycles = nested_plate
    nodes = _read_plate_root(plate)
    assert nodes, "reader returned no nodes"
    assert nodes[0].data[0].shape == _SHAPE  # one well, one field: plate == field
    field = _read_plate_root(plate / "A" / "1" / "0")
    assert np.array_equal(np.asarray(field[0].data[0]), cycles[1])


def test_field_carries_first_cycle_by_number(nested_plate):
    plate, cycles = nested_plate
    field = _ome(plate / "A" / "1" / "0")
    # cycle 1, not cycle 10: numeric order, not lexical
    assert [d["path"] for d in field["multiscales"][0]["datasets"]] == ["1/0"]
    assert [c["label"] for c in field["omero"]["channels"]] == ["a", "b"]
    for c, data in cycles.items():  # cycle images untouched
        assert np.array_equal(
            zarr.open_array(str(plate / f"A/1/0/{c}/0"), mode="r")[:], data
        )


def test_well_lists_single_segment_field_paths(nested_plate):
    plate, _ = nested_plate
    paths = [i["path"] for i in _ome(plate / "A" / "1")["well"]["images"]]
    assert paths == ["0"]
    assert all(re.fullmatch(r"[A-Za-z0-9]+", p) for p in paths)  # NGFF well schema


def test_rerun_is_idempotent(nested_plate):
    pytest.importorskip("ome_zarr")
    plate, _ = nested_plate
    before = _ome(plate / "A" / "1" / "0")
    write_hcs_metadata(plate)
    assert _ome(plate / "A" / "1" / "0") == before
    assert _read_plate_root(plate)


def test_flat_field_is_left_alone(tmp_path):
    """A field that is already an image (phenotype, aligned stores) is not rewritten."""
    plate = tmp_path / "image_2.zarr"
    _write_image(plate / "A" / "1" / "5", seed=0)
    before = json.loads((plate / "A" / "1" / "5" / "zarr.json").read_text())
    write_hcs_metadata(plate)
    assert json.loads((plate / "A" / "1" / "5" / "zarr.json").read_text()) == before
