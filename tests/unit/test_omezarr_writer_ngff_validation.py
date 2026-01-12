import json

import numpy as np
import pytest
import zarr
from ome_zarr.format import FormatV04
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader

from workflow.lib.shared.omezarr_writer import write_image_omezarr


def _read_zattrs(zarr_path):
    return json.loads((zarr_path / ".zattrs").read_text())


@pytest.mark.unit
def test_omezarr_output_conforms_to_ome_ngff_v04_and_is_readable(tmp_path):
    """
    Validate core OME-NGFF metadata for an OME-Zarr v0.4 (Zarr v2) image output.

    The ome-zarr-py docs describe `write_image(...)` producing `multiscales` metadata and
    `Reader(parse_url(...))` reading it back:
    https://ome-zarr.readthedocs.io/en/stable/python.html
    """
    out = tmp_path / "img.ome.zarr"

    # (C, Y, X)
    img = (np.arange(2 * 64 * 80, dtype=np.uint16)).reshape((2, 64, 80))
    pixel_size_um = 0.5

    write_image_omezarr(
        image_data=img,
        out_path=str(out),
        axes="cyx",
        pixel_size_um=pixel_size_um,
        channel_names=["c0", "c1"],
    )

    # Zarr v2 layout checks (no zarr.json, has .zgroup/.zattrs)
    assert (out / ".zgroup").exists()
    assert (out / ".zattrs").exists()
    assert not (out / "zarr.json").exists()

    # Validate required NGFF metadata
    zattrs = _read_zattrs(out)
    assert "multiscales" in zattrs
    assert isinstance(zattrs["multiscales"], list)
    assert len(zattrs["multiscales"]) == 1

    ms0 = zattrs["multiscales"][0]
    assert ms0["version"] == "0.4"
    assert ms0["axes"] == [
        {"name": "c", "type": "channel"},
        {"name": "y", "type": "space"},
        {"name": "x", "type": "space"},
    ]

    datasets = ms0["datasets"]
    assert isinstance(datasets, list)
    assert len(datasets) >= 1
    assert datasets[0]["path"] == "0"
    assert datasets[0]["coordinateTransformations"] == [
        {"type": "scale", "scale": [1.0, pixel_size_um, pixel_size_um]}
    ]

    # Ensure every declared dataset path exists as an array in the group
    root = zarr.open_group(str(out), mode="r")
    for ds in datasets:
        assert ds["path"] in root
        assert hasattr(root[ds["path"]], "shape")

    # Reader roundtrip (use FormatV04 to avoid v0.5 default mismatch warnings)
    reader = Reader(parse_url(str(out), fmt=FormatV04()))
    nodes = list(reader())
    assert len(nodes) >= 1

    img_node = nodes[0]
    data = img_node.data
    assert isinstance(data, list)
    assert len(data) >= 1

    level0 = data[0]
    try:
        import dask.array as da

        if isinstance(level0, da.Array):
            level0 = level0.compute()
    except Exception:
        # If dask isn't available for some reason, the reader may already return numpy.
        pass

    np.testing.assert_array_equal(level0, img)
