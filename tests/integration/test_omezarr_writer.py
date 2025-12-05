"""Tests for the OME-Zarr writer utilities."""

import json
from pathlib import Path

import numpy as np

from lib.preprocess.preprocess import write_multiscale_omezarr
from lib.shared.omezarr_utils import default_omero_color_ints


def test_write_multiscale_omezarr(tmp_path):
    image = np.arange(2 * 130 * 130, dtype=np.uint16).reshape(2, 130, 130)
    output = tmp_path / "test_image.ome.zarr"

    result_path = write_multiscale_omezarr(
        image=image,
        output_dir=output,
        pixel_size=(0.25, 0.25),
        chunk_shape=(1, 64, 64),
        coarsening_factor=2,
        max_levels=3,
    )

    assert result_path == Path(output)
    assert (output / ".zattrs").exists()
    assert (output / ".zgroup").exists()

    with open(output / ".zattrs", "r") as handle:
        attrs = json.load(handle)

    datasets = attrs["multiscales"][0]["datasets"]
    assert len(datasets) == 3
    assert datasets[0]["path"] == "scale0"
    assert datasets[0]["coordinateTransformations"][0]["scale"][1:] == [0.25, 0.25]

    scale0_meta = json.loads((output / "scale0" / ".zarray").read_text())
    assert scale0_meta["shape"] == [2, 130, 130]
    assert scale0_meta["chunks"] == [1, 64, 64]

    # Verify pyramid downscaling
    scale2_meta = json.loads((output / "scale2" / ".zarray").read_text())
    assert scale2_meta["shape"][1] < scale0_meta["shape"][1]
    assert (output / "scale2" / "0.0.0").exists()

    channels = attrs["omero"]["channels"]
    assert len(channels) == 2
    expected_colors = default_omero_color_ints(2)
    assert [channel["color"] for channel in channels] == expected_colors

    # Edge chunks (dim < chunk) should still have full chunk byte-size
    edge_chunk = output / "scale0" / "0.2.2"
    assert edge_chunk.exists()
    expected_chunk_bytes = np.prod((1, 64, 64)) * image.dtype.itemsize
    assert edge_chunk.stat().st_size == expected_chunk_bytes
