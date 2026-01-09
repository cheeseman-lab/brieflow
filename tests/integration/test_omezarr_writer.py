"""Tests for the OME-Zarr writer utilities."""

import json
from pathlib import Path

import numpy as np

from lib.shared.omezarr_io import write_multiscale_omezarr
from lib.shared.omezarr_utils import default_omero_color_ints


def test_write_multiscale_omezarr(tmp_path):
    image = np.arange(2 * 130 * 130, dtype=np.uint16).reshape(2, 130, 130)
    output = tmp_path / "test_image.ome.zarr"

    # Note: Blosc compression is currently disabled in the writer (see omezarr_io.py line 200)
    # So we test with no compression
    compressor = None

    result_path = write_multiscale_omezarr(
        image=image,
        output_dir=output,
        pixel_size=(0.25, 0.25),
        chunk_shape=(1, 64, 64),
        coarsening_factor=2,
        max_levels=3,
        image_name="test_image",
        compressor=compressor,
    )

    assert result_path == Path(output)
    assert (output / ".zattrs").exists()
    assert (output / ".zgroup").exists()

    with open(output / ".zattrs", "r") as handle:
        attrs = json.load(handle)

    multiscale_entry = attrs["multiscales"][0]
    assert multiscale_entry["name"] == "test_image"
    datasets = multiscale_entry["datasets"]
    assert len(datasets) == 3
    assert datasets[0]["path"] == "0"
    assert datasets[0]["coordinateTransformations"][0]["scale"][1:] == [0.25, 0.25]

    scale0_meta = json.loads((output / "0" / ".zarray").read_text())
    assert scale0_meta["shape"] == [2, 130, 130]
    assert scale0_meta["chunks"] == [1, 64, 64]

    # Verify pyramid downscaling
    scale2_meta = json.loads((output / "2" / ".zarray").read_text())
    assert scale2_meta["shape"][1] < scale0_meta["shape"][1]
    assert (output / "2" / "0" / "0" / "0").exists()

    channels = attrs["omero"]["channels"]
    assert len(channels) == 2
    # Colors are now stored as hex strings (NGFF v0.4 format)
    assert len([channel["color"] for channel in channels]) == 2
    # Verify each channel has a color (hex string)
    for channel in channels:
        assert "color" in channel
        assert isinstance(channel["color"], str)
        assert len(channel["color"]) == 6  # RRGGBB format

    # Edge chunks - verify they exist and have correct uncompressed byte size
    # (since compression is disabled, chunks are stored as raw bytes)
    edge_chunk = output / "0" / "0" / "2" / "2"
    assert edge_chunk.exists()
    expected_chunk_bytes = np.prod((1, 64, 64)) * image.dtype.itemsize
    chunk_data = edge_chunk.read_bytes()
    assert len(chunk_data) == expected_chunk_bytes
