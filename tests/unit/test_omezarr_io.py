import json
from pathlib import Path

import numpy as np
import pytest

from lib.shared.omezarr_io import write_multiscale_omezarr


def test_write_multiscale_cyx_metadata_and_chunks(tmp_path: Path) -> None:
    image = (np.arange(2 * 130 * 130, dtype=np.uint16).reshape(2, 130, 130))
    output = tmp_path / "img_cyx.zarr"

    compressor = {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2}
    path = write_multiscale_omezarr(
        image=image,
        output_dir=output,
        pixel_size=(0.25, 0.25),
        chunk_shape=(1, 64, 64),
        coarsening_factor=2,
        max_levels=3,
        image_name="img_cyx",
        compressor=compressor,
        channel_names=["Channel A", "Channel B"],
    )

    assert path == output
    assert (output / ".zgroup").exists()
    attrs = json.loads((output / ".zattrs").read_text())
    multiscales = attrs["multiscales"]
    assert len(multiscales) == 1
    ms = multiscales[0]
    assert ms["name"] == "img_cyx"
    axes = [a["name"] for a in ms["axes"]]
    assert axes == ["c", "y", "x"]
    datasets = ms["datasets"]
    assert len(datasets) == 3
    assert datasets[0]["path"] == "0"
    assert datasets[0]["shape"] == [2, 130, 130]
    assert datasets[0]["coordinateTransformations"][0]["type"] == "scale"
    assert datasets[0]["coordinateTransformations"][0]["scale"] == [1, 0.25, 0.25]

    zarray0 = json.loads((output / "0" / ".zarray").read_text())
    assert zarray0["shape"] == [2, 130, 130]
    assert zarray0["chunks"] == [1, 64, 64]
    assert zarray0["compressor"]["id"] == "blosc"

    # OMERO metadata
    channels = attrs["omero"]["channels"]
    assert [c["label"] for c in channels] == ["Channel A", "Channel B"]
    assert all("color" in c for c in channels)


def test_write_multiscale_czyx_metadata_axes(tmp_path: Path) -> None:
    # CZYX
    image = (np.arange(2 * 3 * 64 * 64, dtype=np.uint16).reshape(2, 3, 64, 64))
    output = tmp_path / "img_czyx.zarr"

    path = write_multiscale_omezarr(
        image=image,
        output_dir=output,
        pixel_size=(1.5, 0.5, 0.5),  # Z, Y, X (micrometers)
        chunk_shape=(1, 1, 32, 32),
        coarsening_factor=2,
        max_levels=2,
        image_name="img_czyx",
        compressor=None,
    )
    assert path == output
    attrs = json.loads((output / ".zattrs").read_text())
    ms = attrs["multiscales"][0]
    axes = [a["name"] for a in ms["axes"]]
    assert axes == ["c", "z", "y", "x"]
    # Level 0 scale: [1, pixelZ, pixelY, pixelX]
    level0 = ms["datasets"][0]
    scale = level0["coordinateTransformations"][0]["scale"]
    assert scale[0] == 1
    assert scale[1] == pytest.approx(1.5)
    assert scale[2] == pytest.approx(0.5)
    assert scale[3] == pytest.approx(0.5)


