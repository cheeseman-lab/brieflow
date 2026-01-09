from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

import numpy as np

import lib.preprocess.preprocess as preprocess_mod


def test_nd2_to_omezarr_forwards_params_and_calls_writer(monkeypatch, tmp_path: Path) -> None:
    # Arrange: fake nd2_to_tiff returns a CZYX array
    fake_image = np.zeros((3, 5, 100, 100), dtype=np.uint16)
    calls: Dict[str, Any] = {}

    def fake_nd2_to_tiff(files, channel_order_flip=False, verbose=False, preserve_z=False):
        calls["nd2_to_tiff"] = {
            "files": list(files) if isinstance(files, list) else [files],
            "channel_order_flip": channel_order_flip,
            "verbose": verbose,
            "preserve_z": preserve_z,
        }
        return fake_image

    def fake_resolve(first_file: Path):
        return (2.0, 0.5, 0.5)  # z, y, x

    def fake_write_multiscale_omezarr(**kwargs):
        calls["write_multiscale_omezarr"] = kwargs
        out = kwargs["output_dir"]
        Path(out).mkdir(parents=True, exist_ok=True)
        return Path(out)

    monkeypatch.setattr(preprocess_mod, "nd2_to_tiff", fake_nd2_to_tiff)
    monkeypatch.setattr(preprocess_mod, "_resolve_pixel_sizes", fake_resolve)
    monkeypatch.setattr(preprocess_mod, "write_multiscale_omezarr", fake_write_multiscale_omezarr)

    files: List[str] = ["a.nd2", "b.nd2"]
    outdir = tmp_path / "out.zarr"
    # Act
    result = preprocess_mod.nd2_to_omezarr(
        files=files,
        output_dir=outdir,
        channel_order_flip=True,
        chunk_shape=(2, 128, 128),
        coarsening_factor=2,
        max_levels=2,
        verbose=False,
        compressor={"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2},
    )

    # Assert calls and forwarded parameters
    assert result == outdir
    assert calls["nd2_to_tiff"]["preserve_z"] is True
    wargs = calls["write_multiscale_omezarr"]
    assert tuple(wargs["chunk_shape"]) == (2, 128, 128)
    assert wargs["coarsening_factor"] == 2
    assert wargs["max_levels"] == 2
    assert wargs["pixel_size"] == (2.0, 0.5, 0.5)
    assert wargs["image"].shape == fake_image.shape
    assert wargs["compressor"]["id"] == "blosc"


