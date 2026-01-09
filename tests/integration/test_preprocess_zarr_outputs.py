"""
Integration tests validating Zarr outputs existence and metadata.
Assumes small_test_analysis data has been processed using the provided scripts.
"""
import json
from pathlib import Path

import yaml

from lib.shared.file_utils import get_filename


TEST_ANALYSIS_PATH = Path(__file__).resolve().parents[1] / "small_test_analysis"
CONFIG_FILE_PATH = TEST_ANALYSIS_PATH / "config" / "config.yml"

TEST_PLATE = 1
TEST_WELL = "A1"
TEST_CYCLE = 11
TEST_TILE_SBS = 0
TEST_TILE_PHENOTYPE = 5


def _load_config():
    with open(CONFIG_FILE_PATH, "r") as f:
        return yaml.safe_load(f)


def test_sbs_zarr_exists_and_matches_config():
    config = _load_config()
    root = TEST_ANALYSIS_PATH / "brieflow_output" / "preprocess" / "omezarr" / "sbs"
    sbs_zarr = root / get_filename(
        {
            "plate": TEST_PLATE,
            "well": TEST_WELL,
            "tile": TEST_TILE_SBS,
            "cycle": TEST_CYCLE,
        },
        "image",
        "zarr",
    )
    assert sbs_zarr.exists() and sbs_zarr.is_dir(), f"Missing Zarr at {sbs_zarr}"

    attrs = json.loads((sbs_zarr / ".zattrs").read_text())
    multiscales = attrs["multiscales"]
    assert len(multiscales) == 1
    ms = multiscales[0]
    assert ms["type"] == "image"

    # Channel labels match config if present
    channel_names = config["sbs"].get("channel_names", [])
    if channel_names:
        omero_channels = attrs["omero"]["channels"]
        assert [c["label"] for c in omero_channels][: len(channel_names)] == channel_names

    # Level 0 chunking reflects configured chunk shape
    configured_chunk = tuple(config["preprocess"]["omezarr_chunk_shape"])
    zarray0 = json.loads((sbs_zarr / "0" / ".zarray").read_text())
    # SBS images are CZYX; chunk expands to (C, Z=1, Y, X) when config is length-3
    expected_chunks = [configured_chunk[0], 1, configured_chunk[1], configured_chunk[2]]
    assert zarray0["chunks"] == expected_chunks


def test_phenotype_zarr_exists_and_has_multiscales():
    root = TEST_ANALYSIS_PATH / "brieflow_output" / "preprocess" / "omezarr" / "phenotype"
    ph_zarr = root / get_filename(
        {
            "plate": TEST_PLATE,
            "well": TEST_WELL,
            "tile": TEST_TILE_PHENOTYPE,
        },
        "image",
        "zarr",
    )
    assert ph_zarr.exists() and ph_zarr.is_dir(), f"Missing Zarr at {ph_zarr}"
    attrs = json.loads((ph_zarr / ".zattrs").read_text())
    assert "multiscales" in attrs and len(attrs["multiscales"]) == 1


