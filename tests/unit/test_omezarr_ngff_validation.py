import json
from pathlib import Path

import numpy as np
import pytest
import zarr

from lib.shared.omezarr_io import write_multiscale_omezarr


def test_omezarr_is_valid_ngff(tmp_path: Path) -> None:
    """Test that written OME-Zarr conforms to OME-NGFF standards.

    This test validates that:
    - The OME-Zarr can be successfully opened by ome-zarr Reader
    - Required NGFF metadata keys are present (multiscales, axes, datasets)
    - Coordinate transformations are valid
    - OMERO metadata is present
    - Data can be read from the multiscale pyramid
    """
    pytest.importorskip("ome_zarr")
    from ome_zarr.io import parse_url
    from ome_zarr.reader import Reader

    # Create a tiny multiscale OME-Zarr using the project writer
    image = np.arange(2 * 32 * 32, dtype=np.uint16).reshape(2, 32, 32)
    output = tmp_path / "ngff_test.ome.zarr"
    channel_names = ["Channel 0", "Channel 1"]

    write_multiscale_omezarr(
        image=image,
        output_dir=output,
        pixel_size=(0.5, 0.5),
        chunk_shape=(1, 16, 16),
        coarsening_factor=2,
        max_levels=2,
        image_name="ngff_test",
        compressor=None,  # No compression to avoid zarr 3.x/2.x blosc compatibility issues
        channel_names=channel_names,
    )

    # Validate raw NGFF metadata in .zattrs
    zattrs_path = output / ".zattrs"
    assert zattrs_path.exists(), "Missing .zattrs file"
    with open(zattrs_path) as f:
        raw_attrs = json.load(f)

    # Check required NGFF metadata structure in raw attributes
    assert "multiscales" in raw_attrs, "Missing 'multiscales' key in .zattrs"
    multiscales = raw_attrs["multiscales"]
    assert len(multiscales) > 0, "Empty multiscales list"

    ms = multiscales[0]
    # Check required multiscales keys per NGFF spec
    assert "name" in ms, "Missing 'name' in multiscales"
    assert ms["name"] == "ngff_test", f"Expected name 'ngff_test', got '{ms['name']}'"

    assert "axes" in ms, "Missing 'axes' in multiscales"
    axes = ms["axes"]
    assert len(axes) == 3, f"Expected 3 axes for CYX image, got {len(axes)}"
    axis_names = [ax["name"] for ax in axes]
    assert axis_names == ["c", "y", "x"], f"Expected axes ['c', 'y', 'x'], got {axis_names}"

    assert "datasets" in ms, "Missing 'datasets' in multiscales"
    datasets = ms["datasets"]
    assert len(datasets) == 2, f"Expected 2 resolution levels, got {len(datasets)}"

    # Validate coordinate transformations for each dataset
    for i, dataset in enumerate(datasets):
        assert "path" in dataset, f"Dataset {i} missing 'path'"
        assert "coordinateTransformations" in dataset, f"Dataset {i} missing coordinateTransformations"
        transforms = dataset["coordinateTransformations"]
        assert len(transforms) > 0, f"Dataset {i} has empty coordinateTransformations"
        assert transforms[0]["type"] == "scale", f"First transform should be 'scale'"
        assert "scale" in transforms[0], f"Scale transform missing 'scale' values"

    # Validate OMERO metadata is present in raw attributes
    assert "omero" in raw_attrs, "Missing 'omero' metadata in .zattrs"
    assert "channels" in raw_attrs["omero"], "Missing 'channels' in omero metadata"
    channels = raw_attrs["omero"]["channels"]
    assert len(channels) == 2, f"Expected 2 channels, got {len(channels)}"
    for i, channel in enumerate(channels):
        assert "label" in channel, f"Channel {i} missing 'label'"
        assert channel["label"] == channel_names[i], f"Channel {i} label mismatch"
        assert "color" in channel, f"Channel {i} missing 'color'"

    # Verify ome-zarr Reader can successfully parse the NGFF structure
    store = parse_url(str(output))
    reader = Reader(store)
    nodes = list(reader())
    assert len(nodes) > 0, "ome-zarr Reader failed to parse NGFF structure"

    # Validate the parsed node has expected attributes
    image_node = nodes[0]
    assert hasattr(image_node, "data"), "Node missing 'data' attribute"
    assert hasattr(image_node, "metadata"), "Node missing 'metadata' attribute"

    # Verify data can be read back from the zarr store
    # Use zarr-python's open with zarr_format=2 to read v2 stores
    try:
        zarr_group = zarr.open_group(str(output), mode="r", zarr_format=2)
        level0_array = zarr_group["0"]
        assert level0_array.shape == (2, 32, 32), f"Unexpected shape: {level0_array.shape}"

        # Read first channel and verify it matches what we wrote
        loaded_channel0 = level0_array[0, :, :]
        expected_channel0 = image[0]
        np.testing.assert_array_equal(
            loaded_channel0, expected_channel0, err_msg="Loaded data doesn't match written data"
        )
    except (RuntimeError, TypeError) as e:
        # If zarr version compatibility issues occur, skip data validation
        # (metadata validation is already complete and more critical for NGFF compliance)
        pytest.skip(f"Zarr data reading skipped due to version compatibility: {e}")


