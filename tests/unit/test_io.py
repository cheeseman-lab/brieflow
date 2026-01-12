import pytest
import numpy as np
from pathlib import Path
import json
import zarr
from tifffile import imread as tiff_imread
from tifffile import imwrite as tiff_imwrite

from workflow.lib.shared.io import read_image, save_image
from workflow.lib.shared.omezarr_io import (
    write_multiscale_omezarr,
)  # Import for direct comparison/validation


def _tiff_imwrite(path, image):
    tiff_imwrite(str(path), image)


# --- Fixtures ---


@pytest.fixture
def dummy_image_3d_uint16():
    """Returns a dummy 3D (C, Y, X) uint16 numpy array."""
    return np.random.randint(0, 2**16 - 1, (3, 100, 100), dtype=np.uint16)


@pytest.fixture
def dummy_image_2d_uint16():
    """Returns a dummy 2D (Y, X) uint16 numpy array."""
    return np.random.randint(0, 2**16 - 1, (100, 100), dtype=np.uint16)


@pytest.fixture
def dummy_image_4d_uint16():
    """Returns a dummy 4D (C, Z, Y, X) uint16 numpy array."""
    return np.random.randint(0, 2**16 - 1, (2, 5, 50, 50), dtype=np.uint16)


# --- Test Cases for save_image ---


def test_save_image_to_tiff_3d(tmp_path, dummy_image_3d_uint16):
    """Test saving a 3D image to TIFF format."""
    file_path = tmp_path / "test_3d.tiff"
    save_image(dummy_image_3d_uint16, file_path)
    assert file_path.exists()
    reloaded_image = tiff_imread(file_path)
    np.testing.assert_array_equal(dummy_image_3d_uint16, reloaded_image)


def test_save_image_to_tiff_2d_normalized(tmp_path, dummy_image_2d_uint16):
    """Test saving a 2D image to TIFF, ensuring it's treated as 1CYX."""
    file_path = tmp_path / "test_2d.tiff"
    save_image(dummy_image_2d_uint16, file_path)
    assert file_path.exists()
    reloaded_image = tiff_imread(file_path)
    # save_image will expand 2D to 3D (1, Y, X) for TIFF writer if it expects multi-channel
    # tifffile.imread usually handles 2D if saved as 2D, but our wrapper passes 3D to omezarr
    # For TIFF direct save, it's fine.
    # The key is that `read_image` normalizes shapes.
    # For this specific test, verify that what was saved is what was intended to be saved to TIFF directly.
    # The wrapper's normalization for OME-Zarr doesn't apply to TIFF writes.
    np.testing.assert_array_equal(dummy_image_2d_uint16, reloaded_image)


def test_save_image_to_omezarr_3d(tmp_path, dummy_image_3d_uint16):
    """Test saving a 3D image to OME-Zarr format with metadata."""
    zarr_path = tmp_path / "test_3d.zarr"
    channel_names = ["Ch1", "Ch2", "Ch3"]
    pixel_size = (0.5, 0.5)  # YX

    save_image(
        dummy_image_3d_uint16,
        zarr_path,
        pixel_size=pixel_size,
        channel_names=channel_names,
        coarsening_factor=2,
        max_levels=2,
    )

    assert zarr_path.is_dir()
    assert (zarr_path / ".zgroup").exists()
    assert (zarr_path / ".zattrs").exists()
    assert (zarr_path / "0").is_dir()  # Level 0 exists

    # Verify content and metadata
    reloaded_image = zarr.open(str(zarr_path), mode="r")["0"][:]
    np.testing.assert_array_equal(dummy_image_3d_uint16, reloaded_image)

    attrs = json.loads((zarr_path / ".zattrs").read_text())
    assert attrs["multiscales"][0]["version"] == "0.4"
    assert attrs["omero"]["channels"][0]["label"] == "Ch1"
    assert attrs["omero"]["pixel_size"]["y"] == pixel_size[0]
    assert attrs["omero"]["pixel_size"]["x"] == pixel_size[1]


def test_save_image_to_omezarr_2d_normalized(tmp_path, dummy_image_2d_uint16):
    """Test saving a 2D image to OME-Zarr, ensuring it's normalized to 1CYX."""
    zarr_path = tmp_path / "test_2d.zarr"

    save_image(dummy_image_2d_uint16, zarr_path)

    assert zarr_path.is_dir()
    reloaded_image = zarr.open(str(zarr_path), mode="r")["0"][:]

    # Expected shape (1, Y, X)
    expected_shape = (1,) + dummy_image_2d_uint16.shape
    assert reloaded_image.shape == expected_shape
    np.testing.assert_array_equal(
        dummy_image_2d_uint16[np.newaxis, ...], reloaded_image
    )


def test_save_image_to_omezarr_is_label(tmp_path, dummy_image_3d_uint16):
    """Test saving an OME-Zarr as a label image."""
    zarr_path = tmp_path / "test_labels.zarr"
    label_image = (dummy_image_3d_uint16 > 1000).astype(np.uint16)  # Create some labels

    save_image(label_image, zarr_path, is_label=True)

    assert zarr_path.is_dir()
    attrs = json.loads((zarr_path / ".zattrs").read_text())
    assert "image-label" in attrs
    # Check that OMERO channels do not have colors for label images (or are default)
    # The current omezarr_io uses default colors if not label. If label, it skips colors.
    assert (
        "color" not in attrs["omero"]["channels"][0]
    )  # Explicitly check for no color if is_label


# --- Test Cases for read_image ---


def test_read_image_from_tiff(tmp_path, dummy_image_3d_uint16):
    """Test reading a TIFF file."""
    file_path = tmp_path / "read_test.tiff"
    _tiff_imwrite(file_path, dummy_image_3d_uint16)  # Write directly using tifffile

    reloaded_image = read_image(file_path)
    np.testing.assert_array_equal(dummy_image_3d_uint16, reloaded_image)


def test_read_image_from_omezarr_multiscale(tmp_path, dummy_image_3d_uint16):
    """Test reading an OME-Zarr multiscale image, ensuring highest resolution is returned."""
    zarr_path = tmp_path / "read_test.zarr"
    # Use write_multiscale_omezarr directly to ensure multiscale structure
    write_multiscale_omezarr(
        image=dummy_image_3d_uint16,
        output_dir=zarr_path,
        coarsening_factor=2,
        max_levels=2,
        pixel_size=(0.5, 0.5),
    )

    reloaded_image = read_image(zarr_path)
    np.testing.assert_array_equal(
        dummy_image_3d_uint16, reloaded_image
    )  # Should match original resolution


def test_read_image_from_omezarr_single_level(tmp_path, dummy_image_3d_uint16):
    """Test reading an OME-Zarr with only a single '0' group directly."""
    zarr_path = tmp_path / "single_level.zarr"
    # Create a single-level zarr manually to simulate (force Zarr v2 layout)
    root = zarr.open_group(str(zarr_path), mode="w", zarr_format=2)
    root.create_dataset(
        "0",
        data=dummy_image_3d_uint16,
        shape=dummy_image_3d_uint16.shape,
        chunks=(1, 50, 50),
        dtype=dummy_image_3d_uint16.dtype,
        overwrite=True,
    )
    root.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]

    reloaded_image = read_image(zarr_path)
    np.testing.assert_array_equal(dummy_image_3d_uint16, reloaded_image)


def test_read_image_file_not_found(tmp_path):
    """Test error handling for non-existent files."""
    with pytest.raises(FileNotFoundError):
        read_image(tmp_path / "non_existent.tiff")
    with pytest.raises(ValueError, match="Could not find image data in OME-Zarr"):
        # Create a directory that looks like a zarr but without '0' or 'multiscales'
        (tmp_path / "bad.zarr").mkdir()
        (tmp_path / "bad.zarr" / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        read_image(tmp_path / "bad.zarr")


def test_read_image_zarr_not_installed(tmp_path, monkeypatch):
    """Test read_image behavior when zarr is not installed."""
    import sys

    # Simulate missing zarr by masking the import
    monkeypatch.setitem(sys.modules, "zarr", None)

    zarr_path = tmp_path / "no_zarr_test.zarr"
    zarr_path.mkdir()
    (zarr_path / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
    (zarr_path / ".zattrs").write_text(
        json.dumps({"multiscales": [{"datasets": [{"path": "0"}]}]})
    )

    with pytest.raises(ImportError, match="zarr package is required"):
        read_image(zarr_path)


# --- Test Cases for _downscale_spatial_dims from omezarr_io (indirectly tested by save_image) ---
# Direct unit tests for _downscale_spatial_dims could be added to test_omezarr_io.py if it were created.
# For now, it's implicitly tested by multiscale generation in save_image tests.
