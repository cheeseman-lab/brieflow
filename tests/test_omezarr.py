"""
Consolidated OME-Zarr transition tests.

This file collects all zarr-specific validation tests written during the
zarr3-transition branch.  It is intended to be **removed** once the branch
is merged to main and the transition is considered stable.  Permanent
regression tests live in tests/integration/test_preprocess.py (which
already handles both TIFF and Zarr output paths).

Sections
--------
1. Fixtures .............. shared dummy arrays and temp paths
2. omezarr_writer ........ roundtrip tests for write_image/labels/table
3. NGFF compliance ....... OME-NGFF v0.4 metadata validation
4. Pixel-size / scales ... coordinate transform metadata
5. IO roundtrip .......... read_image / save_image for zarr and tiff
6. Zarr structural ........ chunk layout, compression, multiscale structure
7. target_utils .......... output_to_input() regression test
8. ND2 metadata .......... pixel-size extraction from real nd2 files
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml
import zarr
from tifffile import imread as tiff_imread
from tifffile import imwrite as tiff_imwrite

# ---------------------------------------------------------------------------
# Ensure repo root is importable (replaces old conftest.py)
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.shared.file_utils import get_filename
from workflow.lib.shared.io import read_image, save_image
from workflow.lib.shared.omezarr_io import write_multiscale_omezarr
from workflow.lib.shared.omezarr_writer import (
    write_image_omezarr,
    write_labels_omezarr,
    write_table_zarr,
)

# ===========================================================================
# Section 1: Fixtures
# ===========================================================================


@pytest.fixture
def dummy_3d_uint16():
    """Random (C, Y, X) uint16 array — the most common image layout."""
    return np.random.randint(0, 2**16 - 1, (3, 100, 100), dtype=np.uint16)


@pytest.fixture
def dummy_2d_uint16():
    """Random (Y, X) uint16 array — single-channel image."""
    return np.random.randint(0, 2**16 - 1, (100, 100), dtype=np.uint16)


# ===========================================================================
# Section 2: omezarr_writer — roundtrip tests
# ===========================================================================


class TestOmezarrWriterRoundtrip:
    """Verify write_image_omezarr, write_labels_omezarr, and write_table_zarr
    produce stores that can be read back with identical data."""

    def test_write_image_roundtrip(self, tmp_path):
        """Write a (C,Y,X) image and read level-0 back unchanged."""
        shape = (3, 64, 64)
        data = np.random.randint(0, 255, size=shape, dtype=np.uint8)
        out = tmp_path / "img.ome.zarr"

        write_image_omezarr(
            image_data=data,
            out_path=str(out),
            channel_names=["r", "g", "b"],
            axes="cyx",
        )

        store = zarr.open(str(out), mode="r")
        assert "multiscales" in store.attrs
        np.testing.assert_array_equal(data, store["0"][:])

        omero = store.attrs.get("omero")
        assert omero is not None
        assert len(omero["channels"]) == 3
        assert omero["channels"][0]["label"] == "r"

    def test_write_labels_roundtrip(self, tmp_path):
        """Write labels into an existing image store and read them back."""
        shape = (64, 64)
        out = tmp_path / "img.ome.zarr"
        write_image_omezarr(np.zeros(shape, dtype=np.uint8), str(out), axes="yx")

        labels = np.random.randint(0, 5, size=shape, dtype=np.uint32)
        write_labels_omezarr(
            label_data=labels, out_path=str(out), label_name="nuclei", axes="yx"
        )

        store = zarr.open(str(out), mode="r")
        assert "labels" in store
        assert "nuclei" in store["labels"]
        np.testing.assert_array_equal(labels, store["labels"]["nuclei"]["0"][:])

    def test_write_table_roundtrip(self, tmp_path):
        """Write a DataFrame as columnar Zarr and verify columns."""
        df = pd.DataFrame(
            {"cell_id": [1, 2, 3], "score": [0.1, 0.5, 0.9], "cls": ["A", "B", "A"]}
        )
        table_path = tmp_path / "table.zarr"
        write_table_zarr(df, str(table_path))

        store = zarr.open(str(table_path), mode="r")
        np.testing.assert_array_equal(store["cell_id"][:], df["cell_id"].values)
        np.testing.assert_array_almost_equal(store["score"][:], df["score"].values)
        assert np.all(store["cls"][:] == df["cls"].values)


# ===========================================================================
# Section 3: NGFF compliance — OME-NGFF v0.4 metadata validation
# ===========================================================================


def _read_zattrs(zarr_path: Path) -> dict:
    """Read .zattrs JSON from a Zarr v2 store."""
    return json.loads((zarr_path / ".zattrs").read_text())


class TestNGFFCompliance:
    """Validate that write_image_omezarr produces spec-compliant
    OME-NGFF v0.4 (Zarr v2) metadata."""

    def test_v04_metadata_structure(self, tmp_path):
        """Check multiscales version, axes, datasets, and coordinateTransformations."""
        out = tmp_path / "img.ome.zarr"
        img = np.arange(2 * 64 * 80, dtype=np.uint16).reshape((2, 64, 80))
        pixel_size_um = 0.5

        write_image_omezarr(
            image_data=img,
            out_path=str(out),
            axes="cyx",
            pixel_size_um=pixel_size_um,
            channel_names=["c0", "c1"],
        )

        # Zarr v2 layout: .zgroup + .zattrs, no zarr.json
        assert (out / ".zgroup").exists()
        assert (out / ".zattrs").exists()
        assert not (out / "zarr.json").exists()

        zattrs = _read_zattrs(out)
        assert "multiscales" in zattrs
        assert len(zattrs["multiscales"]) == 1

        ms0 = zattrs["multiscales"][0]
        assert ms0["version"] == "0.4"
        assert ms0["axes"] == [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ]

        datasets = ms0["datasets"]
        assert datasets[0]["path"] == "0"
        assert datasets[0]["coordinateTransformations"] == [
            {"type": "scale", "scale": [1.0, pixel_size_um, pixel_size_um]}
        ]

        # Every declared path must exist as an array
        root = zarr.open_group(str(out), mode="r")
        for ds in datasets:
            assert ds["path"] in root
            assert hasattr(root[ds["path"]], "shape")

    def test_ome_zarr_reader_roundtrip(self, tmp_path):
        """ome-zarr-py Reader can parse the store and recover pixel data."""
        ome_zarr = pytest.importorskip("ome_zarr")
        from ome_zarr.format import FormatV04
        from ome_zarr.io import parse_url
        from ome_zarr.reader import Reader

        out = tmp_path / "img.ome.zarr"
        img = np.arange(2 * 64 * 80, dtype=np.uint16).reshape((2, 64, 80))

        write_image_omezarr(
            image_data=img,
            out_path=str(out),
            axes="cyx",
            pixel_size_um=0.5,
            channel_names=["c0", "c1"],
        )

        reader = Reader(parse_url(str(out), fmt=FormatV04()))
        nodes = list(reader())
        assert len(nodes) >= 1

        level0 = nodes[0].data[0]
        try:
            import dask.array as da

            if isinstance(level0, da.Array):
                level0 = level0.compute()
        except ImportError:
            pass
        np.testing.assert_array_equal(level0, img)


# ===========================================================================
# Section 4: Pixel-size / scales
# ===========================================================================


class TestPixelSizeScales:
    """Verify pixel_size_um → coordinateTransformations scale mapping."""

    def test_scalar_pixel_size_sets_xy_scale(self, tmp_path):
        """Scalar pixel_size_um → [1.0, ps, ps] for cyx."""
        out = tmp_path / "img.zarr"
        img = np.zeros((1, 256, 256), dtype=np.uint16)
        write_image_omezarr(img, str(out), axes="cyx", pixel_size_um=0.325)

        scale0 = _read_zattrs(out)["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]["scale"]
        assert scale0 == [1.0, 0.325, 0.325]

    def test_dict_pixel_size_sets_xyz_scale(self, tmp_path):
        """Dict pixel_size_um → [1.0, z, y, x] for czyx."""
        out = tmp_path / "img3d.zarr"
        img = np.zeros((1, 2, 128, 128), dtype=np.uint16)
        write_image_omezarr(
            img,
            str(out),
            axes="czyx",
            pixel_size_um={"z": 1.5, "y": 0.325, "x": 0.325},
        )

        scale0 = _read_zattrs(out)["multiscales"][0]["datasets"][0][
            "coordinateTransformations"
        ][0]["scale"]
        assert scale0 == [1.0, 1.5, 0.325, 0.325]

    def test_z_not_downsampled_in_pyramid(self, tmp_path):
        """Pyramid levels downsample Y/X only; Z stays constant."""
        out = tmp_path / "img3d.zarr"
        img = np.zeros((1, 4, 128, 128), dtype=np.uint16)
        write_image_omezarr(
            img,
            str(out),
            axes="czyx",
            pixel_size_um={"z": 1.5, "y": 0.325, "x": 0.325},
            coarsening_factor=2,
            max_levels=2,
        )

        datasets = _read_zattrs(out)["multiscales"][0]["datasets"]
        scale0 = datasets[0]["coordinateTransformations"][0]["scale"]
        scale1 = datasets[1]["coordinateTransformations"][0]["scale"]

        assert scale0 == [1.0, 1.5, 0.325, 0.325]
        assert scale1 == [1.0, 1.5, 0.65, 0.65]


# ===========================================================================
# Section 5: IO roundtrip — read_image / save_image
# ===========================================================================


class TestIORoundtrip:
    """Verify the unified read_image/save_image API handles both TIFF and
    Zarr paths correctly."""

    def test_save_and_read_tiff_3d(self, tmp_path, dummy_3d_uint16):
        """TIFF write → read preserves data."""
        fp = tmp_path / "test.tiff"
        save_image(dummy_3d_uint16, fp)
        np.testing.assert_array_equal(dummy_3d_uint16, read_image(fp))

    def test_save_and_read_tiff_2d(self, tmp_path, dummy_2d_uint16):
        """2D TIFF write → read preserves data."""
        fp = tmp_path / "test.tiff"
        save_image(dummy_2d_uint16, fp)
        np.testing.assert_array_equal(dummy_2d_uint16, read_image(fp))

    def test_save_and_read_omezarr_3d(self, tmp_path, dummy_3d_uint16):
        """Zarr write → read preserves data and metadata."""
        zp = tmp_path / "test.zarr"
        channel_names = ["Ch1", "Ch2", "Ch3"]
        save_image(
            dummy_3d_uint16,
            zp,
            pixel_size=(0.5, 0.5),
            channel_names=channel_names,
            coarsening_factor=2,
            max_levels=2,
        )

        assert zp.is_dir()
        assert (zp / ".zgroup").exists()
        np.testing.assert_array_equal(
            dummy_3d_uint16, zarr.open(str(zp), mode="r")["0"][:]
        )

        attrs = _read_zattrs(zp)
        assert attrs["multiscales"][0]["version"] == "0.4"
        assert attrs["omero"]["channels"][0]["label"] == "Ch1"

    def test_save_omezarr_2d_gets_singleton_channel(self, tmp_path, dummy_2d_uint16):
        """2D array saved as zarr is expanded to (1,Y,X)."""
        zp = tmp_path / "test.zarr"
        save_image(dummy_2d_uint16, zp)

        stored = zarr.open(str(zp), mode="r")["0"][:]
        assert stored.shape == (1,) + dummy_2d_uint16.shape
        np.testing.assert_array_equal(dummy_2d_uint16[np.newaxis, ...], stored)

    def test_save_omezarr_label_flag(self, tmp_path, dummy_3d_uint16):
        """is_label=True produces image-label metadata without channel colors."""
        zp = tmp_path / "labels.zarr"
        label_img = (dummy_3d_uint16 > 1000).astype(np.uint16)
        save_image(label_img, zp, is_label=True)

        attrs = _read_zattrs(zp)
        assert "image-label" in attrs
        assert "color" not in attrs["omero"]["channels"][0]

    def test_read_omezarr_multiscale(self, tmp_path, dummy_3d_uint16):
        """read_image returns full-resolution level from a multiscale store."""
        zp = tmp_path / "ms.zarr"
        write_multiscale_omezarr(
            image=dummy_3d_uint16,
            output_dir=zp,
            coarsening_factor=2,
            max_levels=2,
            pixel_size=(0.5, 0.5),
        )
        np.testing.assert_array_equal(dummy_3d_uint16, read_image(zp))

    def test_read_omezarr_single_level(self, tmp_path, dummy_3d_uint16):
        """read_image handles a store with only one '0' group."""
        zp = tmp_path / "single.zarr"
        root = zarr.open_group(str(zp), mode="w", zarr_format=2)
        root.create_dataset(
            "0",
            data=dummy_3d_uint16,
            shape=dummy_3d_uint16.shape,
            chunks=(1, 50, 50),
            dtype=dummy_3d_uint16.dtype,
            overwrite=True,
        )
        root.attrs["multiscales"] = [{"datasets": [{"path": "0"}]}]

        np.testing.assert_array_equal(dummy_3d_uint16, read_image(zp))

    def test_read_image_file_not_found(self, tmp_path):
        """Appropriate errors for missing files."""
        with pytest.raises(FileNotFoundError):
            read_image(tmp_path / "nope.tiff")

        # Empty zarr dir without array data
        bad = tmp_path / "bad.zarr"
        bad.mkdir()
        (bad / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        with pytest.raises(ValueError, match="Could not find image data"):
            read_image(bad)

    def test_read_image_zarr_not_installed(self, tmp_path, monkeypatch):
        """ImportError raised when zarr package is missing."""
        monkeypatch.setitem(sys.modules, "zarr", None)

        zp = tmp_path / "no_zarr.zarr"
        zp.mkdir()
        (zp / ".zgroup").write_text(json.dumps({"zarr_format": 2}))
        (zp / ".zattrs").write_text(
            json.dumps({"multiscales": [{"datasets": [{"path": "0"}]}]})
        )
        with pytest.raises(ImportError, match="zarr package is required"):
            read_image(zp)


# ===========================================================================
# Section 6: Zarr structural — chunks, compression, multiscale from pipeline
# ===========================================================================

# These integration tests read artifacts from a prior Snakemake run.
# They skip gracefully if the output directory is not present.

_TEST_ANALYSIS = Path(__file__).resolve().parent / "small_test_analysis"
_TEST_PLATE = 1
_TEST_WELL = "A1"
_TEST_CYCLE = 11
_TEST_TILE_SBS = 0
_TEST_TILE_PHENOTYPE = 5


def _resolve_output_dir() -> Path:
    """Find the brieflow output directory from a prior test run."""
    canonical = _TEST_ANALYSIS / "brieflow_output"
    if canonical.exists():
        return canonical

    candidates = sorted(
        [
            p
            for p in _TEST_ANALYSIS.iterdir()
            if p.is_dir() and p.name.startswith("brieflow_output")
        ],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for p in candidates:
        if (p / "preprocess" / "metadata").exists():
            return p

    pytest.skip(
        "Brieflow output directory not found. Run small_test_analysis/run_brieflow.sh first."
    )


class TestZarrStructural:
    """Integration tests verifying the structure of zarr outputs produced
    by the Snakemake pipeline (chunk layout, compression, multiscale)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.root = _resolve_output_dir()
        self.preprocess = self.root / "preprocess"

    def _sbs_zarr_path(self) -> Path:
        return (
            self.preprocess
            / "images"
            / "sbs"
            / get_filename(
                {
                    "plate": _TEST_PLATE,
                    "well": _TEST_WELL,
                    "tile": _TEST_TILE_SBS,
                    "cycle": _TEST_CYCLE,
                },
                "image",
                "zarr",
            )
        )

    @pytest.mark.integration
    def test_zarr_chunks_are_reasonable(self):
        """Spatial chunks are square-ish and between 256-2048 px;
        channel dim is unchunked."""
        zp = self._sbs_zarr_path()
        if not zp.exists():
            pytest.skip("Standard Zarr output not found.")

        arr = zarr.open(str(zp), mode="r")["0"]
        c_chunk, y_chunk, x_chunk = arr.chunks
        assert c_chunk == arr.shape[0], "Channel dim should not be chunked"
        assert 256 <= y_chunk <= 2048, f"Y chunk {y_chunk} out of range"
        assert 256 <= x_chunk <= 2048, f"X chunk {x_chunk} out of range"
        assert max(y_chunk, x_chunk) / min(y_chunk, x_chunk) <= 2.0

    @pytest.mark.integration
    def test_zarr_compression_applied(self):
        """On-disk size should not wildly exceed uncompressed size."""
        zp = self._sbs_zarr_path()
        if not zp.exists():
            pytest.skip("Standard Zarr output not found.")

        arr = zarr.open(str(zp), mode="r")["0"]
        uncompressed = np.prod(arr.shape) * arr.dtype.itemsize
        actual = sum(
            f.stat().st_size for f in (Path(zp) / "0").rglob("*") if f.is_file()
        )
        ratio = actual / uncompressed
        assert ratio < 5.0, f"Compression ratio {ratio:.2f} is unexpectedly high"

    @pytest.mark.integration
    def test_omezarr_multiscale_structure(self):
        """OME-Zarr export has valid multiscales metadata with >=2 levels."""
        omezarr_path = (
            self.preprocess
            / "omezarr"
            / "sbs"
            / get_filename(
                {
                    "plate": _TEST_PLATE,
                    "well": _TEST_WELL,
                    "tile": _TEST_TILE_SBS,
                    "cycle": _TEST_CYCLE,
                },
                "image",
                "zarr",
            )
        )
        if not omezarr_path.exists():
            pytest.skip("OME-Zarr export not found.")

        store = zarr.open(str(omezarr_path), mode="r")
        ms = store.attrs["multiscales"]
        assert isinstance(ms, list) and len(ms) > 0

        ms0 = ms[0]
        assert ms0["version"] == "0.4"
        assert "axes" in ms0
        datasets = ms0["datasets"]
        assert len(datasets) >= 2, "Expected >=2 resolution levels"

        for ds in datasets:
            assert ds["path"] in store

    @pytest.mark.integration
    def test_omezarr_matches_standard_zarr_at_level0(self):
        """OME-Zarr level-0 pixel data is identical to standard Zarr output."""
        std = self._sbs_zarr_path()
        ome = (
            self.preprocess
            / "omezarr"
            / "sbs"
            / get_filename(
                {
                    "plate": _TEST_PLATE,
                    "well": _TEST_WELL,
                    "tile": _TEST_TILE_SBS,
                    "cycle": _TEST_CYCLE,
                },
                "image",
                "zarr",
            )
        )
        if not std.exists() or not ome.exists():
            pytest.skip("Both standard Zarr and OME-Zarr needed.")

        std_data = zarr.open(str(std), mode="r")["0"][:]
        ome_data = zarr.open(str(ome), mode="r")["0"][:]
        np.testing.assert_array_equal(std_data, ome_data)

    @pytest.mark.integration
    def test_zarr_tiff_equivalence_sbs(self):
        """Standard Zarr and TIFF outputs have identical pixel data (SBS)."""
        base = self.preprocess / "images" / "sbs"
        name_base = get_filename(
            {
                "plate": _TEST_PLATE,
                "well": _TEST_WELL,
                "tile": _TEST_TILE_SBS,
                "cycle": _TEST_CYCLE,
            },
            "image",
            "zarr",
        ).rsplit(".", 1)[0]

        zarr_p = base / f"{name_base}.zarr"
        tiff_p = base / f"{name_base}.tiff"
        if not zarr_p.exists() or not tiff_p.exists():
            pytest.skip("Both formats needed for equivalence test.")

        tiff_data = tiff_imread(str(tiff_p))
        zarr_data = zarr.open(str(zarr_p), mode="r")["0"][:]
        assert tiff_data.shape == zarr_data.shape
        np.testing.assert_array_equal(tiff_data, zarr_data)


# ===========================================================================
# Section 7: target_utils — output_to_input() regression
# ===========================================================================


class TestTargetUtils:
    """Regression tests for output_to_input() which broke during zarr
    transition when given a single Path template instead of a list."""

    def test_single_path_template(self):
        """Single Path template with metadata expansion returns correct string."""
        from workflow.lib.shared.target_utils import output_to_input

        combos = pd.DataFrame([{"plate": "1", "well": "A1", "tile": "2"}])
        template = (
            Path("brieflow_output")
            / "sbs"
            / "tsvs"
            / get_filename(
                {"plate": "{plate}", "well": "{well}", "tile": "{tile}"},
                "segmentation_stats",
                "tsv",
            )
        )

        result = output_to_input(
            template,
            wildcards={"plate": "1"},
            expansion_values=["well", "tile"],
            metadata_combos=combos,
        )
        assert result == [
            "brieflow_output/sbs/tsvs/P-1_W-A1_T-2__segmentation_stats.tsv"
        ]

    def test_list_of_one_template(self):
        """List-of-one template with metadata expansion."""
        from workflow.lib.shared.target_utils import output_to_input

        combos = pd.DataFrame([{"plate": "1", "well": "A1", "tile": "2"}])
        template = [
            Path("brieflow_output")
            / "sbs"
            / "parquets"
            / get_filename({"plate": "{plate}", "well": "{well}"}, "cells", "parquet")
        ]

        result = output_to_input(
            template,
            wildcards={"plate": "1"},
            expansion_values=["well"],
            metadata_combos=combos,
        )
        assert result == ["brieflow_output/sbs/parquets/P-1_W-A1__cells.parquet"]


# ===========================================================================
# Section 8: ND2 metadata — pixel size extraction
# ===========================================================================


class TestND2Metadata:
    """Verify that extract_metadata_tile_nd2 captures pixel-size and optics
    fields needed for OME-Zarr coordinate transforms."""

    _nd2_path = (
        Path(__file__).resolve().parent
        / "small_test_analysis"
        / "small_test_data"
        / "phenotype"
        / "empty_images"
        / "P001_Pheno_20x_Wells-A1_Points-002__Channel_AF750,Cy3,GFP,DAPI.nd2"
    )

    def test_metadata_includes_pixel_size_and_optics(self):
        """extract_metadata_tile_nd2 returns z pixel size and optics columns."""
        nd2 = pytest.importorskip("nd2")  # noqa: F841
        from workflow.lib.preprocess.preprocess import extract_metadata_tile_nd2

        if not self._nd2_path.exists():
            pytest.skip("ND2 test data not found.")

        df = extract_metadata_tile_nd2(
            file_path=str(self._nd2_path),
            plate="1",
            well="A1",
            tile="2",
            verbose=False,
        )
        row = df.iloc[0]

        for col in [
            "pixel_size_z",
            "objective_magnification",
            "zoom_magnification",
            "binning_xy",
        ]:
            assert col in df.columns, f"Missing column: {col}"

        for col in ["pixel_size_x", "pixel_size_y", "pixel_size_z"]:
            assert row[col] is not None, f"{col} should not be None"

    def test_convert_to_array_preserve_z(self):
        """convert_to_array with preserve_z=True returns CZYX."""
        pytest.importorskip("nd2")
        from workflow.lib.preprocess.preprocess import convert_to_array

        if not self._nd2_path.exists():
            pytest.skip("ND2 test data not found.")

        arr = convert_to_array(
            files=str(self._nd2_path),
            data_format="nd2",
            data_organization="tile",
            preserve_z=True,
            verbose=False,
        )
        assert arr.ndim == 4, f"Expected CZYX (4D), got {arr.ndim}D"
