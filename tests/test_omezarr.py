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
3. NGFF compliance ....... OME-NGFF v0.5 metadata validation
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
from workflow.lib.shared.image_io import read_image, save_image, write_image_omezarr

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
        assert "multiscales" in _ome_metadata(store)
        np.testing.assert_array_equal(data, store["0"][:])

        omero = _ome_metadata(store).get("omero")
        assert omero is not None
        assert len(omero["channels"]) == 3
        assert omero["channels"][0]["label"] == "r"


# ===========================================================================
# Section 3: NGFF compliance — OME-NGFF v0.5 metadata validation
# ===========================================================================


def _ome_metadata(store_or_path) -> dict:
    """OME-NGFF metadata block of a store, given the store or its path.

    NGFF 0.5 nests everything under an ``ome`` key; 0.4 held the same keys at the
    root. Reading through here keeps the assertions about content, not placement.
    """
    if isinstance(store_or_path, (str, Path)):
        attrs = dict(zarr.open_group(str(store_or_path), mode="r").attrs)
    else:
        attrs = dict(store_or_path.attrs)
    return attrs.get("ome", attrs)


class TestNGFFCompliance:
    """Validate that write_image_omezarr produces spec-compliant
    OME-NGFF v0.5 (Zarr v3) metadata."""

    def test_v05_metadata_structure(self, tmp_path):
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

        # Zarr v3 layout: one zarr.json per node, no .zgroup / .zattrs
        assert (out / "zarr.json").exists()
        assert not (out / ".zgroup").exists()
        assert not (out / ".zattrs").exists()

        ome = _ome_metadata(out)
        assert ome["version"] == "0.5"
        assert "multiscales" in ome
        assert len(ome["multiscales"]) == 1

        ms0 = ome["multiscales"][0]
        assert ms0["axes"] == [
            {"name": "c", "type": "channel"},
            {"name": "y", "type": "space", "unit": "micrometer"},
            {"name": "x", "type": "space", "unit": "micrometer"},
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

        scale0 = _ome_metadata(out)["multiscales"][0]["datasets"][0][
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

        scale0 = _ome_metadata(out)["multiscales"][0]["datasets"][0][
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

        datasets = _ome_metadata(out)["multiscales"][0]["datasets"]
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
        """Zarr write → read preserves data and metadata.

        save_image promotes to the pipeline's TCZYX layout, so (C, Y, X) is stored
        as (1, C, 1, Y, X).
        """
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
        assert (zp / "zarr.json").exists()
        stored = zarr.open(str(zp), mode="r")["0"][:]
        assert stored.shape == (1, 3, 1) + dummy_3d_uint16.shape[1:]
        np.testing.assert_array_equal(
            dummy_3d_uint16[np.newaxis, :, np.newaxis, ...], stored
        )

        ome = _ome_metadata(zp)
        assert ome["version"] == "0.5"
        assert ome["omero"]["channels"][0]["label"] == "Ch1"

    def test_save_omezarr_2d_gets_singleton_tczyx(self, tmp_path, dummy_2d_uint16):
        """2D array saved as zarr is expanded to (1, 1, 1, Y, X)."""
        zp = tmp_path / "test.zarr"
        save_image(dummy_2d_uint16, zp)

        stored = zarr.open(str(zp), mode="r")["0"][:]
        assert stored.shape == (1, 1, 1) + dummy_2d_uint16.shape
        np.testing.assert_array_equal(
            dummy_2d_uint16[np.newaxis, np.newaxis, np.newaxis, ...], stored
        )

    def test_save_omezarr_label_flag(self, tmp_path, dummy_3d_uint16):
        """is_label=True produces image-label metadata without channel colors."""
        zp = tmp_path / "labels.zarr"
        label_img = (dummy_3d_uint16 > 1000).astype(np.uint16)
        save_image(label_img, zp, is_label=True)

        ome = _ome_metadata(zp)
        assert "image-label" in ome
        assert "color" not in ome["omero"]["channels"][0]

    def test_read_omezarr_multiscale(self, tmp_path, dummy_3d_uint16):
        """read_image returns full-resolution level from a multiscale store."""
        zp = tmp_path / "ms.zarr"
        write_image_omezarr(
            image_data=dummy_3d_uint16,
            out_path=str(zp),
            axes="cyx",
            coarsening_factor=2,
            max_levels=2,
            pixel_size_um=(0.5, 0.5),
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


# ===========================================================================
# Section 6: Zarr structural — chunks, compression, multiscale from pipeline
# ===========================================================================

# These integration tests read artifacts from a prior Snakemake run.
# They skip gracefully if the output directory is not present.

_TEST_ANALYSIS = Path(__file__).resolve().parent / "small_test_analysis"


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
        "Brieflow output directory not found. Run small_test_analysis/run_brieflow.sh --zarr first."
    )


def _find_multiscale_groups(root: Path, limit: int = 1) -> list:
    """Locate OME-NGFF image groups under a pipeline output directory.

    Resolves stores by discovery instead of reconstructing a filename, because
    the on-disk layout differs by output format: zarr mode writes one HCS plate
    store per plate (``preprocess/sbs/image_1.zarr/A/1/0``) while tiff mode
    writes per-tile files. A hardcoded path silently skips in both.

    Args:
        root: Directory to search (typically ``<output>/preprocess``).
        limit: Stop after this many image groups.

    Returns:
        List of directories whose metadata declares ``ome.multiscales``.
    """
    found = []
    for meta_fp in sorted(root.rglob("zarr.json")):
        try:
            meta = json.loads(meta_fp.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if "multiscales" in meta.get("attributes", {}).get("ome", {}):
            found.append(meta_fp.parent)
            if len(found) >= limit:
                break
    return found


class TestZarrStructural:
    """Integration tests verifying the structure of zarr outputs produced
    by the Snakemake pipeline (chunk layout, compression, multiscale)."""

    @pytest.fixture(autouse=True)
    def _setup(self):
        self.root = _resolve_output_dir()
        self.preprocess = self.root / "preprocess"

    def _image_group(self):
        """Return the first pipeline-written OME-Zarr image group, or skip."""
        groups = _find_multiscale_groups(self.preprocess)
        if not groups:
            pytest.skip(
                "No OME-Zarr image group under "
                f"{self.preprocess}. Run run_brieflow.sh --zarr first."
            )
        return zarr.open_group(str(groups[0]), mode="r")

    @staticmethod
    def _levels(group) -> list:
        """Dataset paths declared by the group's multiscales metadata."""
        return [d["path"] for d in group.attrs["ome"]["multiscales"][0]["datasets"]]

    @pytest.mark.integration
    def test_zarr_chunks_are_reasonable(self):
        """Spatial chunks are square-ish and 256-2048 px; C and Z unchunked."""
        arr = self._image_group()["0"]
        chunks = dict(zip("tczyx"[-arr.ndim :], arr.chunks))
        shape = dict(zip("tczyx"[-arr.ndim :], arr.shape))
        for axis in ("c", "z"):
            if axis in chunks:
                assert chunks[axis] == shape[axis], f"{axis} dim should not be chunked"
        y_chunk, x_chunk = chunks["y"], chunks["x"]
        assert 256 <= y_chunk <= 2048, f"Y chunk {y_chunk} out of range"
        assert 256 <= x_chunk <= 2048, f"X chunk {x_chunk} out of range"
        assert max(y_chunk, x_chunk) / min(y_chunk, x_chunk) <= 2.0

    @pytest.mark.integration
    def test_zarr_compression_codec_is_configured(self):
        """Every level carries a real Blosc codec, named, not inferred from size.

        The previous size-ratio assertion could not fail: uncompressed 16-bit
        data lands near 1.0, well under the 5.0 threshold it checked.
        """
        group = self._image_group()
        for level in self._levels(group):
            codecs = [c["name"] for c in group[level].metadata.to_dict()["codecs"]]
            assert "blosc" in codecs, (
                f"level {level} has codecs {codecs}; expected Blosc from "
                "all.zarr_compression. zstd alone means the compressor was dropped."
            )

    @pytest.mark.integration
    def test_omezarr_multiscale_structure(self):
        """Pipeline output declares a real pyramid and every level exists."""
        group = self._image_group()
        ome = group.attrs["ome"]
        assert ome["version"] == "0.5"
        ms0 = ome["multiscales"][0]
        assert "axes" in ms0

        levels = self._levels(group)
        assert len(levels) >= 2, (
            f"Expected >=2 resolution levels, got {levels}. "
            "all.zarr_max_levels may not be reaching save_image."
        )
        for level in levels:
            assert level in group

    @pytest.mark.integration
    def test_level0_full_resolution_and_channels_preserved(self):
        """Level 0 is not downsampled and C is constant across the pyramid.

        Guards the ome_zarr.writer.write_image numpy-path behaviour, which
        scales the channel axis alongside Y/X (3 -> 2 -> 1) and smooths
        level 0 rather than writing it verbatim.
        """
        group = self._image_group()
        levels = self._levels(group)
        shapes = [group[level].shape for level in levels]
        axes = "tczyx"[-len(shapes[0]) :]
        c_idx, y_idx, x_idx = axes.index("c"), axes.index("y"), axes.index("x")

        for i, shape in enumerate(shapes[1:], start=1):
            assert shape[c_idx] == shapes[0][c_idx], (
                f"level {i} has {shape[c_idx]} channels vs {shapes[0][c_idx]} at "
                "level 0; the channel axis is being downsampled"
            )
            for idx in (y_idx, x_idx):
                assert shape[idx] == shapes[0][idx] // (2**i), (
                    f"level {i} axis {axes[idx]} is {shape[idx]}, "
                    f"expected {shapes[0][idx] // (2**i)}"
                )

    @pytest.mark.integration
    def test_zarr_tiff_equivalence_sbs(self):
        """Zarr level 0 and the TIFF output hold identical pixel data (SBS)."""
        tiff_dir = self.preprocess / "images" / "sbs"
        tiffs = sorted(tiff_dir.glob("*__image.tiff")) if tiff_dir.exists() else []
        if not tiffs:
            pytest.skip("No TIFF SBS output; run without --zarr to compare formats.")

        group = self._image_group()
        zarr_data = np.squeeze(group["0"][:])
        tiff_data = np.squeeze(tiff_imread(str(tiffs[0])))
        assert tiff_data.shape == zarr_data.shape
        np.testing.assert_array_equal(tiff_data, zarr_data)


# ===========================================================================
# Section 6b: Compression and pyramid — unit coverage for the writer
# ===========================================================================


class TestCompressionAndPyramid:
    """Direct coverage of the codec parameter and explicit pyramid loop."""

    @staticmethod
    def _codecs(arr):
        return arr.metadata.to_dict()["codecs"]

    def test_pyramid_levels_and_halving(self, tmp_path):
        """Each level halves Y/X; the channel axis is never scaled."""
        out = tmp_path / "pyr.zarr"
        img = np.random.randint(0, 4000, (2, 256, 256), dtype=np.uint16)
        write_image_omezarr(
            img, str(out), axes="cyx", channel_names=["a", "b"], max_levels=4
        )
        root = zarr.open_group(str(out), mode="r")
        datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
        assert [d["path"] for d in datasets] == ["0", "1", "2", "3"]
        for i in range(4):
            assert root[str(i)].shape == (2, 256 // 2**i, 256 // 2**i)

    def test_level0_bit_identical_to_single_level(self, tmp_path):
        """Adding levels never rewrites level 0."""
        img = np.random.randint(0, 4000, (2, 200, 200), dtype=np.uint16)
        single, pyr = tmp_path / "one.zarr", tmp_path / "many.zarr"
        write_image_omezarr(img, str(single), axes="cyx", max_levels=1)
        write_image_omezarr(img, str(pyr), axes="cyx", max_levels=5)
        s0 = zarr.open_group(str(single), mode="r")["0"][:]
        p0 = zarr.open_group(str(pyr), mode="r")["0"][:]
        np.testing.assert_array_equal(s0, p0)
        np.testing.assert_array_equal(img, p0)

    def test_labels_use_nearest_neighbour(self, tmp_path):
        """Label pyramids never invent values between existing labels."""
        out = tmp_path / "lab.zarr"
        lab = np.random.randint(0, 6, (1, 256, 256)).astype(np.uint16)
        write_image_omezarr(lab, str(out), axes="cyx", is_label=True, max_levels=4)
        root = zarr.open_group(str(out), mode="r")
        base = set(np.unique(root["0"][:]).tolist())
        for i in range(1, 4):
            assert set(np.unique(root[str(i)][:]).tolist()).issubset(base)
        assert root.attrs["ome"]["multiscales"][0]["downsamplingMethod"] == "nearest"

    def test_compression_applies_to_every_level(self, tmp_path):
        """The requested codec reaches every level, not just level 0."""
        out = tmp_path / "comp.zarr"
        img = np.random.randint(0, 4000, (2, 128, 128), dtype=np.uint16)
        write_image_omezarr(
            img,
            str(out),
            axes="cyx",
            max_levels=3,
            compression="blosc-zstd-bitshuffle",
        )
        root = zarr.open_group(str(out), mode="r")
        for i in range(3):
            blosc = [c for c in self._codecs(root[str(i)]) if c["name"] == "blosc"]
            assert blosc, f"level {i} is not Blosc-compressed"
            assert blosc[0]["configuration"]["cname"] == "zstd"
            assert blosc[0]["configuration"]["shuffle"] == "bitshuffle"

    @pytest.mark.parametrize(
        "spec,cname,shuffle,clevel",
        [
            ("blosc-zstd-bitshuffle", "zstd", "bitshuffle", 5),
            ("blosc-lz4-shuffle", "lz4", "shuffle", 5),
            ("blosc-zstd", "zstd", "bitshuffle", 5),
            ("blosc-blosclz-noshuffle", "blosclz", "noshuffle", 5),
            ("blosc-zstd-bitshuffle:9", "zstd", "bitshuffle", 9),
        ],
    )
    def test_compression_spec_varies_codec(
        self, tmp_path, spec, cname, shuffle, clevel
    ):
        """Codec, shuffle and level are all selectable from the config string."""
        out = tmp_path / f"{spec.replace(':', '_')}.zarr"
        img = np.random.randint(0, 4000, (1, 64, 64), dtype=np.uint16)
        write_image_omezarr(img, str(out), axes="cyx", max_levels=1, compression=spec)
        arr = zarr.open_group(str(out), mode="r")["0"]
        cfg = [c for c in self._codecs(arr) if c["name"] == "blosc"][0]["configuration"]
        assert (cfg["cname"], cfg["shuffle"], cfg["clevel"]) == (cname, shuffle, clevel)
        np.testing.assert_array_equal(img, arr[:])

    @pytest.mark.parametrize("spec", ["gzip", "blosc", "blosc-zstd-sideways", "lz4"])
    def test_unknown_compression_spec_raises(self, tmp_path, spec):
        """A malformed spec fails loudly instead of silently writing raw."""
        img = np.random.randint(0, 4000, (1, 32, 32), dtype=np.uint16)
        with pytest.raises(ValueError):
            write_image_omezarr(
                img, str(tmp_path / "bad.zarr"), axes="cyx", compression=spec
            )

    def test_writer_defaults_are_conservative(self, tmp_path):
        """Writer default stays single-level and uncompressed.

        Callers opt in, so the other save_image callers keep prior behaviour.
        """
        out = tmp_path / "default.zarr"
        img = np.random.randint(0, 4000, (1, 64, 64), dtype=np.uint16)
        write_image_omezarr(img, str(out), axes="cyx")
        root = zarr.open_group(str(out), mode="r")
        assert not any(c["name"] == "blosc" for c in self._codecs(root["0"]))
        datasets = root.attrs["ome"]["multiscales"][0]["datasets"]
        assert [d["path"] for d in datasets] == ["0"]

    def test_compression_can_be_disabled(self, tmp_path):
        """compression='none' falls back to zarr's default codec."""
        out = tmp_path / "raw.zarr"
        img = np.random.randint(0, 4000, (1, 64, 64), dtype=np.uint16)
        write_image_omezarr(img, str(out), axes="cyx", max_levels=1, compression="none")
        assert not any(
            c["name"] == "blosc"
            for c in zarr.open_group(str(out), mode="r")["0"].metadata.to_dict()[
                "codecs"
            ]
        )


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
