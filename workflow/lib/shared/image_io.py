"""Unified IO module for reading and writing images in TIFF and OME-Zarr formats."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import numcodecs
import zarr
from ome_zarr.format import CurrentFormat
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_multiscales_metadata
from tifffile import imread as tiff_imread
from tifffile import imwrite as tiff_imwrite
from zarr.codecs import BloscCodec, BloscShuffle

# Zarr on-disk format version.
# 3 = Zarr v3 / OME-NGFF v0.5 (zarr.json metadata).
# 2 = Zarr v2 / OME-NGFF v0.4 (.zarray/.zgroup/.zattrs) for legacy compat.
ZARR_FORMAT = 3

# Default channel color palette for OME-Zarr OMERO rendering metadata.
DEFAULT_CHANNEL_COLORS = [
    "0000FF",  # blue
    "00FF00",  # green
    "FF0000",  # red
    "FF00FF",  # magenta
    "FFFF00",  # yellow
    "00FFFF",  # cyan
    "FF8000",  # orange
    "8000FF",  # purple
]

PathLike = Union[str, Path]

# Pyramid depth and compression applied by the *preprocess convert* path when
# config.yml omits all.zarr_max_levels / all.zarr_compression.
# These are call-site defaults, NOT writer defaults: save_image and
# write_image_omezarr stay single-level and uncompressed unless asked, so every
# other save_image caller keeps its existing behavior.
DEFAULT_MAX_LEVELS = 5
DEFAULT_ZARR_COMPRESSION = "blosc-zstd-bitshuffle"
DEFAULT_BLOSC_CLEVEL = 5

# Cap the process-wide c-blosc thread pool so a tile-conversion job doesn't
# spawn one thread per core on top of Snakemake's own parallelism.
_BLOSC_NTHREADS = int(os.environ.get("BRIEFLOW_BLOSC_NTHREADS", "4"))
numcodecs.blosc.set_nthreads(_BLOSC_NTHREADS)

_BLOSC_SHUFFLE = {
    "noshuffle": BloscShuffle.noshuffle,
    "shuffle": BloscShuffle.shuffle,
    "bitshuffle": BloscShuffle.bitshuffle,
}


def _make_compressor(compression: Optional[str]):
    """Build a zarr-v3 Blosc codec from a ``blosc-<cname>[-<shuffle>][:<clevel>]`` string.

    Any cname and shuffle c-blosc supports can be selected from config without
    editing this module, e.g. ``blosc-zstd-bitshuffle`` (the preprocess default),
    ``blosc-lz4-shuffle`` for speed over ratio, or ``blosc-zstd-bitshuffle:9`` to
    override the compression level. ``shuffle`` defaults to ``bitshuffle`` and
    ``clevel`` to ``DEFAULT_BLOSC_CLEVEL``.

    Returns None, meaning zarr's default codec, for ``None`` or ``"none"``.
    Blosc is lossless for every cname, so a write/read roundtrip is always
    bit-identical.

    Args:
        compression: Codec spec string, or None/"none" to leave zarr's default.

    Returns:
        A configured BloscCodec, or None to use zarr's default codec.

    Raises:
        ValueError: If the spec is malformed or names an unknown shuffle.
    """
    if compression is None:
        return None
    key = str(compression).strip().lower()
    if key in ("", "none"):
        return None

    spec, _, clevel = key.partition(":")
    parts = spec.split("-")
    if parts[0] != "blosc" or not 2 <= len(parts) <= 3:
        raise ValueError(
            f"Unknown zarr compression {compression!r}; expected 'none' or "
            "'blosc-<cname>[-<shuffle>][:<clevel>]'"
        )
    shuffle = parts[2] if len(parts) == 3 else "bitshuffle"
    if shuffle not in _BLOSC_SHUFFLE:
        raise ValueError(
            f"Unknown blosc shuffle {shuffle!r} in {compression!r}; "
            f"expected one of {sorted(_BLOSC_SHUFFLE)}"
        )
    # BloscCodec rejects an unknown cname itself, so no cname table to maintain.
    return BloscCodec(
        cname=parts[1],
        clevel=int(clevel) if clevel else DEFAULT_BLOSC_CLEVEL,
        shuffle=_BLOSC_SHUFFLE[shuffle],
    )


# ---------------------------------------------------------------------------
# High-level API (used by Snakemake scripts)
# ---------------------------------------------------------------------------


def read_image(path: PathLike) -> np.ndarray:
    """Read an image from TIFF or OME-Zarr.

    For OME-Zarr, returns the highest-resolution (level 0) array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    # zarr.json sentinel path (HCS): open the parent directory as zarr store
    if p.name == "zarr.json":
        p = p.parent

    if p.suffix.lower() in {".tif", ".tiff"}:
        return tiff_imread(str(p))

    if p.suffix.lower() == ".zarr" or p.is_dir():
        root = zarr.open_group(str(p), mode="r")
        ds_path: Optional[str] = None
        ms = root.attrs.get("multiscales")
        if isinstance(ms, list) and ms:
            datasets = ms[0].get("datasets", [])
            if datasets and isinstance(datasets, list):
                ds_path = datasets[0].get("path")
        if ds_path is None and "0" in root:
            ds_path = "0"
        if ds_path is None:
            raise ValueError("Could not find image data in OME-Zarr")
        arr = root[ds_path][:]
        # Squeeze singleton dimensions added for 5D TCZYX OME-Zarr storage.
        # 5D TCZYX: squeeze T (always singleton), squeeze Z if singleton.
        if arr.ndim == 5:
            arr = arr[0]  # squeeze T → CZYX
            if arr.shape[1] == 1:  # Z is singleton
                arr = arr[:, 0, :, :]  # → CYX
        elif arr.ndim == 4:
            # Legacy CZYX: squeeze Z if singleton
            if arr.shape[1] == 1:
                arr = arr[:, 0, :, :]
        # Squeeze singleton channel dim (from 2D input expanded to CYX)
        if arr.ndim > 2 and arr.shape[0] == 1:
            arr = arr[0]
        return arr

    raise ValueError(f"Unsupported image path: {p}")


def save_image(
    image: np.ndarray,
    output_path: PathLike,
    *,
    pixel_size: Optional[Union[float, Tuple[float, ...]]] = None,
    channel_names: Optional[Sequence[str]] = None,
    coarsening_factor: int = 2,
    max_levels: int = 1,
    is_label: bool = False,
    compression: Optional[str] = None,
) -> None:
    """Save an image to TIFF or OME-Zarr depending on the output path suffix.

    Pyramids and compression are opt-in: pass ``max_levels`` / ``compression``
    (see :func:`_make_compressor` for the accepted strings). The preprocess
    convert path threads ``all.zarr_max_levels`` / ``all.zarr_compression`` in.
    """
    out = Path(output_path)
    suffix = out.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        out.parent.mkdir(parents=True, exist_ok=True)
        tiff_imwrite(str(out), image)
        return

    # zarr.json sentinel path (HCS): write zarr store to parent directory.
    # zarr v3 automatically creates zarr.json in the store root, satisfying Snakemake.
    if out.name == "zarr.json":
        out = out.parent

    # Detect zarr: explicit .zarr suffix, or any ancestor directory ends in .zarr
    # (handles HCS paths like aligned_1.zarr/A/1/0 where the terminal
    # component is an integer tile/cycle directory with no extension)
    _is_zarr = (
        suffix == ".zarr"
        or out.name.endswith(".zarr")
        or any(part.endswith(".zarr") for part in out.parts)
    )
    if _is_zarr:
        axes: str
        data = image
        if image.ndim == 2:
            # (Y, X) → (1, 1, 1, Y, X)  TCZYX
            data = image[np.newaxis, np.newaxis, np.newaxis, ...]
            axes = "TCZYX"
        elif image.ndim == 3:
            # (C, Y, X) → (1, C, 1, Y, X)  TCZYX
            data = image[np.newaxis, :, np.newaxis, ...]
            axes = "TCZYX"
        elif image.ndim == 4:
            # (C, Z, Y, X) → (1, C, Z, Y, X)  TCZYX
            data = image[np.newaxis, ...]
            axes = "TCZYX"
        else:
            raise ValueError(f"Unsupported image.ndim={image.ndim} for OME-Zarr export")

        ch_names = list(channel_names) if channel_names is not None else None
        if ch_names is None and "C" in axes:
            c_len = int(data.shape[axes.index("C")])
            ch_names = [f"c{i}" for i in range(c_len)]

        write_image_omezarr(
            image_data=data,
            out_path=str(out),
            channel_names=ch_names,
            axes=axes,
            pixel_size_um=pixel_size,
            coarsening_factor=coarsening_factor,
            max_levels=max_levels,
            is_label=is_label,
            compression=compression,
        )
        return

    raise ValueError(f"Unsupported output path: {out}")


# ---------------------------------------------------------------------------
# OME-Zarr writing
# ---------------------------------------------------------------------------


def write_image_omezarr(
    image_data: Union[np.ndarray, da.Array],
    out_path: str,
    channel_names: Optional[List[str]] = None,
    axes: str = "TCZYX",
    pixel_size_um: Optional[Union[float, Tuple[float, ...], Dict[str, float]]] = None,
    coarsening_factor: int = 2,
    max_levels: int = 1,
    is_label: bool = False,
    chunk_size: Optional[Tuple[int, ...]] = None,
    compression: Optional[str] = None,
) -> None:
    """Write an image array to OME-Zarr format with pyramids.

    Args:
        image_data: Numpy or Dask array containing image data.
        out_path: Path to the output .zarr directory.
        channel_names: List of channel names. Length must match channel dimension.
        axes: String describing axes, e.g., "CYX", "TCZYX". Normalized to uppercase.
        pixel_size_um: Pixel size in microns.
            - float: applied to X and Y
            - tuple: (y, x) or (z, y, x) depending on available axes
            - dict: keys from {"x","y","z"} (values can be None)
        coarsening_factor: Factor by which to downscale the image.
        max_levels: Number of pyramid levels to generate (1 = no downsampling).
        is_label: Whether the image is a label image.
        chunk_size: Tuple for chunking (optional).
        compression: Codec spec, ``blosc-<cname>[-<shuffle>][:<clevel>]``, or
            ``"none"``/None for zarr's default. See :func:`_make_compressor`.
    """
    # Normalize axis names to uppercase (OPS schema convention).
    axes = axes.upper()

    os.makedirs(out_path, exist_ok=True)
    root = zarr.open_group(out_path, mode="w", zarr_format=ZARR_FORMAT)

    if not isinstance(image_data, da.Array):
        if len(axes) != len(image_data.shape):
            raise ValueError(
                f"Axes '{axes}' (len={len(axes)}) does not match image_data.ndim={len(image_data.shape)} "
                f"with shape={image_data.shape}."
            )
        if chunk_size is None:
            shape = image_data.shape
            chunks = list(shape)
            if "Y" in axes and "X" in axes:
                y_idx = axes.find("Y")
                x_idx = axes.find("X")
                chunks[y_idx] = min(shape[y_idx], 1024)
                chunks[x_idx] = min(shape[x_idx], 1024)
            chunk_size = tuple(chunks)

        image_data = da.from_array(image_data, chunks=chunk_size)

    if max_levels < 1:
        raise ValueError(f"max_levels must be >= 1, got {max_levels}")
    if coarsening_factor < 2:
        raise ValueError(f"coarsening_factor must be >= 2, got {coarsening_factor}")

    ps = _parse_pixel_sizes(pixel_size_um)

    coordinate_transformations = []
    for i in range(max_levels):
        scale_transform = [1.0] * len(axes)
        if "Z" in axes and ps.get("z") is not None:
            scale_transform[axes.find("Z")] = ps["z"]
        if "Y" in axes and ps.get("y") is not None:
            scale_transform[axes.find("Y")] = ps["y"] * (coarsening_factor**i)
        if "X" in axes and ps.get("x") is not None:
            scale_transform[axes.find("X")] = ps["x"] * (coarsening_factor**i)
        coordinate_transformations.append([{"scale": scale_transform, "type": "scale"}])

    metadata: Dict[str, Any] = {}
    omero: Dict[str, Any] = {}

    dtype_max = (
        float(np.iinfo(image_data.dtype).max)
        if np.issubdtype(image_data.dtype, np.integer)
        else 1.0
    )
    default_window = {"start": 0.0, "end": dtype_max, "min": 0.0, "max": dtype_max}

    if channel_names:
        if is_label:
            omero["channels"] = [
                {"label": name, "active": True} for name in channel_names
            ]
        else:
            omero["channels"] = [
                {
                    "label": name,
                    "active": True,
                    "color": DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)],
                    "window": dict(default_window),
                }
                for i, name in enumerate(channel_names)
            ]
    if any(v is not None for v in ps.values()):
        omero["pixel_size"] = {k: v for k, v in ps.items() if v is not None}
    if omero:
        metadata["omero"] = omero
    if is_label:
        metadata["image-label"] = {"version": "0.5"}

    # ome_zarr only recognises lowercase axis names for type inference;
    # pass explicit dicts so uppercase names work without validation errors.
    axes_dicts = _axes_str_to_dicts(axes)
    dimension_names = [a["name"] for a in axes_dicts]

    compressor = _make_compressor(compression)
    if chunk_size is None:
        chunk_size = tuple(c[0] for c in image_data.chunks)

    scaler = Scaler(
        method="nearest" if is_label else "gaussian",
        downscale=coarsening_factor,
        max_layer=max_levels - 1,
        labeled=is_label,
    )

    # Build and write the pyramid here rather than calling
    # ome_zarr.writer.write_image, which is not usable on the zarr-v3 stack this
    # repo pins (zarr 3.x, ome-zarr 0.13.0):
    #
    #   * its dask branch silently drops the compressor, so every array lands
    #     with zarr's default codec (zstd level 0) no matter what is requested;
    #   * its numpy branch treats the channel axis as spatial, downsampling
    #     C alongside Y/X (3 -> 2 -> 1) and applying the gaussian to level 0,
    #     so level 0 is no longer the input.
    #
    # Writing each level with da.to_zarr lets us attach a real codec and keeps
    # level 0 bit-identical to the single-level output.
    zarr_array_kwargs: Dict[str, Any] = {"dimension_names": dimension_names}
    if compressor is not None:
        zarr_array_kwargs["compressors"] = [compressor]

    level = image_data
    datasets = []
    for i in range(max_levels):
        if i > 0:
            level = scaler.resize_image(level)
        level_chunks = tuple(min(c, s) for c, s in zip(chunk_size, level.shape))
        da.to_zarr(
            arr=level.rechunk(level_chunks),
            url=root.store,
            component=str(Path(root.path, str(i))),
            compute=True,
            zarr_array_kwargs=zarr_array_kwargs,
        )
        datasets.append(
            {
                "path": str(i),
                "coordinateTransformations": coordinate_transformations[i],
            }
        )

    write_multiscales_metadata(root, datasets, CurrentFormat(), axes_dicts, **metadata)

    # Merge our metadata (omero, image-label) into the ``ome`` namespace
    # that ome_zarr.writer already created for multiscales.  This ensures
    # iohub (and any OME-NGFF v0.5 reader) finds them at
    # attributes.ome.omero / attributes.ome.image-label.
    ome_attrs = dict(root.attrs.get("ome", {}))
    if metadata:
        for k, v in metadata.items():
            ome_attrs[k] = v

    # Record downsamplingMethod on multiscales (OPS schema RECOMMENDED).
    ms = ome_attrs.get("multiscales", [])
    if ms:
        ms[0]["downsamplingMethod"] = "nearest" if is_label else "gaussian"

    root.attrs["ome"] = ome_attrs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Maps uppercase axis character → OME-NGFF axis type.
_AXIS_TYPES: Dict[str, str] = {
    "T": "time",
    "C": "channel",
    "Z": "space",
    "Y": "space",
    "X": "space",
}


def _axes_str_to_dicts(axes: str) -> List[Dict[str, str]]:
    """Convert an axes string to a list of axis dicts for ome_zarr.

    OME-NGFF v0.5 requires lowercase axis names. The internal ``axes``
    parameter may be uppercase for readability; this function lowercases
    the stored ``name`` so that ngio and other spec-compliant readers
    can parse the metadata without an AxesSetup override.
    """
    _AXIS_UNITS: Dict[str, str] = {
        "T": "second",
        "Z": "micrometer",
        "Y": "micrometer",
        "X": "micrometer",
    }
    result = []
    for ch in axes.upper():
        d: Dict[str, str] = {"name": ch.lower()}
        if ch in _AXIS_TYPES:
            d["type"] = _AXIS_TYPES[ch]
        if ch in _AXIS_UNITS:
            d["unit"] = _AXIS_UNITS[ch]
        result.append(d)
    return result


def _parse_pixel_sizes(
    ps: Optional[Union[float, Tuple[float, ...], Dict[str, float]]],
) -> Dict[str, Optional[float]]:
    """Parse pixel size specification into a dict of {x, y, z} floats."""
    if ps is None:
        return {"x": None, "y": None, "z": None}
    if isinstance(ps, (int, float)):
        v = float(ps)
        return {"x": v, "y": v, "z": None}
    if isinstance(ps, dict):
        return {
            "x": float(ps["x"]) if ps.get("x") is not None else None,
            "y": float(ps["y"]) if ps.get("y") is not None else None,
            "z": float(ps["z"]) if ps.get("z") is not None else None,
        }
    if isinstance(ps, tuple):
        if len(ps) == 2:
            y, x = ps
            return {
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "z": None,
            }
        if len(ps) == 3:
            z, y, x = ps
            return {
                "x": float(x) if x is not None else None,
                "y": float(y) if y is not None else None,
                "z": float(z) if z is not None else None,
            }
    raise ValueError(
        f"Unsupported pixel_size_um type: {type(ps)}. Expected float, tuple, or dict."
    )
