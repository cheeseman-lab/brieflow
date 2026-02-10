"""Unified IO module for reading and writing images in TIFF and OME-Zarr formats."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import dask.array as da
import numpy as np
import zarr
from ome_zarr.scale import Scaler
from ome_zarr.writer import write_image
from tifffile import imread as tiff_imread
from tifffile import imwrite as tiff_imwrite

# Zarr on-disk format version.
# 3 = Zarr v3 / OME-NGFF v0.5 (zarr.json metadata).
# 2 = Zarr v2 / OME-NGFF v0.4 (.zarray/.zgroup/.zattrs) for legacy compat.
ZARR_FORMAT = 3

PathLike = Union[str, Path]


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
        # Squeeze singleton leading dimension added by save_image() for OME-Zarr
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
    max_levels: int = 5,
    is_label: bool = False,
) -> None:
    """Save an image to TIFF or OME-Zarr depending on the output path suffix."""
    out = Path(output_path)
    suffix = out.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        out.parent.mkdir(parents=True, exist_ok=True)
        tiff_imwrite(str(out), image)
        return

    if suffix == ".zarr" or suffix.endswith(".ome.zarr") or out.name.endswith(".zarr"):
        axes: str
        data = image
        if image.ndim == 2:
            data = image[np.newaxis, ...]
            axes = "cyx"
        elif image.ndim == 3:
            axes = "cyx"
        elif image.ndim == 4:
            axes = "czyx"
        else:
            raise ValueError(f"Unsupported image.ndim={image.ndim} for OME-Zarr export")

        ch_names = list(channel_names) if channel_names is not None else None
        if ch_names is None and "c" in axes:
            c_len = int(data.shape[axes.index("c")])
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
    axes: str = "cyx",
    pixel_size_um: Optional[Union[float, Tuple[float, ...], Dict[str, float]]] = None,
    coarsening_factor: int = 2,
    max_levels: int = 4,
    is_label: bool = False,
    chunk_size: Optional[Tuple[int, ...]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
) -> None:
    """Write an image array to OME-Zarr format with pyramids.

    Args:
        image_data: Numpy or Dask array containing image data.
        out_path: Path to the output .zarr directory.
        channel_names: List of channel names. Length must match channel dimension.
        axes: String describing axes, e.g., "cyx", "tcyx".
        pixel_size_um: Pixel size in microns.
            - float: applied to X and Y
            - tuple: (y, x) or (z, y, x) depending on available axes
            - dict: keys from {"x","y","z"} (values can be None)
        coarsening_factor: Factor by which to downscale the image.
        max_levels: Maximum number of pyramid levels to generate.
        is_label: Whether the image is a label image.
        chunk_size: Tuple for chunking (optional).
        storage_options: Options for storage backend (optional).
    """
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
            if "y" in axes and "x" in axes:
                y_idx = axes.find("y")
                x_idx = axes.find("x")
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
        if "z" in axes and ps.get("z") is not None:
            scale_transform[axes.find("z")] = ps["z"]
        if "y" in axes and ps.get("y") is not None:
            scale_transform[axes.find("y")] = ps["y"] * (coarsening_factor**i)
        if "x" in axes and ps.get("x") is not None:
            scale_transform[axes.find("x")] = ps["x"] * (coarsening_factor**i)
        coordinate_transformations.append([{"scale": scale_transform, "type": "scale"}])

    metadata: Dict[str, Any] = {}
    omero: Dict[str, Any] = {}
    if channel_names:
        if is_label:
            omero["channels"] = [
                {"label": name, "active": True} for name in channel_names
            ]
        else:
            omero["channels"] = [
                {"label": name, "active": True, "color": "FFFFFF"}
                for name in channel_names
            ]
    if any(v is not None for v in ps.values()):
        omero["pixel_size"] = {k: v for k, v in ps.items() if v is not None}
    if omero:
        metadata["omero"] = omero
    if is_label:
        metadata["image-label"] = {}

    write_image(
        image=image_data,
        group=root,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        scaler=Scaler(
            method="nearest",
            downscale=coarsening_factor,
            max_layer=max_levels - 1,
            labeled=is_label,
        ),
        **metadata,
    )

    for k, v in metadata.items():
        root.attrs[k] = v


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
