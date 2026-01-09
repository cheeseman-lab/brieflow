"""
Lightweight IO helpers used by unit tests.

These utilities provide a small, stable API around TIFF and OME-Zarr reading/writing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np
from tifffile import imread as tiff_imread
from tifffile import imwrite as tiff_imwrite

from workflow.lib.io.omezarr_writer import write_image_omezarr


PathLike = Union[str, Path]


def read_image(path: PathLike) -> np.ndarray:
    """
    Read an image from TIFF or OME-Zarr.

    For OME-Zarr, returns the highest-resolution (level 0) array.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)

    if p.suffix.lower() in {".tif", ".tiff"}:
        return tiff_imread(str(p))

    if p.suffix.lower() == ".zarr" or p.is_dir():
        try:
            import zarr  # local import to support tests that simulate missing dependency
        except Exception as e:
            raise ImportError("zarr package is required") from e

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
        return root[ds_path][:]

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
    """
    Save an image to TIFF or OME-Zarr depending on the output path suffix.
    """
    out = Path(output_path)
    suffix = out.suffix.lower()

    if suffix in {".tif", ".tiff"}:
        out.parent.mkdir(parents=True, exist_ok=True)
        tiff_imwrite(str(out), image)
        return

    if suffix == ".zarr" or suffix.endswith(".ome.zarr") or out.name.endswith(".zarr"):
        # Normalize 2D to (1, Y, X) for OME-Zarr writing
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


