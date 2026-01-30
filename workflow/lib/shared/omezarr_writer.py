"""OME-Zarr Writer Module for Brieflow.

This module provides functions to export images, labels, and tables to OME-Zarr format.
It uses the ome-zarr-py library for NGFF compliance.

Zarr format version is configurable via ``ZARR_FORMAT``:
  - ``2``: Zarr v2 on-disk format (.zarray/.zgroup/.zattrs), OME-NGFF v0.4 (default)
  - ``3``: Zarr v3 on-disk format (zarr.json), OME-NGFF v0.5
    Requires ome-zarr-py with v0.5 writing support (unreleased as of Jan 2026;
    available on ome-zarr-py main branch after PR #413).
"""

import os
import zarr
import numpy as np
import pandas as pd
from typing import List, Optional, Union, Dict, Any, Tuple
from ome_zarr.io import parse_url
from ome_zarr.writer import write_image, write_labels
from ome_zarr.scale import Scaler
import dask.array as da

# Default Zarr on-disk format version.
# Set to 2 for broad compatibility (OME-NGFF v0.4).
# Switch to 3 for Zarr v3 / OME-NGFF v0.5 when ome-zarr-py releases full v0.5 writing.
ZARR_FORMAT = 2


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

    # Ensure dask array for efficient scaling
    if not isinstance(image_data, da.Array):
        # Validate that axes string matches the image dimensionality.
        # This prevents confusing IndexError during chunk heuristic below.
        if len(axes) != len(image_data.shape):
            raise ValueError(
                f"Axes '{axes}' (len={len(axes)}) does not match image_data.ndim={len(image_data.shape)} "
                f"with shape={image_data.shape}. If exporting with Z, ensure the input array includes Z "
                f"(e.g. preserve_z=True)."
            )
        # Determine chunking if not provided
        if chunk_size is None:
            # Simple heuristic: keep C/T small, Y/X around 1024
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

    # Coordinate transformations
    coordinate_transformations = []

    def _parse_pixel_sizes(ps) -> Dict[str, Optional[float]]:
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

    ps = _parse_pixel_sizes(pixel_size_um)

    for i in range(max_levels):
        scale_transform = [1.0] * len(axes)
        # Apply per-axis scales for spatial axes.
        # ome-zarr-py downscales in X/Y only (not Z), so Z scale stays constant.
        if "z" in axes and ps.get("z") is not None:
            scale_transform[axes.find("z")] = ps["z"]
        if "y" in axes and ps.get("y") is not None:
            scale_transform[axes.find("y")] = ps["y"] * (coarsening_factor**i)
        if "x" in axes and ps.get("x") is not None:
            scale_transform[axes.find("x")] = ps["x"] * (coarsening_factor**i)
        coordinate_transformations.append([{"scale": scale_transform, "type": "scale"}])
    # Metadata
    metadata = {}
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
        # Store base pixel sizes (level 0) for convenience; NGFF scaling is encoded via
        # coordinateTransformations in multiscales datasets.
        omero["pixel_size"] = {k: v for k, v in ps.items() if v is not None}
    if omero:
        metadata["omero"] = omero
    if is_label:
        # Minimal marker so viewers can treat the image as label data.
        metadata["image-label"] = {}

    # Write image
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

    # ome-zarr-py does not reliably persist all extra root-level metadata passed via
    # **metadata across all versions, so set root attrs explicitly.
    for k, v in metadata.items():
        root.attrs[k] = v


def write_labels_omezarr(
    label_data: Union[np.ndarray, da.Array],
    out_path: str,
    label_name: str,
    axes: str = "yx",
    pixel_size_um: Optional[Union[float, Tuple[float, ...], Dict[str, float]]] = None,
    chunk_size: Optional[Tuple[int, ...]] = None,
) -> None:
    """Write a label mask to OME-Zarr format as a child of an existing image or standalone.

    If out_path points to an existing .zarr image, labels are added under /labels.
    """
    # Check if out_path is an existing zarr group (v2: .zattrs, v3: zarr.json)
    if os.path.exists(out_path) and (
        os.path.exists(os.path.join(out_path, ".zattrs"))
        or os.path.exists(os.path.join(out_path, "zarr.json"))
    ):
        mode = "r+"
    else:
        mode = "w"
        os.makedirs(out_path, exist_ok=True)

    root = zarr.open_group(out_path, mode=mode, zarr_format=ZARR_FORMAT)

    # Ensure label dtype is a reasonable integer type for viewer compatibility.
    #
    # Note: Although unsigned integer labels are valid, some napari/vispy stacks
    # are more robust with *signed* int32 label textures. Prefer int32 when safe.
    def _coerce_labels_to_int32_if_safe(arr):
        if not np.issubdtype(arr.dtype, np.integer):
            raise ValueError(
                f"Label data for '{label_name}' must be integer type, got dtype={arr.dtype}"
            )

        # Fast path: already int32
        if arr.dtype == np.int32:
            return arr

        # Numpy arrays: check range if we want to downcast safely
        if isinstance(arr, np.ndarray):
            # Empty labels are fine; just cast to int32
            if arr.size == 0:
                return arr.astype(np.int32)
            vmax = int(arr.max())
            vmin = int(arr.min())
            if vmin >= np.iinfo(np.int32).min and vmax <= np.iinfo(np.int32).max:
                return arr.astype(np.int32)
            # Fall back to int64 (still integer, but larger); keeps correctness.
            return arr.astype(np.int64)

        # Dask arrays: we may not want to compute min/max; just cast based on dtype size.
        if isinstance(arr, da.Array):
            if arr.dtype == np.int64:
                # Keep int64 to avoid accidental overflow without computing bounds.
                return arr
            if arr.dtype == np.uint32:
                # Keep uint32 to avoid overflow without computing bounds.
                return arr
            # Most other integer types are safe to cast to int32 (e.g. uint16, int16)
            return arr.astype(np.int32)

        return arr

    label_data = _coerce_labels_to_int32_if_safe(label_data)

    if not isinstance(label_data, da.Array):
        if chunk_size is None:
            # Default chunking for labels
            shape = label_data.shape
            chunks = list(shape)
            if "y" in axes and "x" in axes:
                y_idx = axes.find("y")
                x_idx = axes.find("x")
                chunks[y_idx] = min(shape[y_idx], 1024)
                chunks[x_idx] = min(shape[x_idx], 1024)
            chunk_size = tuple(chunks)
        label_data = da.from_array(label_data, chunks=chunk_size)

    # Transformations
    coordinate_transformations = []
    # TODO: add num_levels under ome-zarr config optional params
    num_levels = 5  # Default for Scaler()

    def _parse_pixel_sizes(ps) -> Dict[str, Optional[float]]:
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

    ps = _parse_pixel_sizes(pixel_size_um)

    for i in range(num_levels):
        scale_transform = [1.0] * len(axes)
        # Labels follow the same downsampling behavior: X/Y only, Z constant.
        if "z" in axes and ps.get("z") is not None:
            scale_transform[axes.find("z")] = ps["z"]
        if "y" in axes and ps.get("y") is not None:
            scale_transform[axes.find("y")] = ps["y"] * (2**i)
        if "x" in axes and ps.get("x") is not None:
            scale_transform[axes.find("x")] = ps["x"] * (2**i)
        coordinate_transformations.append([{"scale": scale_transform, "type": "scale"}])

    write_labels(
        labels=label_data,
        group=root,
        name=label_name,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
    )


def write_table_zarr(
    df: pd.DataFrame,
    out_path: str,
    chunk_size: int = 10000,
) -> None:
    """Write a pandas DataFrame to Zarr (columnar format).

    This is NOT OME-NGFF AnnData, but a simple columnar store for interoperability.
    """
    os.makedirs(out_path, exist_ok=True)
    store = zarr.open_group(out_path, mode="w", zarr_format=ZARR_FORMAT)

    # Write metadata
    store.attrs["columns"] = list(df.columns)
    store.attrs["index_name"] = df.index.name
    store.attrs["num_rows"] = len(df)

    # Write index
    if df.index.name:
        _write_series_to_zarr(df.index.to_series(), store, df.index.name, chunk_size)

    # Write columns
    for col in df.columns:
        _write_series_to_zarr(df[col], store, col, chunk_size)


def _write_series_to_zarr(
    series: pd.Series, group: zarr.Group, name: str, chunk_size: int
):
    # Handle nullable integers (Int64, Int32)
    if pd.api.types.is_integer_dtype(series):
        if series.hasnans:
            # Fill NaNs with -1 for integer columns (common convention for IDs)
            values = series.fillna(-1).astype(np.int64).values
        else:
            # Ensure numpy dtype (not pandas IntegerArray)
            values = series.astype(np.int64).values
        dtype = values.dtype
    elif pd.api.types.is_float_dtype(series):
        values = series.astype(np.float64).values
        dtype = values.dtype
    elif pd.api.types.is_string_dtype(series) or series.dtype == "O":
        values = series.astype(str).values
        dtype = str
    else:
        values = series.values
        dtype = values.dtype

    group.create_array(
        name,
        data=values,
        shape=values.shape,
        chunks=(chunk_size,),
        dtype=dtype,
        overwrite=True,
    )
