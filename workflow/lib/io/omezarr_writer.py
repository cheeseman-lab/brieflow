"""
OME-Zarr Writer Module for Brieflow.

This module provides functions to export images, labels, and tables to OME-Zarr (v2) format.
It uses the ome-zarr-py library for NGFF compliance.
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


def write_image_omezarr(
    image_data: Union[np.ndarray, da.Array],
    out_path: str,
    channel_names: Optional[List[str]] = None,
    axes: str = "cyx",
    pixel_size_um: Optional[Union[float, Tuple[float, ...], Dict[str, float]]] = None,
    chunk_size: Optional[Tuple[int, ...]] = None,
    storage_options: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Write an image array to OME-Zarr format with pyramids.

    Args:
        image_data: Numpy or Dask array containing image data.
        out_path: Path to the output .zarr directory.
        channel_names: List of channel names. Length must match channel dimension.
        axes: String describing axes, e.g., "cyx", "tcyx".
        pixel_size_um: Pixel size in microns.
            - float: applied to X and Y
            - tuple: (y, x) or (z, y, x) depending on available axes
            - dict: keys from {"x","y","z"} (values can be None)
        chunk_size: Tuple for chunking (optional).
        storage_options: Options for storage backend (optional).
    """
    os.makedirs(out_path, exist_ok=True)
    # Enforce Zarr v2 for OME-NGFF v0.4 compliance
    # zarr.open_group handles store creation and avoids v3 zarr.json default
    root = zarr.open_group(out_path, mode='w', zarr_format=2)

    # Ensure dask array for efficient scaling
    if not isinstance(image_data, da.Array):
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

    # Coordinate transformations
    coordinate_transformations = []
    num_levels = 5 # Default for Scaler()
    
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
        # Apply per-axis scales for spatial axes. We assume pyramids downsample spatial
        # dimensions uniformly by 2**i.
        if "z" in axes and ps.get("z") is not None:
            scale_transform[axes.find("z")] = ps["z"] * (2**i)
        if "y" in axes and ps.get("y") is not None:
            scale_transform[axes.find("y")] = ps["y"] * (2**i)
        if "x" in axes and ps.get("x") is not None:
            scale_transform[axes.find("x")] = ps["x"] * (2**i)
        coordinate_transformations.append([{"scale": scale_transform, "type": "scale"}])
    # Metadata
    metadata = {}
    if channel_names:
        metadata["omero"] = {
            "channels": [
                {"label": name, "active": True, "color": "FFFFFF"} 
                for name in channel_names
            ]
        }

    # Write image
    write_image(
        image=image_data,
        group=root,
        axes=axes,
        coordinate_transformations=coordinate_transformations,
        scaler=Scaler(method="nearest"),
    )


def write_labels_omezarr(
    label_data: Union[np.ndarray, da.Array],
    out_path: str,
    label_name: str,
    axes: str = "yx",
    pixel_size_um: Optional[Union[float, Tuple[float, ...], Dict[str, float]]] = None,
    chunk_size: Optional[Tuple[int, ...]] = None,
) -> None:
    """
    Write a label mask to OME-Zarr format as a child of an existing image or standalone.
    
    If out_path points to an existing .zarr image, labels are added under /labels.
    """
    # Check if out_path is an existing group
    if os.path.exists(out_path) and os.path.exists(os.path.join(out_path, ".zattrs")):
        mode = "r+"
    else:
        mode = "w"
        os.makedirs(out_path, exist_ok=True)

    # Enforce Zarr v2
    root = zarr.open_group(out_path, mode=mode, zarr_format=2)

    # Ensure data type is compatible with OME-Zarr labels (unsigned int)
    if isinstance(label_data, np.ndarray):
        if label_data.dtype == np.int64 or label_data.dtype == np.int32:
             # Check if values fit in uint32
             if label_data.max() < 2**32:
                 label_data = label_data.astype(np.uint32)
    elif isinstance(label_data, da.Array):
         if label_data.dtype == np.int64 or label_data.dtype == np.int32:
             label_data = label_data.astype(np.uint32)

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
    num_levels = 5 # Default for Scaler()
    
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
        if "z" in axes and ps.get("z") is not None:
            scale_transform[axes.find("z")] = ps["z"] * (2**i)
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
    chunk_size: int = 10000
) -> None:
    """
    Write a pandas DataFrame to Zarr (columnar format).
    This is NOT OME-NGFF AnnData, but a simple columnar store for interoperability.
    """
    os.makedirs(out_path, exist_ok=True)
    # Enforce Zarr v2
    store = zarr.open(out_path, mode='w', zarr_format=2)
    
    # Write metadata
    store.attrs['columns'] = list(df.columns)
    store.attrs['index_name'] = df.index.name
    store.attrs['num_rows'] = len(df)
    
    # Write index
    if df.index.name:
        _write_series_to_zarr(df.index.to_series(), store, df.index.name, chunk_size)
    
    # Write columns
    for col in df.columns:
        _write_series_to_zarr(df[col], store, col, chunk_size)


def _write_series_to_zarr(series: pd.Series, group: zarr.Group, name: str, chunk_size: int):
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
    elif pd.api.types.is_string_dtype(series) or series.dtype == 'O':
        values = series.astype(str).values
        dtype = str
    else:
        values = series.values
        dtype = values.dtype
        
    ds = group.create_dataset(
        name,
        data=values,
        shape=values.shape,
        chunks=(chunk_size,),
        dtype=dtype,
        overwrite=True
    )
