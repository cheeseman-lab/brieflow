"""Unified I/O interface for image reading and writing."""

from pathlib import Path
from typing import Optional, Union, List, Any, Tuple

import numpy as np
from tifffile import imread as _tiff_imread
from tifffile import imwrite as _tiff_imwrite

try:
    import zarr
except ImportError:
    zarr = None

from lib.shared.omezarr_io import write_multiscale_omezarr


def read_image(path: Union[str, Path]) -> np.ndarray:
    """Read an image from TIFF or OME-Zarr.

    Auto-detects format:
    - If path is a directory with .zgroup/.zattrs or ends in .zarr -> OME-Zarr
    - Else -> TIFF

    Returns:
        np.ndarray: Image data in (C, Y, X) or (C, Z, Y, X) format.
    """
    path = Path(path)

    is_zarr = False
    if path.suffix == ".zarr":
        is_zarr = True
    elif path.is_dir() and (path / ".zgroup").exists():
        is_zarr = True

    if is_zarr:
        if zarr is None:
            raise ImportError("zarr package is required to read OME-Zarr files.")
        
        # Open the group
        store = zarr.open_group(str(path), mode="r")
        
        # Check for multiscales
        attrs = store.attrs.asdict()
        if "multiscales" in attrs:
            # Get the path to the highest resolution (usually index 0)
            datasets = attrs["multiscales"][0]["datasets"]
            if datasets:
                high_res_path = datasets[0]["path"]
                arr = store[high_res_path]
                # Load into memory as numpy array
                return np.array(arr)
        
        # Fallback: try to find '0' or 'scale0'
        if "0" in store:
            return np.array(store["0"])
        
        raise ValueError(f"Could not find image data in OME-Zarr: {path}")

    # Assume TIFF
    return _tiff_imread(path)


def read_pixel_size(path: Union[str, Path]) -> Tuple[float, float, float]:
    """Read pixel size (Z, Y, X) from image metadata (OME-Zarr).

    Returns:
        Tuple[float, float, float]: (pixel_size_z, pixel_size_y, pixel_size_x).
        Defaults to (1.0, 1.0, 1.0) if not found or not OME-Zarr.
    """
    path = Path(path)
    # Check for OME-Zarr
    is_zarr = path.suffix == ".zarr" or (path.is_dir() and (path / ".zgroup").exists())

    if is_zarr:
        if zarr is None:
            return (1.0, 1.0, 1.0)
        try:
            store = zarr.open_group(str(path), mode="r")
            attrs = store.attrs.asdict()
            if "omero" in attrs and "pixel_size" in attrs["omero"]:
                p = attrs["omero"]["pixel_size"]
                return (float(p.get("z", 1.0)), float(p.get("y", 1.0)), float(p.get("x", 1.0)))
        except Exception:
            pass
            
    return (1.0, 1.0, 1.0)


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    pixel_size: Optional[tuple] = None,
    channel_names: Optional[List[str]] = None,
    is_label: bool = False,
    **kwargs: Any,
) -> None:
    """Save an image to TIFF or OME-Zarr.

    Args:
        image: Numpy array (C, Y, X) or (C, Z, Y, X).
        path: Output path. Extension determines format.
        pixel_size: (Z, Y, X) or (Y, X) pixel size in microns.
        channel_names: List of channel names.
        is_label: If True, treats as segmentation mask (OME-Zarr only).
        **kwargs: Additional arguments passed to the writer.
    """
    path = Path(path)
    
    # Ensure correct dimensionality for OME-Zarr writer (CYX or CZYX)
    if image.ndim == 2:
        # YX -> 1YX
        image = image[np.newaxis, ...]
    
    if path.suffix == ".zarr":
        # Extract pixel_size from kwargs if not provided
        if pixel_size is None:
            if "pixel_size_z" in kwargs and "pixel_size_y" in kwargs and "pixel_size_x" in kwargs:
                pixel_size = (kwargs["pixel_size_z"], kwargs["pixel_size_y"], kwargs["pixel_size_x"])
            elif "pixel_size_y" in kwargs and "pixel_size_x" in kwargs:
                pixel_size = (kwargs["pixel_size_y"], kwargs["pixel_size_x"])

        # Defaults
        chunk_shape = kwargs.get("chunk_shape", (1, 1024, 1024))
        coarsening_factor = kwargs.get("coarsening_factor", 2)
        max_levels = kwargs.get("max_levels", None)
        # TODO: Re-enable blosc compression once blosc dependency is resolved
        # compressor = kwargs.get("compressor", {"id": "blosc", "cname": "zstd", "clevel": 3, "shuffle": 2})
        compressor = kwargs.get("compressor", None)  # Temporarily disable compression
        
        default_pixel_size = (1.0, 1.0, 1.0) if image.ndim == 4 else (1.0, 1.0)
        p_size = pixel_size if pixel_size is not None else default_pixel_size

        write_multiscale_omezarr(
            image=image,
            output_dir=path,
            pixel_size=p_size,
            chunk_shape=chunk_shape,
            coarsening_factor=coarsening_factor,
            max_levels=max_levels,
            compressor=compressor,
            is_label=is_label,
            channel_names=channel_names,
        )
    else:
        # TIFF fallback
        # For segmentation masks in TIFF, we just write the array
        _tiff_imwrite(path, image)
