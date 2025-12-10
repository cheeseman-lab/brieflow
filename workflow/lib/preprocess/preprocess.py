"""Functions for preprocessing ND2 files in preparation for downstream BrieFlow steps."""

import json
import math
import shutil
from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
try:
    from skimage.transform import downscale_local_mean as _skimage_downscale_local_mean
except ModuleNotFoundError:
    _skimage_downscale_local_mean = None

try:
    import nd2  # type: ignore
except ModuleNotFoundError:
    nd2 = None

from lib.shared.omezarr_utils import default_omero_color_ints


def _require_nd2() -> None:
    """Ensure the nd2 package is available before accessing reader functionality."""
    if nd2 is None:
        raise ModuleNotFoundError(
            "The 'nd2' package is required for ND2 file operations. "
            "Install it in the active environment to proceed."
        )


def _downscale_local_mean(image: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Downscale an image by averaging local neighborhoods."""
    if _skimage_downscale_local_mean is not None:
        return _skimage_downscale_local_mean(image, factors)

    c_factor, y_factor, x_factor = factors
    if c_factor != 1:
        raise ValueError(
            "Fallback downscale supports channel factors of 1 only. "
            f"Got {c_factor}."
        )

    pad_y = (y_factor - (image.shape[1] % y_factor)) % y_factor
    pad_x = (x_factor - (image.shape[2] % x_factor)) % x_factor
    if pad_y or pad_x:
        image = np.pad(
            image,
            ((0, 0), (0, pad_y), (0, pad_x)),
            mode="edge",
        )

    c, y, x = image.shape
    reshaped = image.reshape(
        c,
        y // y_factor,
        y_factor,
        x // x_factor,
        x_factor,
    )
    return reshaped.mean(axis=(2, 4), dtype=np.float32)


def extract_tile_metadata(
    tile_fp: str,
    plate: int,
    well: str,
    tile: int,
    cycle: int = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extracts metadata from a single ND2 file for a specific tile.

    Args:
        tile_fp (str): File path pointing to the ND2 file for the tile.
        plate (int): Plate number to associate with this metadata.
        well (str): Well to associate with this metadata.
        tile (int): Tile number to associate with this metadata.
        cycle (int, optional): Cycle number to associate with this metadata. Defaults to None.
        z_interval (int, optional): If set, samples z-planes at this interval to ensure metadata is one line per position. Defaults to 4.
        verbose (bool, optional): If True, prints metadata information. Defaults to False.

    Returns:
        pd.DataFrame: Extracted metadata for the given tile.
    """
    _require_nd2()

    if verbose:
        print(f"Processing tile {tile} from file {tile_fp}")

    with nd2.ND2File(tile_fp) as images:
        frame_meta = images.frame_metadata(0)

        if verbose:
            print(f"File shape: {images.shape}")
            print(f"Number of dimensions: {images.ndim}")
            print(f"Data type: {images.dtype}")
            print(f"Sizes (by axes): {images.sizes}")

        # Get position data from first channel's position information
        if frame_meta.channels and hasattr(frame_meta.channels[0], "position"):
            stage_pos = frame_meta.channels[0].position.stagePositionUm
            metadata = {
                "x_pos": stage_pos.x,
                "y_pos": stage_pos.y,
                "z_pos": stage_pos.z,
                "pfs_offset": frame_meta.channels[0].position.pfsOffset,
            }
        else:
            metadata = {
                "x_pos": None,
                "y_pos": None,
                "z_pos": None,
                "pfs_offset": None,
            }

        # Add basic metadata
        metadata.update(
            {
                "plate": plate,
                "well": well,
                "tile": tile,
            }
        )

        # Conditionally add cycle after tile
        if cycle is not None:
            metadata["cycle"] = cycle

        # Add remaining metadata
        metadata.update(
            {
                "filename": tile_fp,
                "channels": frame_meta.contents.channelCount,
            }
        )

        # Get pixel size from first channel's volume information
        if frame_meta.channels and hasattr(frame_meta.channels[0], "volume"):
            x_cal, y_cal, _ = frame_meta.channels[0].volume.axesCalibration
            metadata.update(
                {
                    "pixel_size_x": x_cal,
                    "pixel_size_y": y_cal,
                }
            )
        else:
            metadata.update(
                {
                    "pixel_size_x": None,
                    "pixel_size_y": None,
                }
            )

        df = pd.DataFrame([metadata])

    return df


def nd2_to_tiff(
    files: Union[str, List[str], Path, List[Path]],
    channel_order_flip: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Converts one or multiple ND2 files to a multidimensional numpy array, ensuring CYX structure.

    Args:
        files: Path(s) to the ND2 file(s). Can be a single path or list of paths.
        channel_order_flip: If True, flips the channel order. Defaults to False.
        verbose: If True, prints dimension information. Defaults to False.

    Returns:
        np.ndarray: Image data as a multidimensional numpy array in CYX format.

    Raises:
        ValueError: If files have incompatible dimensions.
    """
    _require_nd2()

    # Convert input to list of Path objects
    if isinstance(files, (str, Path)):
        files = [Path(files)]
    else:
        files = [Path(f) for f in files]

    # Process all files
    image_arrays = []
    for i, file in enumerate(files, 1):
        if verbose:
            print(f"Processing file {i}/{len(files)}: {file}")

        image = nd2.imread(str(file), xarray=True)

        if verbose:
            print(f"Original dimensions for {file}: {image.dims}")

        # Handle Z-stack if present
        if "Z" in image.dims:
            image = image.max(dim="Z")

        # Convert to numpy array based on dimensions present
        if "C" in image.dims:
            # If C dimension exists, ensure CYX order
            img_array = image.transpose("C", "Y", "X").values

            # Flip channel order if needed
            if channel_order_flip:
                img_array = np.flip(img_array, axis=0)
        else:
            # If no C dimension, assume YX and add channel dimension
            img_array = image.transpose("Y", "X").values
            img_array = np.expand_dims(img_array, axis=0)  # Add channel dimension

        if verbose:
            print(f"Array shape after processing: {img_array.shape}")

        # Check dimensions match if not first image
        if image_arrays and img_array.shape[1:] != image_arrays[0].shape[1:]:
            raise ValueError(
                f"File {file} has incompatible dimensions: {img_array.shape} vs {image_arrays[0].shape}"
            )

        image_arrays.append(img_array)

    # Concatenate along channel axis (axis 0)
    result = np.concatenate(image_arrays, axis=0)

    if verbose:
        print(f"Final dimensions (CYX): {result.shape}")

    return result.astype(np.uint16)


def _resolve_pixel_size(first_file: Path) -> Tuple[float, float]:
    """Fetch XY pixel size (micrometers) from an ND2 file, defaulting to 1.0 when missing."""
    _require_nd2()

    try:
        with nd2.ND2File(str(first_file)) as handle:
            frame_meta = handle.frame_metadata(0)
            if (
                frame_meta.channels
                and hasattr(frame_meta.channels[0], "volume")
                and frame_meta.channels[0].volume.axesCalibration
            ):
                x_cal, y_cal, _ = frame_meta.channels[0].volume.axesCalibration
                return float(x_cal) if x_cal else 1.0, float(y_cal) if y_cal else 1.0
    except Exception as exc:
        print(f"Warning: unable to read pixel size from {first_file}: {exc}")

    return 1.0, 1.0


def _build_pyramid(
    image: np.ndarray,
    coarsening_factor: int,
    chunk_shape: Sequence[int],
    max_levels: int | None,
) -> List[np.ndarray]:
    """Create a multiscale pyramid with downscale_local_mean."""
    if coarsening_factor < 1:
        raise ValueError("coarsening_factor must be greater than or equal to 1.")

    pyramid = [np.ascontiguousarray(image)]

    while True:
        if max_levels is not None and len(pyramid) >= max_levels:
            break

        current = pyramid[-1]
        if coarsening_factor == 1:
            break

        next_y = current.shape[-2] // coarsening_factor
        next_x = current.shape[-1] // coarsening_factor
        if next_y < 1 or next_x < 1:
            break

        reduced = _downscale_local_mean(
            current.astype(np.float32),
            (1, coarsening_factor, coarsening_factor),
        )

        if np.issubdtype(image.dtype, np.integer):
            reduced = np.clip(np.round(reduced), 0, np.iinfo(image.dtype).max).astype(
                image.dtype
            )
        else:
            reduced = reduced.astype(image.dtype)

        pyramid.append(np.ascontiguousarray(reduced))

    return pyramid


def _write_zarr_array(
    level_data: np.ndarray,
    level_path: Path,
    chunk_shape: Sequence[int],
) -> None:
    """Write a single pyramid level to a Zarr array on disk."""
    level_path.mkdir(parents=True, exist_ok=True)

    dtype = level_data.dtype

    # zarr array metadata
    zarray_meta = {
        "chunks": list(chunk_shape),
        "compressor": None,
        "dtype": np.dtype(dtype).str,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(level_data.shape),
        "zarr_format": 2,
        "dimension_separator": "/",
    }
    (level_path / ".zarray").write_text(json.dumps(zarray_meta), encoding="utf-8")
    (level_path / ".zattrs").write_text(
        json.dumps({"_ARRAY_DIMENSIONS": ["c", "y", "x"]}), encoding="utf-8"
    )

    chunk_ranges = [
        range(math.ceil(dim / chunk)) for dim, chunk in zip(level_data.shape, chunk_shape)
    ]

    for c_idx in chunk_ranges[0]:
        c_start = c_idx * chunk_shape[0]
        c_end = min((c_idx + 1) * chunk_shape[0], level_data.shape[0])

        for y_idx in chunk_ranges[1]:
            y_start = y_idx * chunk_shape[1]
            y_end = min((y_idx + 1) * chunk_shape[1], level_data.shape[1])

            for x_idx in chunk_ranges[2]:
                x_start = x_idx * chunk_shape[2]
                x_end = min((x_idx + 1) * chunk_shape[2], level_data.shape[2])

                chunk = np.zeros(chunk_shape, dtype=level_data.dtype)
                chunk_c = c_end - c_start
                chunk_y = y_end - y_start
                chunk_x = x_end - x_start
                chunk[:chunk_c, :chunk_y, :chunk_x] = level_data[
                    c_start:c_end,
                    y_start:y_end,
                    x_start:x_end,
                ]

                chunk_fp = level_path / str(c_idx) / str(y_idx) / str(x_idx)
                chunk_fp.parent.mkdir(parents=True, exist_ok=True)

                chunk_view = chunk[:chunk_c, :chunk_y, :chunk_x]
                chunk_bytes = np.ascontiguousarray(chunk_view).tobytes(order="C")
                chunk_fp.write_bytes(chunk_bytes)


def _write_group_metadata(
    output_dir: Path,
    datasets: List[dict],
    pixel_size: Tuple[float, float],
    dtype: np.dtype,
) -> None:
    """Write root-level Zarr metadata."""
    (output_dir / ".zgroup").write_text(
        json.dumps({"zarr_format": 2}), encoding="utf-8"
    )

    pixel_x, pixel_y = pixel_size

    multiscales = [
        {
            "version": "0.4",
            "type": "image",
            "metadata": {"method": "mean"},
            "axes": [
                {"name": "c", "type": "channel"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": datasets,
        }
    ]

    dtype_info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
    window_max = int(dtype_info.max) if dtype_info else 1.0
    window_min = 0
    window_start = window_min
    window_end = window_max
    n_channels = int(datasets[0]["shape"][0]) if datasets else 0
    channel_colors = default_omero_color_ints(n_channels) if n_channels else []

    def _format_color(color_value: int | str | None) -> str:
        """Ensure channel colors are NGFF-compliant hex strings."""
        if isinstance(color_value, str):
            cleaned = color_value.strip().lstrip("#").upper()
            if len(cleaned) == 6 and all(c in "0123456789ABCDEF" for c in cleaned):
                return cleaned
            # Fall through to integer conversion if string is not valid hex
            try:
                color_value = int(color_value, 16)
            except ValueError:
                return "FFFFFF"

        if color_value is None:
            return "FFFFFF"

        try:
            color_int = int(color_value)
        except (TypeError, ValueError):
            return "FFFFFF"

        return f"{color_int & 0xFFFFFF:06X}"

    omero = {
        "channels": [
            {
                "label": f"Channel {idx}",
                "color": _format_color(
                    channel_colors[idx] if idx < len(channel_colors) else None
                ),
                "window": {
                    "min": window_min,
                    "max": window_max,
                    "start": window_start,
                    "end": window_end,
                },
            }
            for idx in range(n_channels)
        ],
        "pixel_size": {
            "x": pixel_x,
            "y": pixel_y,
            "unit": "micrometer",
        },
    }

    root_attrs = {
        "multiscales": multiscales,
        "omero": omero,
    }

    (output_dir / ".zattrs").write_text(json.dumps(root_attrs), encoding="utf-8")


def write_multiscale_omezarr(
    image: np.ndarray,
    output_dir: Union[str, Path],
    pixel_size: Tuple[float, float] = (1.0, 1.0),
    chunk_shape: Sequence[int] = (1, 512, 512),
    coarsening_factor: int = 2,
    max_levels: int | None = None,
) -> Path:
    """Persist a CYX image array as a multiscale OME-Zarr pyramid."""
    if image.ndim != 3:
        raise ValueError("Expected image in CYX format.")
    if len(chunk_shape) != 3:
        raise ValueError("chunk_shape must define three dimensions (C, Y, X).")
    if any(dim <= 0 for dim in chunk_shape):
        raise ValueError("chunk_shape values must be positive.")

    output_path = Path(output_dir)
    if output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            raise ValueError(f"Output path {output_path} exists and is not a directory.")

    output_path.mkdir(parents=True, exist_ok=False)

    pyramid = _build_pyramid(
        image,
        coarsening_factor=coarsening_factor,
        chunk_shape=chunk_shape,
        max_levels=max_levels,
    )

    datasets_meta: List[dict] = []
    base_pixel_size = pixel_size

    for idx, level in enumerate(pyramid):
        level_name = str(idx)
        level_path = output_path / level_name
        _write_zarr_array(level, level_path, chunk_shape)

        datasets_meta.append(
            {
                "path": level_name,
                "shape": list(level.shape),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": [
                            1,
                            base_pixel_size[1] * (coarsening_factor**idx),
                            base_pixel_size[0] * (coarsening_factor**idx),
                        ],
                    }
                ],
            }
        )

    _write_group_metadata(
        output_path, datasets_meta, base_pixel_size, dtype=image.dtype
    )

    return output_path


def nd2_to_omezarr(
    files: Union[str, List[str], Path, List[Path]],
    output_dir: Union[str, Path],
    channel_order_flip: bool = False,
    chunk_shape: Sequence[int] = (1, 512, 512),
    coarsening_factor: int = 2,
    max_levels: int | None = None,
    verbose: bool = False,
) -> Path:
    """Convert ND2 file(s) to a multiscale OME-Zarr pyramid."""
    _require_nd2()

    if isinstance(files, (str, Path)):
        normalized_files = [Path(files)]
    else:
        normalized_files = [Path(file) for file in files]

    if not normalized_files:
        raise ValueError("No ND2 files supplied for OME-Zarr conversion.")

    if verbose:
        print(
            f"Converting {len(normalized_files)} ND2 file(s) to OME-Zarr at {output_dir}."
        )

    image = nd2_to_tiff(
        normalized_files, channel_order_flip=channel_order_flip, verbose=verbose
    )
    pixel_size = _resolve_pixel_size(normalized_files[0])

    return write_multiscale_omezarr(
        image=image,
        output_dir=output_dir,
        pixel_size=pixel_size,
        chunk_shape=chunk_shape,
        coarsening_factor=coarsening_factor,
        max_levels=max_levels,
    )
