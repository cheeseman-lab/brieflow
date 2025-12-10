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
    preserve_z: bool = False,
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
            if preserve_z:
                pass
            else:
                image = image.max(dim="Z")
        elif preserve_z:
            image = image.expand_dims("Z")

        # Convert to numpy array based on dimensions present
        if "C" in image.dims:
            if preserve_z:
                if "Z" not in image.dims:
                    image = image.expand_dims("Z")
                img_array = image.transpose("C", "Z", "Y", "X").values
            else:
                img_array = image.transpose("C", "Y", "X").values

            # Flip channel order if needed
            if channel_order_flip:
                img_array = np.flip(img_array, axis=0)
        else:
            # If no C dimension, assume YX and add channel dimension
            if preserve_z:
                img_array = image.transpose("Z", "Y", "X").values
                img_array = np.expand_dims(img_array, axis=0)
            else:
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
        suffix = "CZYX" if preserve_z else "CYX"
        print(f"Final dimensions ({suffix}): {result.shape}")

    return result.astype(np.uint16)


def _resolve_pixel_sizes(first_file: Path) -> Tuple[float, float, float]:
    """Fetch ZYX pixel sizes (micrometers) from an ND2 file, defaulting to 1.0."""
    _require_nd2()

    pixel_x = 1.0
    pixel_y = 1.0
    pixel_z = 1.0

    try:
        with nd2.ND2File(str(first_file)) as handle:
            frame_meta = handle.frame_metadata(0)
            if (
                frame_meta.channels
                and hasattr(frame_meta.channels[0], "volume")
                and frame_meta.channels[0].volume.axesCalibration
            ):
                axes_calibration = frame_meta.channels[0].volume.axesCalibration
                if len(axes_calibration) >= 3:
                    x_cal, y_cal, z_cal = axes_calibration[:3]
                else:
                    x_cal = axes_calibration[0]
                    y_cal = axes_calibration[1]
                    z_cal = None
                pixel_x = float(x_cal) if x_cal else 1.0
                pixel_y = float(y_cal) if y_cal else 1.0
                pixel_z = float(z_cal) if z_cal else 1.0
    except Exception as exc:
        print(f"Warning: unable to read pixel size from {first_file}: {exc}")

    return pixel_z, pixel_y, pixel_x


def _build_pyramid(
    image: np.ndarray,
    coarsening_factor: int,
    max_levels: int | None,
) -> List[np.ndarray]:
    """Create a multiscale pyramid by reducing only spatial (Y/X) axes."""
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

        reduced = _downscale_spatial_dims(
            current.astype(np.float32), coarsening_factor
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
    """Write a single pyramid level (CYX or CZYX) to a Zarr array on disk."""
    level_path.mkdir(parents=True, exist_ok=True)

    if level_data.ndim not in (3, 4):
        raise ValueError("Expected data in CYX or CZYX format.")

    normalized_chunk = tuple(int(x) for x in chunk_shape)
    if level_data.ndim == 4:
        if len(normalized_chunk) == 3:
            normalized_chunk = (
                normalized_chunk[0],
                1,
                normalized_chunk[1],
                normalized_chunk[2],
            )
        elif len(normalized_chunk) != 4:
            raise ValueError(
                "chunk_shape must have 3 (C, Y, X) or 4 (C, Z, Y, X) entries when data has a Z axis."
            )
    else:
        if len(normalized_chunk) != 3:
            raise ValueError("chunk_shape must define (C, Y, X) for 3D data.")

    dimensions = ["c", "z", "y", "x"] if level_data.ndim == 4 else ["c", "y", "x"]

    zarray_meta = {
        "chunks": list(normalized_chunk),
        "compressor": None,
        "dtype": np.dtype(level_data.dtype).str,
        "fill_value": 0,
        "filters": None,
        "order": "C",
        "shape": list(level_data.shape),
        "zarr_format": 2,
        "dimension_separator": "/",
    }
    (level_path / ".zarray").write_text(json.dumps(zarray_meta), encoding="utf-8")
    (level_path / ".zattrs").write_text(
        json.dumps({"_ARRAY_DIMENSIONS": dimensions}), encoding="utf-8"
    )

    chunk_ranges = [
        range(math.ceil(dim / chunk))
        for dim, chunk in zip(level_data.shape, normalized_chunk)
    ]

    for indices in product(*chunk_ranges):
        slices = []
        bounds = []
        for axis, (idx, chunk_len) in enumerate(zip(indices, normalized_chunk)):
            start = idx * chunk_len
            end = min((idx + 1) * chunk_len, level_data.shape[axis])
            slices.append(slice(start, end))
            bounds.append(slice(0, end - start))

        chunk = np.zeros(normalized_chunk, dtype=level_data.dtype)
        chunk[tuple(bounds)] = level_data[tuple(slices)]

        chunk_fp = level_path
        for idx in indices:
            chunk_fp /= str(idx)
        chunk_fp.parent.mkdir(parents=True, exist_ok=True)
        chunk_bytes = np.ascontiguousarray(chunk).tobytes(order="C")
        chunk_fp.write_bytes(chunk_bytes)


def _write_group_metadata(
    output_dir: Path,
    datasets: List[dict],
    pixel_size: Tuple[float, float, float],
    dtype: np.dtype,
    has_z: bool,
    image_name: str | None = None,
) -> None:
    """Write root-level Zarr metadata."""
    (output_dir / ".zgroup").write_text(
        json.dumps({"zarr_format": 2}), encoding="utf-8"
    )

    pixel_z, pixel_y, pixel_x = pixel_size

    multiscale_entry: dict = {
        "version": "0.4",
        "type": "image",
        "metadata": {"method": "mean"},
        "axes": (
                [
                    {"name": "c", "type": "channel"},
                    *(
                        [{
                            "name": "z",
                            "type": "space",
                            "unit": "micrometer",
                        }]
                        if has_z
                        else []
                    ),
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ]
            ),
        "datasets": datasets,
    }
    if image_name:
        multiscale_entry["name"] = image_name

    multiscales = [multiscale_entry]

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

    pixel_dict = {
        "x": pixel_x,
        "y": pixel_y,
        "unit": "micrometer",
    }
    if has_z:
        pixel_dict["z"] = pixel_z

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
        "pixel_size": pixel_dict,
    }

    root_attrs = {
        "multiscales": multiscales,
        "omero": omero,
    }

    (output_dir / ".zattrs").write_text(json.dumps(root_attrs), encoding="utf-8")


def write_multiscale_omezarr(
    image: np.ndarray,
    output_dir: Union[str, Path],
    pixel_size: Sequence[float] = (1.0, 1.0, 1.0),
    chunk_shape: Sequence[int] = (1, 512, 512),
    coarsening_factor: int = 2,
    max_levels: int | None = None,
    image_name: str | None = None,
) -> Path:
    """Persist a CYX or CZYX image array as a multiscale OME-Zarr pyramid."""
    if image.ndim not in (3, 4):
        raise ValueError("Expected image in CYX or CZYX format.")
    if any(dim <= 0 for dim in chunk_shape):
        raise ValueError("chunk_shape values must be positive.")

    has_z = image.ndim == 4

    if len(pixel_size) == 3:
        pixel_z, pixel_y, pixel_x = pixel_size
    elif len(pixel_size) == 2:
        pixel_y, pixel_x = pixel_size
        pixel_z = 1.0
    else:
        raise ValueError("pixel_size must contain 2 (Y, X) or 3 (Z, Y, X) values.")

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
        max_levels=max_levels,
    )

    datasets_meta: List[dict] = []
    base_pixel_size = (pixel_z, pixel_y, pixel_x)

    for idx, level in enumerate(pyramid):
        level_name = str(idx)
        level_path = output_path / level_name
        _write_zarr_array(level, level_path, chunk_shape)

        scale = [1]
        if has_z:
            scale.append(pixel_z)
        scale.extend(
            [
                pixel_y * (coarsening_factor**idx),
                pixel_x * (coarsening_factor**idx),
            ]
        )

        datasets_meta.append(
            {
                "path": level_name,
                "shape": list(level.shape),
                "coordinateTransformations": [
                    {
                        "type": "scale",
                        "scale": scale,
                    }
                ],
            }
        )

    _write_group_metadata(
        output_path,
        datasets_meta,
        base_pixel_size,
        dtype=image.dtype,
        has_z=has_z,
        image_name=image_name,
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
        normalized_files,
        channel_order_flip=channel_order_flip,
        verbose=verbose,
        preserve_z=True,
    )
    pixel_size = _resolve_pixel_sizes(normalized_files[0])

    primary_names = [path.stem for path in normalized_files]
    if len(primary_names) == 1:
        image_name = primary_names[0]
    else:
        image_name = ", ".join(primary_names)

    return write_multiscale_omezarr(
        image=image,
        output_dir=output_dir,
        pixel_size=pixel_size,
        chunk_shape=chunk_shape,
        coarsening_factor=coarsening_factor,
        max_levels=max_levels,
        image_name=image_name,
    )
