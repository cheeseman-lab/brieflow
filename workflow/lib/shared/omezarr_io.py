"""Low-level OME-Zarr writing utilities."""

import json
import math
import shutil
from itertools import product
from pathlib import Path
from typing import List, Sequence, Tuple, Union, Optional

import numpy as np
from lib.shared.omezarr_utils import default_omero_color_ints

try:
    from skimage.transform import downscale_local_mean as _skimage_downscale_local_mean
except ModuleNotFoundError:
    _skimage_downscale_local_mean = None


def _downscale_local_mean(image: np.ndarray, factors: Sequence[int]) -> np.ndarray:
    """Downscale an image by averaging local neighborhoods."""
    if _skimage_downscale_local_mean is not None:
        return _skimage_downscale_local_mean(image, factors)

    c_factor, y_factor, x_factor = factors
    if c_factor != 1:
        raise ValueError(
            f"Fallback downscale supports channel factors of 1 only. Got {c_factor}."
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


def _downscale_spatial_dims(image: np.ndarray, factor: int) -> np.ndarray:
    """Downscale Y and X dimensions by `factor`."""
    if image.ndim == 3:  # CYX
        factors = (1, factor, factor)
    elif image.ndim == 4:  # CZYX
        factors = (1, 1, factor, factor)
    else:
        raise ValueError(
            "Unsupported dimensionality for downscaling (expected 3 or 4)."
        )
    return _downscale_local_mean(image, factors)


def _build_pyramid(
    image: np.ndarray,
    coarsening_factor: int,
    max_levels: int | None,
    min_spatial_shape: Tuple[int, int] | None = None,
) -> List[np.ndarray]:
    """Create a multiscale pyramid by reducing only spatial (Y/X) axes."""
    if coarsening_factor < 1:
        raise ValueError("coarsening_factor must be greater than or equal to 1.")

    pyramid = [np.ascontiguousarray(image)]

    while True:
        if max_levels is not None and len(pyramid) >= max_levels:
            break

        current = pyramid[-1]

        # Check if we should stop based on minimum dimensions (e.g. chunk size)
        # If the current image fits within the min shape, we don't need further downsampling
        if min_spatial_shape is not None:
            curr_y, curr_x = current.shape[-2:]
            min_y, min_x = min_spatial_shape
            if curr_y <= min_y and curr_x <= min_x:
                break

        if coarsening_factor == 1:
            break

        next_y = current.shape[-2] // coarsening_factor
        next_x = current.shape[-1] // coarsening_factor
        if next_y < 1 or next_x < 1:
            break

        reduced = _downscale_spatial_dims(current.astype(np.float32), coarsening_factor)

        if np.issubdtype(image.dtype, np.integer):
            # Clip and round to maintain integer integrity
            dtype_info = np.iinfo(image.dtype)
            reduced = np.clip(np.round(reduced), dtype_info.min, dtype_info.max).astype(
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
    compressor: dict | None = None,
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
        "compressor": compressor,
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

    compressor_id = compressor.get("id") if compressor else None

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
        raw_bytes = np.ascontiguousarray(chunk).tobytes(order="C")

        # TODO: Re-enable blosc compression once blosc dependency is properly configured
        # if compressor_id == "blosc":
        #     import blosc
        #     chunk_bytes = blosc.compress(
        #         raw_bytes,
        #         typesize=chunk.dtype.itemsize,
        #         cname=compressor.get("cname", "zstd"),
        #         clevel=int(compressor.get("clevel", 3)),
        #         shuffle=int(compressor.get("shuffle", 2)),
        #     )
        # else:
        #     chunk_bytes = raw_bytes

        # Temporarily disable compression to avoid blosc dependency issues
        chunk_bytes = raw_bytes

        chunk_fp.write_bytes(chunk_bytes)


def _write_group_metadata(
    output_dir: Path,
    datasets: List[dict],
    pixel_size: Tuple[float, float, float],
    dtype: np.dtype,
    has_z: bool,
    image_name: str | None = None,
    is_label: bool = False,
    channel_names: Optional[List[str]] = None,
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
                    [
                        {
                            "name": "z",
                            "type": "space",
                            "unit": "micrometer",
                        }
                    ]
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

    # Base attributes
    root_attrs = {
        "multiscales": multiscales,
    }

    # Add image-label metadata if this is a segmentation mask
    if is_label:
        root_attrs["image-label"] = {
            "version": "0.4",
            "source": {
                "image": "../../"  # Default relative path convention
            },
        }
    else:
        # Add OMERO metadata (colors, windows)
        # Only if it's NOT a label image (usually labels rely on separate lookup or random coloring)
        # OR if we want to provide labels for the channels (e.g. Nuclei, Cells)

        dtype_info = np.iinfo(dtype) if np.issubdtype(dtype, np.integer) else None
        window_max = int(dtype_info.max) if dtype_info else 1.0
        window_min = 0
        window_start = window_min
        window_end = window_max

        n_channels = int(datasets[0]["shape"][0]) if datasets else 0

        # Generate OME metadata
        pixel_dict = {
            "x": pixel_x,
            "y": pixel_y,
            "unit": "micrometer",
        }
        if has_z:
            pixel_dict["z"] = pixel_z

        omero = {
            "channels": [],
            "pixel_size": pixel_dict,
        }

        channel_colors = default_omero_color_ints(n_channels) if n_channels else []

        def _format_color(color_value: int | str | None) -> str:
            """Ensure channel colors are NGFF-compliant hex strings."""
            if color_value is None:
                return "FFFFFF"
            return f"{int(color_value) & 0xFFFFFF:06X}"

        for idx in range(n_channels):
            label = f"Channel {idx}"
            if channel_names and idx < len(channel_names):
                label = channel_names[idx]

            channel_entry = {
                "label": label,
                "window": {
                    "min": window_min,
                    "max": window_max,
                    "start": window_start,
                    "end": window_end,
                },
            }

            color = channel_colors[idx] if idx < len(channel_colors) else None
            channel_entry["color"] = _format_color(color)

            omero["channels"].append(channel_entry)

        root_attrs["omero"] = omero

    (output_dir / ".zattrs").write_text(json.dumps(root_attrs), encoding="utf-8")


def write_multiscale_omezarr(
    image: np.ndarray,
    output_dir: Union[str, Path],
    pixel_size: Sequence[float] = (1.0, 1.0, 1.0),
    chunk_shape: Sequence[int] = (1, 1024, 1024),
    coarsening_factor: int = 2,
    max_levels: int | None = None,
    image_name: str | None = None,
    compressor: dict | None = None,
    is_label: bool = False,
    channel_names: Optional[List[str]] = None,
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
            raise ValueError(
                f"Output path {output_path} exists and is not a directory."
            )

    output_path.mkdir(parents=True, exist_ok=False)

    pyramid = _build_pyramid(
        image,
        coarsening_factor=coarsening_factor,
        max_levels=max_levels,
        min_spatial_shape=tuple(chunk_shape[-2:]),
    )

    datasets_meta: List[dict] = []
    base_pixel_size = (pixel_z, pixel_y, pixel_x)

    for idx, level in enumerate(pyramid):
        level_name = str(idx)
        level_path = output_path / level_name
        _write_zarr_array(level, level_path, chunk_shape, compressor=compressor)

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
        is_label=is_label,
        channel_names=channel_names,
    )

    return output_path
