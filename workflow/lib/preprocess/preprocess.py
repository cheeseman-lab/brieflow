"""Functions for preprocessing ND2 files in preparation for downstream BrieFlow steps."""

from pathlib import Path
from typing import List, Sequence, Tuple, Union

import numpy as np
import pandas as pd

try:
    import nd2  # type: ignore
except ModuleNotFoundError:
    nd2 = None

from lib.shared.omezarr_io import write_multiscale_omezarr


def _require_nd2() -> None:
    """Ensure the nd2 package is available before accessing reader functionality."""
    if nd2 is None:
        raise ModuleNotFoundError(
            "The 'nd2' package is required for ND2 file operations. "
            "Install it in the active environment to proceed."
        )


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
        preserve_z (bool): If True, preserves the Z-stack dimension; otherwise, flattens by taking the max projection. Defaults to False.

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


def nd2_to_omezarr(
    files: Union[str, List[str], Path, List[Path]],
    output_dir: Union[str, Path],
    channel_order_flip: bool = False,
    chunk_shape: Sequence[int] = (1, 1024, 1024),
    coarsening_factor: int = 2,
    max_levels: int | None = None,
    verbose: bool = False,
    compressor: dict | None = None,
    preserve_z: bool = False,
) -> Path:
    """Convert ND2 file(s) to a multiscale OME-Zarr pyramid.
    
    Args:
        files: Path(s) to ND2 file(s)
        output_dir: Output directory for OME-Zarr
        channel_order_flip: If True, flips channel order
        chunk_shape: Chunk shape for Zarr arrays
        coarsening_factor: Factor for pyramid downsampling
        max_levels: Maximum pyramid levels
        verbose: Print progress information
        compressor: Compression settings
        preserve_z: If True, preserves Z-stacks; if False, does max projection (default: False to match TIFF behavior)
    """
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
        preserve_z=True,  # Always preserve Z for OME-Zarr to match TIFF workflow IC calculation
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
        compressor=compressor,
    )