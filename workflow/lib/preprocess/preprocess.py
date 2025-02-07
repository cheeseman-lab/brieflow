"""Functions for preprocessing ND2 files in preparation for downstream BrieFlow steps."""

import pandas as pd
import numpy as np
import nd2
import gc
import warnings
from typing import Union, List, Dict, Tuple
from pathlib import Path


def extract_tile_metadata(
    tile_fp: str, well: str, tile: int, verbose: bool = False
) -> pd.DataFrame:
    """Extracts metadata from a single ND2 file for a specific tile.

    Args:
        tile_fp (str): File path pointing to the ND2 file for the tile.
        well (str): Well to associate with this metadata.
        tile (int): Tile number to associate with this metadata.
        z_interval (int, optional): If set, samples z-planes at this interval to ensure metadata is one line per position. Defaults to 4.
        verbose (bool, optional): If True, prints metadata information. Defaults to False.

    Returns:
        pd.DataFrame: Extracted metadata for the given tile.
    """
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
                "well": well,
                "tile": tile,
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


def extract_well_metadata(
        well_fp: str, well: str, verbose: bool = False
) -> pd.DataFrame:
    """Extracts metadata from an ND2 file containing multiple fields of view.

    Args:
        well_fp (str): File path pointing to the ND2 file for the well.
        well (str): Well to associate with this metadata.
        verbose (bool, optional): If True, prints metadata information. Defaults to False.

    Returns:
        pd.DataFrame: Extracted metadata for all fields of view in the well.
    """
    metadata_rows = []

    if verbose:
        print(f"Processing well file: {well_fp}")

    with nd2.ND2File(well_fp) as images:
        # Get basic file info
        if verbose:
            print(f"File shape: {images.shape}")
            print(f"Number of dimensions: {images.ndim}")
            print(f"Data type: {images.dtype}")
            print(f"Sizes (by axes): {images.sizes}")

        # Get number of positions (using 'P' instead of 'V')
        num_positions = images.sizes.get("P", 1)

        if verbose:
            print(f"Number of positions: {num_positions}")

        for pos_idx in range(num_positions):
            frame_meta = images.frame_metadata(pos_idx)

            # Extract position data if available
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
                    "tile": pos_idx,  # Using position index as tile number
                    "well": well,
                    "filename": well_fp,
                    "channels": images.sizes.get(
                        "C", 1
                    ),  # Get channel count from sizes
                }
            )

            # Get pixel calibration if available
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

            metadata_rows.append(metadata)

    df = pd.DataFrame(metadata_rows)
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


def nd2_to_tiff_well(
    files: Union[str, List[str], Path, List[Path]],
    position: int = 0,
    channel_order_flip: bool = False,
    return_tiles: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Extracts a specific position/FOV from one or multiple ND2 files. Handles Z-stacks by computing maximum intensity projection.

    Args:
        files: Path(s) to the ND2 file(s). Can be a single path or list of paths.
        position: Position/field of view to extract. Defaults to 0.
        channel_order_flip: If True, flips the channel order. Defaults to False.
        return_tiles: If True, returns the number of tiles. Defaults to False.
        verbose: If True, prints dimension information. Defaults to False.

    Returns:
        Union[np.ndarray, Tuple[np.ndarray, int]]: Image data as a multidimensional
            numpy array in CYX format. If return_tiles is True, also returns the
            number of tiles as a tuple.
    """
    # Convert input to list of Path objects
    if isinstance(files, (str, Path)):
        files = [Path(files)]
    else:
        files = [Path(f) for f in files]

    image_arrays = []

    for file in files:
        if verbose:
            print(f"\nProcessing file: {file}")

        nd2_obj = nd2.ND2File(str(file))
        try:
            if verbose:
                print(f"File dimensions: {nd2_obj.sizes}")

            # Get and save 'P' data from the ND2 file
            tiles = nd2_obj.sizes["P"]

            # Check if we have Z dimension
            if "Z" in nd2_obj.sizes:
                if verbose:
                    print(f"Z-stack detected with {nd2_obj.sizes['Z']} planes")

                # Get all Z planes for this position
                z_planes = []
                for z in range(nd2_obj.sizes["Z"]):
                    # Convert position and Z coordinates to sequence index
                    coords = {"P": position, "Z": z}
                    seq_idx = nd2_obj._seq_index_from_coords(
                        [coords[dim] for dim in nd2_obj.sizes if dim in coords]
                    )
                    z_planes.append(nd2_obj.read_frame(seq_idx))

                # Stack Z planes and take max projection
                img_data = np.stack(z_planes, axis=0)

                if verbose:
                    print(f"Z-stack shape: {img_data.shape}")

                # Compute max intensity projection
                img_data = np.max(img_data, axis=0)
            else:
                # No Z dimension, just read the position directly
                img_data = nd2_obj.read_frame(position)

            # Convert to uint16 and make a copy
            img_data = np.array(img_data, dtype=np.uint16, copy=True)

            if verbose:
                print(f"Frame shape after Z processing: {img_data.shape}")

            # If the image is 2D, add a channel dimension
            if img_data.ndim == 2:
                img_data = np.expand_dims(img_data, axis=0)

            # Flip channel order if needed
            if channel_order_flip and img_data.ndim > 2:
                img_data = np.flip(img_data, axis=0)

            image_arrays.append(img_data)

        except Exception as e:
            warnings.warn(f"Error processing {file}: {str(e)}")
            raise
        finally:
            try:
                nd2_obj.close()
            except OSError:
                pass
            gc.collect()

    if len(files) == 1:
        result = image_arrays[0]
    else:
        try:
            # Now all arrays should be 3D (CYX), so we can concatenate along channel axis
            result = np.concatenate(image_arrays, axis=0)

        except ValueError as e:
            shapes = [arr.shape for arr in image_arrays]
            raise ValueError(f"Cannot concatenate arrays with shapes {shapes}") from e

    if verbose:
        print(f"Final dimensions (CYX): {result.shape}")
        print(f"Array size in bytes: {result.nbytes}")

    if return_tiles:
        return result.astype(np.uint16), tiles
    else:
        return result.astype(np.uint16)
