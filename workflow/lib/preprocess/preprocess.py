"""Unified preprocessing module for microscopy image data.

This module provides a unified interface for preprocessing microscopy data from different
sources (ND2, TIFF) and organizations (tile-based, well-based). The main functions handle
metadata extraction and image conversion to standardized formats.

The module outputs images in CYX format (Channel, Y, X) which is the standard format
for downstream processing in the pipeline. This ensures consistent data structure
regardless of the input format.
"""

import re

import pandas as pd
import numpy as np
import nd2
from typing import Union, List, Dict, Any, Optional, Tuple, Sequence
from pathlib import Path
import warnings
import gc


# Data organization and format constants
DATA_FORMATS = {"nd2", "tiff"}
DATA_ORGANIZATIONS = {"tile", "well"}


def get_data_config(image_type: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Get data configuration for a specific image type.

    Args:
        image_type: Type of image data ('sbs' or 'phenotype')
        config: Configuration dictionary containing preprocessing parameters

    Returns:
        Dictionary with data configuration settings including:
        - data_format: 'nd2' or 'tiff'
        - data_organization: 'tile' or 'well'
        - channel_order_flip: Whether to reverse channel order
        - channel_order: List of channels in desired order
        - metadata_samples_df_fp: Path to metadata samples dataframe
        - n_z_planes: Number of z-planes per channel (None if no z-stacking)
    """
    base_config = config.get("preprocess", {})

    data_format = base_config.get(f"{image_type}_data_format", "nd2")
    data_org = base_config.get(f"{image_type}_data_organization", "tile")

    return {
        "data_format": data_format,
        "image_data_organization": "tile" if data_format == "tiff" else data_org,
        "metadata_data_organization": "well" if data_format == "tiff" else data_org,
        "channel_order_flip": base_config.get(
            f"{image_type}_channel_order_flip", False
        ),
        "channel_order": base_config.get(f"{image_type}_channel_order", None),
        "metadata_samples_df_fp": base_config.get(
            f"{image_type}_metadata_samples_df_fp", None
        ),
        "n_z_planes": base_config.get(f"{image_type}_n_z_planes", None),
    }


# TODO add direct nd2 to zarr conversion - bypass tiffs


def extract_metadata_tile_nd2(
    file_path: str,
    plate: Union[int, str],
    well: Union[int, str],
    tile: Union[int, str],
    cycle: Union[int, str] = None,
    round: Union[int, str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract metadata from a single ND2 tile file.

    Tile-based organization means each file contains a single field of view (FOV).
    This function extracts position information, pixel calibration, and other
    metadata from the ND2 file header.

    Args:
        file_path: Path to the ND2 file
        plate: Plate identifier
        well: Well identifier (e.g., 'A01', 'B12')
        tile: Tile/FOV number within the well
        cycle: Optional cycle number for SBS imaging
        round: Optional round number for multiplexed imaging
        verbose: Print debug information

    Returns:
        DataFrame with one row containing metadata for this tile
    """
    if verbose:
        print(f"Processing tile {tile} from file {file_path}")

    with nd2.ND2File(file_path) as images:
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
                "filename": file_path,
                "channels": frame_meta.contents.channelCount,
            }
        )

        # Conditionally add cycle and round
        if cycle is not None:
            metadata["cycle"] = cycle
        if round is not None:
            metadata["round"] = round

        pixel_size_x, pixel_size_y, pixel_size_z, objective_mag, zoom_mag = (
            _extract_pixel_calibration(frame_meta, file_path)
        )
        binning_xy = _parse_binning_from_nd2(file_path)
        _validate_camera_pixel_size(
            pixel_size_x, objective_mag, zoom_mag, binning_xy, file_path
        )

        metadata.update(
            {
                "pixel_size_x": pixel_size_x,
                "pixel_size_y": pixel_size_y,
                "pixel_size_z": pixel_size_z,
                "objective_magnification": objective_mag,
                "zoom_magnification": zoom_mag,
                "binning_xy": binning_xy,
            }
        )

        df = pd.DataFrame([metadata])

    return df


def extract_metadata_well_nd2(
    file_path: str,
    plate: Union[int, str],
    well: Union[int, str],
    cycle: Union[int, str] = None,
    round: Union[int, str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract metadata from well-based ND2 file containing multiple positions.

    Well-based organization means a single file contains multiple fields of view
    (positions) from the same well. This function extracts metadata for all
    positions, creating one row per position.

    Args:
        file_path: Path to the ND2 file containing multiple positions
        plate: Plate identifier
        well: Well identifier (e.g., 'A01', 'B12')
        cycle: Optional cycle number for SBS imaging
        round: Optional round number for multiplexed imaging
        verbose: Print debug information

    Returns:
        DataFrame with one row per position/tile in the well
    """
    metadata_rows = []
    if verbose:
        print(f"Processing well file: {file_path}")

    with nd2.ND2File(file_path) as images:
        if verbose:
            print(f"File shape: {images.shape}")
            print(f"Number of dimensions: {images.ndim}")
            print(f"Data type: {images.dtype}")
            print(f"Sizes (by axes): {images.sizes}")

        # Get number of unique XY positions
        num_positions = images.sizes.get("P", 1)
        z_planes = images.sizes.get("Z", 1)

        if verbose:
            print(f"Number of positions: {num_positions}")
            print(f"Number of Z planes: {z_planes}")

        binning_xy = _parse_binning_from_nd2(file_path)

        # Only process one frame per unique XY position
        for pos_idx in range(0, num_positions * z_planes, z_planes):
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
                    "plate": plate,
                    "well": well,
                    "tile": pos_idx // z_planes,  # Adjust tile number based on z_planes
                    "filename": file_path,
                    "channels": images.sizes.get("C", 1),
                }
            )

            # Conditionally add cycle and round
            if cycle is not None:
                metadata["cycle"] = cycle
            if round is not None:
                metadata["round"] = round

            pixel_size_x, pixel_size_y, pixel_size_z, objective_mag, zoom_mag = (
                _extract_pixel_calibration(frame_meta, file_path, pos_idx=pos_idx)
            )
            _validate_camera_pixel_size(
                pixel_size_x,
                objective_mag,
                zoom_mag,
                binning_xy,
                file_path,
                pos_idx=pos_idx,
            )

            metadata.update(
                {
                    "pixel_size_x": pixel_size_x,
                    "pixel_size_y": pixel_size_y,
                    "pixel_size_z": pixel_size_z,
                    "objective_magnification": objective_mag,
                    "zoom_magnification": zoom_mag,
                    "binning_xy": binning_xy,
                }
            )

            metadata_rows.append(metadata)

    df = pd.DataFrame(metadata_rows)
    return df


def extract_metadata_tiff(
    file_path: str,
    plate: Union[int, str],
    well: Union[int, str],
    tile: Union[int, str],
    cycle: Union[int, str] = None,
    round: Union[int, str] = None,
    metadata_file_path: str = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract metadata for TIFF files using external metadata file.

    TIFF files don't contain position metadata in their headers, so this function
    reads from an external CSV/TSV file containing stage positions and other metadata.
    The function handles various column naming conventions and unit conversions.

    Args:
        file_path: Path to the TIFF file
        plate: Plate identifier
        well: Well identifier
        tile: Tile/FOV number
        cycle: Optional cycle number
        round: Optional round number
        metadata_file_path: Path to CSV/TSV with position metadata
        verbose: Print debug information

    Returns:
        DataFrame with metadata, either from external file or with null values
    """
    if metadata_file_path and Path(metadata_file_path).exists():
        if verbose:
            print(f"Reading metadata from: {metadata_file_path}")

        try:
            # Read metadata file
            if metadata_file_path.endswith(".csv"):
                metadata_df = pd.read_csv(metadata_file_path)
            else:  # assume TSV
                metadata_df = pd.read_csv(metadata_file_path, sep="\t")

            if verbose:
                print(f"Columns: {list(metadata_df.columns)}")

            # Define flexible column mappings for different coordinate file formats
            column_mappings = {
                # Well/region mappings
                "well": [
                    "well",
                    "Well",
                    "WELL",
                    "region",
                    "Region",
                    "REGION",
                    "site_name",
                    "Site_Name",
                ],
                # Tile/position mappings
                "tile": [
                    "tile",
                    "Tile",
                    "TILE",
                    "fov",
                    "FOV",
                    "field",
                    "Field",
                    "position",
                    "Position",
                    "site",
                    "Site",
                ],
                # X coordinate mappings
                "x_pos": [
                    "x_pos",
                    "X_pos",
                    "x (mm)",
                    "X (mm)",
                    "x(mm)",
                    "X(mm)",
                    "x_mm",
                    "X_mm",
                    "x",
                    "X",
                    "stage_x",
                    "Stage_X",
                ],
                # Y coordinate mappings
                "y_pos": [
                    "y_pos",
                    "Y_pos",
                    "y (mm)",
                    "Y (mm)",
                    "y(mm)",
                    "Y(mm)",
                    "y_mm",
                    "Y_mm",
                    "y",
                    "Y",
                    "stage_y",
                    "Stage_Y",
                ],
                # Z coordinate mappings
                "z_pos": [
                    "z_pos",
                    "Z_pos",
                    "z (um)",
                    "Z (um)",
                    "z(um)",
                    "Z(um)",
                    "z_um",
                    "Z_um",
                    "z",
                    "Z",
                    "stage_z",
                    "Stage_Z",
                ],
                # Other metadata mappings
                "pixel_size_x": ["pixel_size_x", "PixelSizeX", "pixel_x", "Pixel_X"],
                "pixel_size_y": ["pixel_size_y", "PixelSizeY", "pixel_y", "Pixel_Y"],
                "channels": ["channels", "Channels", "channel_count", "Channel_Count"],
                "pfs_offset": ["pfs_offset", "PFS_offset", "pfs", "PFS"],
            }

            def find_column(target_name: str) -> Optional[str]:
                """Find the actual column name in the dataframe for a target metadata field."""
                possible_names = column_mappings.get(target_name, [target_name])
                for name in possible_names:
                    if name in metadata_df.columns:
                        return name
                return None

            # Convert entire dataframe to standardized format
            metadata_rows = []

            for idx, row in metadata_df.iterrows():
                # Extract coordinates with unit conversion
                x_col = find_column("x_pos")
                y_col = find_column("y_pos")
                z_col = find_column("z_pos")

                # Handle coordinate conversion based on likely units
                x_pos = None
                y_pos = None
                z_pos = None

                if x_col:
                    x_val = row[x_col]
                    if pd.notna(x_val):
                        # Convert based on column name hints
                        if "mm" in x_col.lower() or ("(" in x_col and "mm" in x_col):
                            x_pos = float(x_val) * 1000  # mm to μm
                        elif "um" in x_col.lower() or ("(" in x_col and "um" in x_col):
                            x_pos = float(x_val)  # already in μm
                        else:
                            x_pos = float(x_val)  # assume μm

                if y_col:
                    y_val = row[y_col]
                    if pd.notna(y_val):
                        if "mm" in y_col.lower() or ("(" in y_col and "mm" in y_col):
                            y_pos = float(y_val) * 1000  # mm to μm
                        elif "um" in y_col.lower() or ("(" in y_col and "um" in y_col):
                            y_pos = float(y_val)  # already in μm
                        else:
                            y_pos = float(y_val)  # assume μm

                if z_col:
                    z_val = row[z_col]
                    if pd.notna(z_val):
                        if "mm" in z_col.lower() or ("(" in z_col and "mm" in z_col):
                            z_pos = float(z_val) * 1000  # mm to μm
                        elif "um" in z_col.lower() or ("(" in z_col and "um" in z_col):
                            z_pos = float(z_val)  # already in μm
                        else:
                            z_pos = float(z_val)  # assume μm

                # Extract well and tile information
                well_col = find_column("well")
                tile_col = find_column("tile")

                extracted_well = row[well_col] if well_col else well
                extracted_tile = row[tile_col] if tile_col else idx

                # Extract other metadata fields
                pixel_x_col = find_column("pixel_size_x")
                pixel_y_col = find_column("pixel_size_y")
                channels_col = find_column("channels")
                pfs_col = find_column("pfs_offset")

                pixel_size_x = (
                    row[pixel_x_col]
                    if pixel_x_col and pd.notna(row[pixel_x_col])
                    else None
                )
                pixel_size_y = (
                    row[pixel_y_col]
                    if pixel_y_col and pd.notna(row[pixel_y_col])
                    else None
                )
                channels = (
                    row[channels_col]
                    if channels_col and pd.notna(row[channels_col])
                    else None
                )
                pfs_offset = (
                    row[pfs_col] if pfs_col and pd.notna(row[pfs_col]) else None
                )

                # Build standardized metadata row
                metadata = {
                    "plate": plate,
                    "well": extracted_well,
                    "tile": extracted_tile,
                    "filename": file_path,
                    "x_pos": x_pos,
                    "y_pos": y_pos,
                    "z_pos": z_pos,
                    "pfs_offset": pfs_offset,
                    "channels": channels,
                    "pixel_size_x": pixel_size_x,
                    "pixel_size_y": pixel_size_y,
                }

                # Add cycle and round if provided
                if cycle is not None:
                    metadata["cycle"] = cycle
                if round is not None:
                    metadata["round"] = round

                metadata_rows.append(metadata)

            result_df = pd.DataFrame(metadata_rows)

            if verbose:
                print(f"Converted {len(result_df)} rows of metadata")

            return result_df

        except Exception as e:
            if verbose:
                print(f"Error reading metadata file {metadata_file_path}: {e}")

    # Fallback: create basic metadata
    if verbose:
        print("Using fallback metadata (no external file found)")

    metadata = {
        "plate": plate,
        "well": well,
        "tile": tile,
        "filename": file_path,
        "x_pos": None,
        "y_pos": None,
        "z_pos": None,
        "pfs_offset": None,
        "channels": None,
        "pixel_size_x": None,
        "pixel_size_y": None,
    }

    # Add cycle and round if provided
    if cycle is not None:
        metadata["cycle"] = cycle
    if round is not None:
        metadata["round"] = round

    return pd.DataFrame([metadata])


def convert_nd2_to_array_tile(
    files: Union[str, List[str], Path, List[Path]],
    channel_order_flip: bool = False,
    verbose: bool = False,
    n_z_planes: int = None,
    preserve_z: bool = False,
) -> np.ndarray:
    """Convert tile-based ND2 files to numpy array in CYX or CZYX format.

    Processes one or more ND2 files where each file contains a single FOV.
    If multiple files are provided, they are concatenated along the channel axis.
    Z-stacks are handled by maximum intensity projection unless preserve_z is True.

    Note: n_z_planes parameter is accepted for API compatibility but not used,
    as Z-stack handling is automatic via maximum intensity projection.

    Args:
        files: Path(s) to ND2 file(s)
        channel_order_flip: Reverse the order of channels
        verbose: Print debug information
        n_z_planes: Accepted for API compatibility but not used (Z-stack handled automatically)
        preserve_z: If True, preserves Z dimension (returning CZYX).

    Returns:
        numpy array in CYX or CZYX format with dtype uint16
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
            if not preserve_z:
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


def convert_nd2_to_array_well(
    files: Union[str, List[str], Path, List[Path]],
    position: int,
    channel_order_flip: bool = False,
    return_tiles: bool = False,
    verbose: bool = False,
    n_z_planes: int = None,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Extract specific position from well-based ND2 files.

    Processes ND2 files containing multiple positions/FOVs and extracts
    a specific position. Handles Z-stacks via maximum intensity projection.

    Args:
        files: Path(s) to ND2 file(s) containing multiple positions
        position: Position index to extract (0-based)
        channel_order_flip: Reverse the order of channels
        return_tiles: If True, also return the total number of tiles
        verbose: Print debug information
        n_z_planes: Accepted for API compatibility but not used (Z-stack handled automatically)

    Returns:
        numpy array in CYX format, optionally with tile count

    Example:
        >>> # Extract position 5 from well file
        >>> img = convert_nd2_to_array_well("well_A01.nd2", position=5)
        >>> img.shape  # (4, 2048, 2048)
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


def convert_tiff_to_array(
    files: Union[str, List[str], Path, List[Path]],
    channel_order_flip: bool = False,
    n_z_planes: Optional[int] = None,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Convert TIFF files to numpy array in CYX format.

    Handles both regular TIFF files and z-split TIFF inputs where multiple
    files represent different z-planes of the same position.

    Args:
        files: Path(s) to TIFF file(s)
        channel_order_flip: Reverse the order of channels
        n_z_planes: Number of z-planes per channel. If provided, files will be grouped
                    into chunks of n_z_planes, each chunk stacked along Z and max projected.
                    For example, n_z_planes=2 with 8 files creates 4 channels (8/2=4).
                    If None, files are concatenated along channel axis (normal behavior).
        verbose: Print debug information
        **kwargs: Additional arguments (for compatibility, including deprecated z_stack)

    Returns:
        numpy array in CYX format with dtype uint16

    Notes:
        - If n_z_planes specified: groups files, stacks each group along Z, max projects to CYX
        - If n_z_planes is None: files are concatenated along channel axis (normal behavior)
        - Files should be ordered: [ch0_z0, ch0_z1, ..., ch1_z0, ch1_z1, ..., chN_zM]
    """
    try:
        from tifffile import imread
    except ImportError:
        raise ImportError("tifffile package required for TIFF support")

    # Convert input to list of Path objects
    if isinstance(files, (str, Path)):
        files = [Path(files)]
    else:
        files = [Path(f) for f in files]

    # Validate n_z_planes
    if n_z_planes is not None and n_z_planes > 0:
        if len(files) % n_z_planes != 0:
            raise ValueError(
                f"Number of files ({len(files)}) must be divisible by n_z_planes ({n_z_planes}). "
                f"Got {len(files) // n_z_planes} complete channels and {len(files) % n_z_planes} remaining files."
            )
        n_channels = len(files) // n_z_planes
        if verbose:
            print(
                f"Z-aware stacking: {n_channels} channels × {n_z_planes} z-planes = {len(files)} files"
            )

    # Read all TIFF files
    image_arrays = []
    for i, file in enumerate(files, 1):
        if verbose:
            print(f"Processing TIFF file {i}/{len(files)}: {file}")

        # Read TIFF file
        img_array = imread(str(file))

        # Ensure we have CYX format (even if single channel, should be (1, Y, X))
        if img_array.ndim == 2:
            # Add channel dimension for grayscale
            img_array = np.expand_dims(img_array, axis=0)
        elif img_array.ndim == 3:
            # Assume it's already in CYX format
            pass

        # Flip channel order if needed (only applies to multi-channel files)
        if channel_order_flip and img_array.shape[0] > 1:
            img_array = np.flip(img_array, axis=0)

        if verbose:
            print(f"TIFF array shape: {img_array.shape}")

        image_arrays.append(img_array)

    # Handle z-aware stacking for multi-channel z-stacks
    if n_z_planes is not None and n_z_planes > 1 and len(image_arrays) > 1:
        if verbose:
            print(
                f"\nZ-aware stacking: Grouping {len(image_arrays)} files into channels of {n_z_planes} z-planes each"
            )

        channel_results = []
        for ch_idx in range(n_channels):
            # Get z-planes for this channel
            start_idx = ch_idx * n_z_planes
            end_idx = start_idx + n_z_planes
            z_planes = image_arrays[start_idx:end_idx]

            if verbose:
                print(
                    f"  Channel {ch_idx + 1}/{n_channels}: Stacking z-planes {start_idx} to {end_idx - 1}"
                )

            # Each z_plane has shape (1, Y, X) for single-channel TIFFs
            # Stack them along Z axis: (1, Y, X) × n_z_planes → (1, n_z_planes, Y, X)
            stacked = np.stack(z_planes, axis=1)  # Insert Z after C

            if verbose:
                print(f"    Stacked shape (CZYX): {stacked.shape}")

            # Max project along Z axis: (1, n_z_planes, Y, X) → (1, Y, X)
            max_projected = np.max(stacked, axis=1)

            if verbose:
                print(f"    After max projection (CYX): {max_projected.shape}")

            channel_results.append(max_projected)

        # Concatenate all channels: [(1, Y, X), (1, Y, X), ...] → (n_channels, Y, X)
        result = np.concatenate(channel_results, axis=0)

        if verbose:
            print(f"\nFinal concatenated result (CYX): {result.shape}")

    # Handle deprecated z_stack=True behavior (all files as z-planes of one channel)
    elif n_z_planes == -1 and len(image_arrays) > 1:
        if verbose:
            print(
                f"Deprecated z_stack mode: Stacking {len(image_arrays)} z-planes as single channel"
            )
        stacked = np.stack(image_arrays, axis=1)
        result = np.max(stacked, axis=1)
        if verbose:
            print(f"After max projection (CYX): {result.shape}")

    # Normal behavior: concatenate along channel axis
    else:
        if len(image_arrays) == 1:
            result = image_arrays[0]
        else:
            result = np.concatenate(image_arrays, axis=0)

        if verbose and len(image_arrays) > 1:
            print(
                f"Concatenated {len(image_arrays)} files along channel axis: {result.shape}"
            )

    return result.astype(np.uint16)


def extract_metadata(
    file_paths: Union[str, List[str]],
    plate: Union[int, str],
    well: Union[int, str],
    tile: Union[int, str] = None,
    cycle: Union[int, str] = None,
    round: Union[int, str] = None,
    data_format: str = "nd2",
    data_organization: str = "tile",
    metadata_file_path: str = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract metadata from microscopy files.

    Main entry point for metadata extraction that dispatches to appropriate
    implementation based on data format and organization.

    Args:
        file_paths: Path(s) to the file(s)
        plate: Plate identifier
        well: Well identifier
        tile: Tile number (required for tile organization, ignored for well)
        cycle: Optional cycle number for SBS imaging
        round: Optional round number for multiplexed imaging
        data_format: 'nd2' or 'tiff'
        data_organization: 'tile' (one FOV per file) or 'well' (multiple FOVs per file)
        metadata_file_path: Path to external metadata CSV/TSV (for TIFF)
        verbose: Print debug information

    Returns:
        DataFrame with extracted metadata

    Examples:
        >>> # Tile-based ND2
        >>> df = extract_metadata("tile_001.nd2", plate=1, well="A01", tile=1,
        ...                      data_format="nd2", data_organization="tile")

        >>> # Well-based ND2 (extracts all positions)
        >>> df = extract_metadata("well_A01.nd2", plate=1, well="A01",
        ...                      data_format="nd2", data_organization="well")

        >>> # TIFF with external metadata
        >>> df = extract_metadata("image.tiff", plate=1, well="A01", tile=1,
        ...                      data_format="tiff", metadata_file_path="positions.csv")
    """
    if data_format not in DATA_FORMATS:
        raise ValueError(f"Unsupported data format: {data_format}")
    if data_organization not in DATA_ORGANIZATIONS:
        raise ValueError(f"Unsupported data organization: {data_organization}")

    if isinstance(file_paths, str):
        file_paths = [file_paths]

    if data_format == "nd2":
        if data_organization == "tile":
            # For tile organization, process each file separately
            metadata_dfs = []
            for i, file_path in enumerate(file_paths):
                current_tile = tile if tile is not None else i

                # Only pass parameters that are not None
                kwargs = {
                    "file_path": file_path,
                    "plate": plate,
                    "well": well,
                    "tile": current_tile,
                    "verbose": verbose,
                }

                if cycle is not None:
                    kwargs["cycle"] = cycle
                if round is not None:
                    kwargs["round"] = round

                df = extract_metadata_tile_nd2(**kwargs)
                metadata_dfs.append(df)
            return pd.concat(metadata_dfs, ignore_index=True)

        elif data_organization == "well":
            # For well organization, process the first file (should be only one)
            kwargs = {
                "file_path": file_paths[0],
                "plate": plate,
                "well": well,
                "verbose": verbose,
            }

            if cycle is not None:
                kwargs["cycle"] = cycle
            if round is not None:
                kwargs["round"] = round

            return extract_metadata_well_nd2(**kwargs)

    elif data_format == "tiff":
        # For TIFF, always treat as tile-based for now
        metadata_dfs = []
        for i, file_path in enumerate(file_paths):
            current_tile = tile if tile is not None else i

            # Only pass parameters that are not None
            kwargs = {
                "file_path": file_path,
                "plate": plate,
                "well": well,
                "tile": current_tile,
                "metadata_file_path": metadata_file_path,
                "verbose": verbose,
            }

            if cycle is not None:
                kwargs["cycle"] = cycle
            if round is not None:
                kwargs["round"] = round

            df = extract_metadata_tiff(**kwargs)
            metadata_dfs.append(df)
        return pd.concat(metadata_dfs, ignore_index=True)


def convert_to_array(
    files: Union[str, List[str], Path, List[Path]],
    data_format: str = "nd2",
    data_organization: str = "tile",
    position: int = None,
    channel_order_flip: bool = False,
    preserve_z: bool = False,
    verbose: bool = False,
    **kwargs,
) -> np.ndarray:
    """Convert microscopy image files to numpy array in CYX or CZYX format.

    Main entry point for image conversion that dispatches to appropriate
    implementation based on data format and organization. The output is always
    a numpy array in CYX format (Channel, Y, X) for consistent downstream processing,
    unless preserve_z=True (tile-based ND2 only) in which case CZYX is returned.

    Args:
        files: Path(s) to image file(s)
        data_format: 'nd2' or 'tiff'
        data_organization: 'tile' (one FOV per file) or 'well' (multiple FOVs per file)
        position: Position/tile to extract (required for well organization)
        channel_order_flip: Reverse the order of channels
        preserve_z: If True (tile-based ND2), preserve Z planes (returning CZYX)
        verbose: Print debug information
        **kwargs: Additional arguments passed to specific converters

    Returns:
        numpy array in CYX format with dtype uint16

    Examples:
        >>> # Tile-based ND2
        >>> img = convert_to_array("tile.nd2", data_format="nd2",
        ...                       data_organization="tile")
        >>> img.shape  # (4, 2048, 2048) for 4-channel image

        >>> # Well-based ND2, extract position 3
        >>> img = convert_to_array("well.nd2", data_format="nd2",
        ...                       data_organization="well", position=3)

        >>> # Multiple TIFF files concatenated
        >>> img = convert_to_array(["ch1.tiff", "ch2.tiff"], data_format="tiff")
    """
    if data_format not in DATA_FORMATS:
        raise ValueError(f"Unsupported data format: {data_format}")
    if data_organization not in DATA_ORGANIZATIONS:
        raise ValueError(f"Unsupported data organization: {data_organization}")

    if data_format == "nd2":
        if data_organization == "tile":
            return convert_nd2_to_array_tile(
                files,
                channel_order_flip=channel_order_flip,
                verbose=verbose,
                preserve_z=preserve_z,
            )
        elif data_organization == "well":
            if position is None:
                raise ValueError("Position must be specified for well organization")
            return convert_nd2_to_array_well(
                files, position, channel_order_flip, verbose=verbose, **kwargs
            )

    elif data_format == "tiff":
        return convert_tiff_to_array(
            files, channel_order_flip=channel_order_flip, verbose=verbose, **kwargs
        )


# Helper functions for Snakemake integration
def get_expansion_values(
    image_type: str, config: dict, metadata_wildcard_combos: pd.DataFrame = None
) -> List[str]:
    """Get expansion values for metadata combination based on data organization and actual metadata structure.

    Used by Snakemake to determine which wildcards need expansion when combining
    metadata files.

    Args:
        image_type: 'sbs' or 'phenotype'
        config: Configuration dictionary
        metadata_wildcard_combos: DataFrame with actual metadata wildcard combinations

    Returns:
        List of wildcard names to expand
    """
    data_config = get_data_config(image_type, config)

    # Use image data organization (tiles exist only for images, not metadata)
    image_org = data_config.get("image_data_organization", "tile")

    # Base expansion values based on organization
    if image_org == "tile":
        if image_type == "sbs":
            base_expansion = ["tile", "cycle"]
        else:  # phenotype
            base_expansion = ["tile"]
    else:  # well organization
        if image_type == "sbs":
            base_expansion = ["cycle"]
        else:  # phenotype
            base_expansion = []  # No expansion needed for well-based phenotype

    # If we have metadata wildcard combinations, check for additional columns
    if metadata_wildcard_combos is not None and len(metadata_wildcard_combos) > 0:
        metadata_columns = list(metadata_wildcard_combos.columns)

        # Add any columns that exist in metadata but not in base expansion
        # Exclude 'plate' and 'well' as these are typically not expanded
        exclude_columns = {"plate", "well"}
        additional_columns = [
            col
            for col in metadata_columns
            if col not in base_expansion and col not in exclude_columns
        ]

        # Combine base expansion with additional columns
        expansion_values = base_expansion + additional_columns

        # Remove duplicates while preserving order
        seen = set()
        expansion_values = [
            x for x in expansion_values if not (x in seen or seen.add(x))
        ]

        return expansion_values

    return base_expansion


def include_tile_in_input(
    image_type: str, config: dict, for_metadata: bool = False
) -> bool:
    """Determine if tile should be included in input file selection.

    Used by Snakemake rules to decide whether to filter by tile when
    selecting input files.

    Args:
        image_type: 'sbs' or 'phenotype'
        config: Configuration dictionary
        for_metadata: If True, check metadata data organization; else image data organization

    Returns:
        True if tile should be included in file selection
    """
    data_config = get_data_config(image_type, config)
    key = "metadata_data_organization" if for_metadata else "image_data_organization"
    return data_config[key] == "tile"


def update_config_for_unified_processing(config: dict) -> dict:
    """Update existing config to work with unified preprocessing.

    Sets default values for new configuration options to ensure backwards
    compatibility with existing configs.

    Args:
        config: Original configuration dictionary

    Returns:
        Updated configuration dictionary with defaults
    """
    preprocess_config = config.get("preprocess", {})

    # Set default values for new configuration options
    defaults = {
        "sbs_data_format": "nd2",
        "sbs_data_organization": "tile",
        "phenotype_data_format": "nd2",
        "phenotype_data_organization": "well",
    }

    for key, default_value in defaults.items():
        if key not in preprocess_config:
            preprocess_config[key] = default_value

    config["preprocess"] = preprocess_config
    return config


# --- Private helpers (deduplicated from extract_metadata_*_nd2) ---


def _parse_binning_from_nd2(file_path: str) -> Optional[str]:
    """Parse camera binning from ND2 text_info metadata (best-effort).

    Args:
        file_path: Path to the ND2 file.

    Returns:
        Binning string like '2x2', or None if not found.
    """
    try:
        img = nd2.imread(str(file_path), xarray=True)
        md = img.attrs.get("metadata", {})
        text_info = md.get("text_info", {})
        desc = text_info.get("description", "") if isinstance(text_info, dict) else ""
        m = re.search(r"Binning:\\s*(\\d+)x(\\d+)", desc)
        if m:
            return f"{m.group(1)}x{m.group(2)}"
    except Exception:
        pass
    return None


def _extract_pixel_calibration(
    frame_meta, file_path: str, *, pos_idx: Optional[int] = None
) -> Tuple[
    Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]
]:
    """Extract pixel calibration and optics metadata from an ND2 frame.

    Args:
        frame_meta: Frame metadata object from ``ND2File.frame_metadata()``.
        file_path: Path to the ND2 file (used in warning messages).
        pos_idx: Optional position index (used in warning messages for well files).

    Returns:
        Tuple of (pixel_size_x, pixel_size_y, pixel_size_z,
                  objective_magnification, zoom_magnification).
        Any value may be None if not available.
    """
    pixel_size_x = pixel_size_y = pixel_size_z = None
    objective_mag = zoom_mag = None

    if not frame_meta.channels:
        return pixel_size_x, pixel_size_y, pixel_size_z, objective_mag, zoom_mag

    ch0 = frame_meta.channels[0]

    if hasattr(ch0, "volume") and getattr(ch0.volume, "axesCalibration", None):
        x_cal, y_cal, z_cal = ch0.volume.axesCalibration
        pixel_size_x = x_cal
        pixel_size_y = y_cal
        pixel_size_z = z_cal

        try:
            if (
                pixel_size_x is not None
                and pixel_size_y is not None
                and abs(float(pixel_size_x) - float(pixel_size_y)) > 1e-6
            ):
                pos_suffix = f" pos {pos_idx}" if pos_idx is not None else ""
                warnings.warn(
                    f"pixel_size_x ({pixel_size_x}) != pixel_size_y ({pixel_size_y}) "
                    f"for {file_path}{pos_suffix}"
                )
        except Exception:
            pass

    if hasattr(ch0, "microscope") and ch0.microscope is not None:
        objective_mag = getattr(ch0.microscope, "objectiveMagnification", None)
        zoom_mag = getattr(ch0.microscope, "zoomMagnification", None)

    return pixel_size_x, pixel_size_y, pixel_size_z, objective_mag, zoom_mag


def _validate_camera_pixel_size(
    pixel_size_x: Optional[float],
    objective_mag: Optional[float],
    zoom_mag: Optional[float],
    binning_xy: Optional[str],
    file_path: str,
    *,
    pos_idx: Optional[int] = None,
) -> None:
    """Warn if the estimated camera pixel size is implausible.

    Typical scientific camera pixel sizes are ~4.5–11 µm.

    Args:
        pixel_size_x: Calibrated pixel size in X (µm).
        objective_mag: Objective magnification.
        zoom_mag: Zoom magnification.
        binning_xy: Binning string (e.g. '2x2').
        file_path: Path to ND2 file (for warning message).
        pos_idx: Optional position index (for warning message).
    """
    try:
        if pixel_size_x is not None and objective_mag and zoom_mag and binning_xy:
            bx, by = binning_xy.split("x")
            bx = float(bx)
            by = float(by)
            if abs(bx - by) < 1e-6 and bx > 0:
                camera_px_um_est = (
                    float(pixel_size_x) * float(objective_mag) * float(zoom_mag) * bx
                )
                if camera_px_um_est < 3.0 or camera_px_um_est > 15.0:
                    pos_suffix = f" pos {pos_idx}" if pos_idx is not None else ""
                    warnings.warn(
                        f"Implausible camera pixel estimate {camera_px_um_est:.3f} µm "
                        f"(pixel_size_x={pixel_size_x}, objective={objective_mag}, "
                        f"zoom={zoom_mag}, binning={binning_xy}) "
                        f"for {file_path}{pos_suffix}"
                    )
    except Exception:
        pass
