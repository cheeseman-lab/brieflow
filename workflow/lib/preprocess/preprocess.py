"""
Unified preprocessing system using functional approach.
"""

import pandas as pd
import numpy as np
import nd2
from typing import Union, List, Dict, Any, Optional, Tuple
from pathlib import Path
import warnings
import gc


# Data organization and format constants
DATA_FORMATS = {"nd2", "tiff"}
DATA_ORGANIZATIONS = {"tile", "well"}


def get_data_config(image_type: str, config: Dict[str, Any]) -> Dict[str, str]:
    """Get data configuration for a specific image type (sbs/phenotype)."""
    base_config = config.get("preprocess", {})
    
    return {
        "data_format": base_config.get(f"{image_type}_data_format", "nd2"),
        "data_organization": base_config.get(f"{image_type}_data_organization", "tile"),
        "channel_order_flip": base_config.get(f"{image_type}_channel_order_flip", False),
        "channel_order": base_config.get(f"{image_type}_channel_order", None),
    }


def extract_metadata_tile_nd2(
    file_path: str,
    plate: Union[int, str],
    well: Union[int, str],
    tile: Union[int, str],
    cycle: Union[int, str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract metadata from a single ND2 tile file."""
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
        metadata.update({
            "plate": plate,
            "well": well,
            "tile": tile,
            "filename": file_path,
            "channels": frame_meta.contents.channelCount,
        })

        # Conditionally add cycle after tile
        if cycle is not None:
            metadata["cycle"] = cycle

        # Get pixel size from first channel's volume information
        if frame_meta.channels and hasattr(frame_meta.channels[0], "volume"):
            x_cal, y_cal, _ = frame_meta.channels[0].volume.axesCalibration
            metadata.update({
                "pixel_size_x": x_cal,
                "pixel_size_y": y_cal,
            })
        else:
            metadata.update({
                "pixel_size_x": None,
                "pixel_size_y": None,
            })

        df = pd.DataFrame([metadata])

    return df


def extract_metadata_well_nd2(
    file_path: str,
    plate: Union[int, str],
    well: Union[int, str],
    cycle: Union[int, str] = None,
    verbose: bool = False
) -> pd.DataFrame:
    """Extract metadata from well ND2 file containing multiple positions."""
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
            metadata.update({
                "plate": plate,
                "well": well,
                "tile": pos_idx // z_planes,  # Adjust tile number based on z_planes
                "filename": file_path,
                "channels": images.sizes.get("C", 1),
            })

            # Conditionally add cycle
            if cycle is not None:
                metadata["cycle"] = cycle

            # Get pixel calibration if available
            if frame_meta.channels and hasattr(frame_meta.channels[0], "volume"):
                x_cal, y_cal, *_ = frame_meta.channels[0].volume.axesCalibration
                metadata.update({
                    "pixel_size_x": x_cal,
                    "pixel_size_y": y_cal,
                })
            else:
                metadata.update({
                    "pixel_size_x": None,
                    "pixel_size_y": None,
                })

            metadata_rows.append(metadata)

    df = pd.DataFrame(metadata_rows)
    return df


def extract_metadata_tiff(
    file_path: str,
    plate: Union[int, str],
    well: Union[int, str],
    tile: Union[int, str],
    cycle: Union[int, str] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Extract metadata from TIFF file (placeholder for future implementation)."""
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
    
    if cycle is not None:
        metadata["cycle"] = cycle
    
    return pd.DataFrame([metadata])


def convert_nd2_to_array_tile(
    files: Union[str, List[str], Path, List[Path]],
    channel_order_flip: bool = False,
    verbose: bool = False,
) -> np.ndarray:
    """Convert ND2 tile files to array."""
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


def convert_nd2_to_array_well(
    files: Union[str, List[str], Path, List[Path]],
    position: int,
    channel_order_flip: bool = False,
    return_tiles: bool = False,
    verbose: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, int]]:
    """Extract specific position from well ND2 files and convert to array."""
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


def convert_tiff_to_array(
    files: Union[str, List[str], Path, List[Path]],
    **kwargs
) -> np.ndarray:
    """Convert TIFF files to array (placeholder for future implementation)."""
    # Placeholder for TIFF conversion
    pass


# Main unified functions that dispatch to the appropriate implementation
def extract_metadata_unified(
    file_paths: Union[str, List[str]],
    plate: Union[int, str],
    well: Union[int, str],
    tile: Union[int, str] = None,
    cycle: Union[int, str] = None,
    data_format: str = "nd2",
    data_organization: str = "tile",
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Unified metadata extraction function that dispatches to appropriate implementation.
    
    Args:
        file_paths: Path(s) to the file(s)
        plate: Plate number
        well: Well identifier
        tile: Tile number (used differently based on organization)
        cycle: Cycle number (optional)
        data_format: "nd2" or "tiff"
        data_organization: "tile" or "well"
        verbose: Print debug information
    
    Returns:
        DataFrame with extracted metadata
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
                df = extract_metadata_tile_nd2(
                    file_path, plate, well, current_tile, cycle, verbose
                )
                metadata_dfs.append(df)
            return pd.concat(metadata_dfs, ignore_index=True)
        
        elif data_organization == "well":
            # For well organization, process the first file (should be only one)
            return extract_metadata_well_nd2(file_paths[0], plate, well, cycle, verbose)
    
    elif data_format == "tiff":
        # For TIFF, always treat as tile-based for now
        metadata_dfs = []
        for i, file_path in enumerate(file_paths):
            current_tile = tile if tile is not None else i
            df = extract_metadata_tiff(
                file_path, plate, well, current_tile, cycle, verbose
            )
            metadata_dfs.append(df)
        return pd.concat(metadata_dfs, ignore_index=True)


def convert_to_array_unified(
    files: Union[str, List[str], Path, List[Path]],
    data_format: str = "nd2",
    data_organization: str = "tile",
    position: int = None,
    channel_order_flip: bool = False,
    verbose: bool = False,
    **kwargs
) -> np.ndarray:
    """
    Unified conversion function that dispatches to appropriate implementation.
    
    Args:
        files: Path(s) to the file(s)
        data_format: "nd2" or "tiff"
        data_organization: "tile" or "well"
        position: Position/tile to extract (for well organization)
        channel_order_flip: Whether to flip channel order
        verbose: Print debug information
        **kwargs: Additional arguments passed to specific converters
    
    Returns:
        numpy array in CYX format
    """
    if data_format not in DATA_FORMATS:
        raise ValueError(f"Unsupported data format: {data_format}")
    if data_organization not in DATA_ORGANIZATIONS:
        raise ValueError(f"Unsupported data organization: {data_organization}")
    
    if data_format == "nd2":
        if data_organization == "tile":
            return convert_nd2_to_array_tile(files, channel_order_flip, verbose)
        elif data_organization == "well":
            if position is None:
                raise ValueError("Position must be specified for well organization")
            return convert_nd2_to_array_well(
                files, position, channel_order_flip, verbose=verbose, **kwargs
            )
    
    elif data_format == "tiff":
        return convert_tiff_to_array(files, **kwargs)


# Helper functions for Snakemake integration
def get_expansion_values(image_type: str, config: dict) -> List[str]:
    """Get expansion values for metadata combination based on data organization."""
    data_config = get_data_config(image_type, config)
    
    if data_config["data_organization"] == "tile":
        if image_type == "sbs":
            return ["tile", "cycle"]
        else:  # phenotype
            return ["tile"]
    else:  # well organization
        if image_type == "sbs":
            return ["cycle"]
        else:  # phenotype
            return []  # No expansion needed for well-based phenotype


def should_include_tile_in_input(image_type: str, config: dict) -> bool:
    """Determine if tile should be included in input file selection."""
    data_config = get_data_config(image_type, config)
    return data_config["data_organization"] == "tile"


def update_config_for_unified_processing(config: dict) -> dict:
    """
    Update existing config to work with unified processing.
    Sets default values for new configuration options.
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
