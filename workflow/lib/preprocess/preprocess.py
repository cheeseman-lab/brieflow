"""Functions for preprocessing ND2 files in preparation for downstream BrieFlow steps."""

import pandas as pd
import numpy as np
import nd2

from lib.preprocess.file_utils import extract_tile_from_filename


def extract_tile_metadata(
    tile_fp: str, tile: int, z_interval: int = 4, verbose: bool = True
) -> pd.DataFrame:
    """Extracts metadata from a single ND2 file for a specific tile.

    Args:
        tile_fp (str): File path pointing to the ND2 file for the tile.
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

        # Get position data from first channel's position information
        if frame_meta.channels and hasattr(frame_meta.channels[0], "position"):
            stage_pos = frame_meta.channels[0].position.stagePositionUm
            metadata = {
                "x_data": [stage_pos.x],
                "y_data": [stage_pos.y],
                "z_data": [stage_pos.z],
                "pfs_offset": frame_meta.channels[0].position.pfsOffset,
            }
        else:
            metadata = {
                "x_data": [],
                "y_data": [],
                "z_data": [],
                "pfs_offset": None,
            }

        # Add basic metadata
        metadata.update(
            {
                "field_of_view": tile,
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

        # Sample z-planes if interval is specified and z_data exists
        if z_interval and len(metadata["z_data"]) > 0:
            df = df.iloc[::z_interval, :]

    return df


def nd2_to_tiff(
    file: str, channel_order_flip: bool = False, verbose: bool = False
) -> np.ndarray:
    """Converts a single ND2 file with one field of view and multiple channels to a multidimensional numpy array, ensuring the structure is CYX.

    Args:
        file (str): Path to the ND2 file.
        channel_order_flip (bool, optional): If True, flips the channel order. Defaults to False.
        verbose (bool, optional): If True, prints dimension information. Defaults to False.

    Returns:
        np.ndarray: Image data as a multidimensional numpy array.
    """
    # Load with xarray to keep dimension labels
    image = nd2.imread(file, xarray=True)

    if verbose:
        print(f"Original dimensions: {image.dims}")

    # Handle Z-stack if present
    if "Z" in image.dims:
        image = image.max(dim="Z")

    # Convert to numpy array in CYX order
    image_array = image.transpose("C", "Y", "X").values

    # Flip channel order on C axis if requested
    if channel_order_flip:
        image_array = np.flip(image_array, axis=0)  # axis 0 is the C axis

    if verbose:
        print(f"Final dimensions: {image_array.shape}")

    return image_array.astype(np.uint16)
