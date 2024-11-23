"""Functions for preprocessing ND2 files in preparation for downstream BrieFlow steps."""

import pandas as pd
import numpy as np
import nd2

from lib.preprocess.file_utils import extract_tile_from_filename


def extract_metadata_tile(
    files: list[str], z_interval: int = 4, verbose: bool = False
) -> pd.DataFrame:
    """Extracts metadata from a list of ND2 files.

    Args:
        files (list[str]): List of file paths pointing to ND2 files.
        z_interval (int, optional): If set, samples z-planes at this interval to ensure metadata is one line per position. Defaults to 4.
        verbose (bool, optional): If True, prints metadata information. Defaults to False.

    Returns:
        pd.DataFrame: Combined extracted metadata from all provided ND2 files.
    """
    all_metadata = []
    for file_path in files:
        if verbose:
            print(f"Processing {file_path}")
        tile = extract_tile_from_filename(file_path)
        
        try:
            with nd2.ND2File(file_path) as images:
                frame_meta = images.frame_metadata(0)
                
                # Get position data from first channel's position information
                if frame_meta.channels and hasattr(frame_meta.channels[0], 'position'):
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
                metadata.update({
                    "field_of_view": tile,
                    "filename": file_path,
                    "channels": frame_meta.contents.channelCount,
                })
                
                # Get pixel size from first channel's volume information
                if frame_meta.channels and hasattr(frame_meta.channels[0], 'volume'):
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
                
                # Sample z-planes if interval is specified and z_data exists
                if z_interval and len(metadata["z_data"]) > 0:
                    df = df.iloc[::z_interval, :]
                
                if verbose:
                    print(f"Found {len(df)} positions for tile {tile}")
                all_metadata.append(df)
                
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue

    if all_metadata:
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        combined_metadata["field_of_view"] = pd.to_numeric(
            combined_metadata["field_of_view"], errors="coerce"
        )
        return combined_metadata.sort_values("field_of_view").reset_index(drop=True)

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
