import re

import pandas as pd
import numpy as np
from nd2reader import ND2Reader

def extract_tile_from_filename(filepath: str) -> int:
    """
    Extracts the tile number from a given filename.

    Args:
        filepath (str): The path to the file as a pathlib.Path object.

    Returns:
        int: The extracted tile number, or None if not found.
    """
    match = re.search(r'Points-(\d+)', filepath)
    if match:
        return int(match.group(1))
    return None

def extract_metadata_tile(files: list[str]) -> pd.DataFrame:
    """
    Extracts metadata from a list of ND2 files.

    Args:
        files (list[str]): List of pathlib.Path objects pointing to ND2 files.

    Returns:
        pandas.DataFrame: Combined extracted metadata from all provided ND2 files.
    """
    all_metadata = []
    
    # Iterate through all provided files
    for file_path in files:
            tile = extract_tile_from_filename(file_path)
            
            with ND2Reader(file_path) as images:
                raw_metadata = images.parser._raw_metadata
                
                data = {
                    'x_data': raw_metadata.x_data,
                    'y_data': raw_metadata.y_data,
                    'z_data': images.metadata['z_coordinates'],
                    'pfs_offset': raw_metadata.pfs_offset,
                    'field_of_view': tile,
                    'filename': file_path,
                }
                
                df = pd.DataFrame(data)
                
                if 'z_levels' in images.metadata and set(images.metadata['z_levels']) == set(range(0, 4)):
                    df = df.iloc[::4, :]
                all_metadata.append(df)
    
    # Combine all metadata
    if all_metadata:
        combined_metadata = pd.concat(all_metadata, ignore_index=True)
        
        # Convert 'field_of_view' to numeric, coercing any non-numeric values to NaN
        combined_metadata['field_of_view'] = pd.to_numeric(combined_metadata['field_of_view'], errors='coerce')
        # Sort by 'field_of_view' in ascending order
        combined_metadata = combined_metadata.sort_values('field_of_view').reset_index(drop=True)
        return combined_metadata
    else:
        print("No valid ND2 files found in the provided list.")
        return pd.DataFrame()  # Return an empty DataFrame if no files were processed

def nd2_to_tif(file: str, channel_order_flip: bool = False) -> np.ndarray:
    """
    Converts a single ND2 file with one field of view and multiple channels to a multidimensional numpy array.

    Args:
        file (str): Path to the ND2 file
        channel_order_flip (bool, optional): If True, reverses the order of channels. Defaults to False.

    Returns:
        np.ndarray: Image data as a multidimensional numpy array.
    """
    with ND2Reader(file) as images:

        # Determine the axes order (always include 'c' for channels)
        axes = 'cyx'
        if 'z' in images.axes:
            axes = 'zcyx'

        images.bundle_axes = axes
        
        # Get the single image (all channels)
        image = images[0]

        # Handle z-stacks: max project if 'z' is in axes
        if 'z' in axes:
            image = np.max(image, axis=0)  # Max projection along z-axis

        # Flip channel order if specified
        if channel_order_flip:
            image = np.flip(image, axis=0)  # Flip along first axis (channels)

        # Ensure the image is uint16
        image_array = np.array(image, dtype=np.uint16)

    return image_array
