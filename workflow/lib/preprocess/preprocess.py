import re

import pandas as pd
from nd2reader import ND2Reader

def extract_tile_from_filename(filename):
    match = re.search(r'Points-(\d+)', filename)
    if match:
        return int(match.group(1))
    return None

def _extract_metadata_tile(files):
    """
    Extracts metadata from a list of ND2 files.

    Args:
        files (list): List of paths to ND2 files.
        parse_function_home (str): Absolute path to the screen directory.
        parse_function_dataset (str): Dataset name within the screen directory.
        parse_function_tiles (bool): Whether to include tile information in the parsing function.

    Returns:
        pandas.DataFrame: Combined extracted metadata from all provided ND2 files.
    """
    all_metadata = []
    
    # Iterate through all provided files
    for file_path in files:
        if file_path.endswith('.nd2'):
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
        print(f"No valid ND2 files found in the provided list.")
        return pd.DataFrame()  # Return an empty DataFrame if no files were processed
