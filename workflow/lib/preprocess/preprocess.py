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
                # Get coordinate data with error handling
                metadata = {
                    "x_data": getattr(images.experiment[0], "position_x", []),
                    "y_data": getattr(images.experiment[0], "position_y", []),
                    "z_data": getattr(images.experiment[0], "position_z", []),
                    "field_of_view": tile,
                    "filename": file_path,
                }

                # Add additional metadata if available
                try:
                    metadata["pfs_offset"] = images.metadata.channels[0].pfs_offset
                except (AttributeError, IndexError):
                    metadata["pfs_offset"] = None

                # Add more metadata fields
                metadata.update(
                    {
                        "channels": images.channel_count(),
                        "pixel_size_x": getattr(images.metadata, "pixel_size_x", None),
                        "pixel_size_y": getattr(images.metadata, "pixel_size_y", None),
                        "z_step": getattr(images.metadata, "z_step", None),
                    }
                )

                df = pd.DataFrame(metadata)

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

    print("No valid ND2 files found in the provided list.")
    return pd.DataFrame()


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
