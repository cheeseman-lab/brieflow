"""Functions for preprocessing ND2 files in preparation for downstream BrieFlow steps."""

import pandas as pd
import numpy as np
import nd2
import cv2
from typing import Union, List
from pathlib import Path
from skimage.measure import shannon_entropy


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

def entropy_focus(image):
    """ Computes a focus score for an image using Shannon entropy.

    The function first ensures the input image is in grayscale and 
    uint8 format, then calculates the Shannon entropy, which serves 
    as a measure of image focus. Higher entropy values generally 
    indicate a more focused (sharper) image, while lower values suggest blurriness.
    However, this should be tested empirically for the specific dataset.

    Args:
        image (np.ndarray): Input image as a NumPy array. Can be a 2D 
            grayscale image or a 3D color image (e.g., RGB).

    Returns:
        float: Shannon entropy value representing the focus score 
        of the input image.
    """
    # Convert to grayscale if the image has multiple channels
    if len(image.shape) > 2 and image.shape[2] > 1:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    # Ensure image is in the right format (uint8)
    if gray.dtype != np.uint8:
        if gray.max() > 0:
            gray = (gray * 255 / gray.max()).astype(np.uint8)
        else:
            gray = gray.astype(np.uint8)

    return shannon_entropy(gray)


def find_best_focus_slice(stack):
    """ Identifies and returns the sharpest (most focused) slice from a z-stack.

    The function computes a focus score for each slice in the 3D image stack using the
    `entropy_focus` function. Note: the function uses `np.argmin`, which implies lower scores are 
    treated as better — this will not be universally applicable to all datasets. Test the behavior 
    of `entropy_focus` to asess what score best caputres focus and adjust as needed (ex. by changing to `np.argmax`).

    Args:
        stack (list or np.ndarray): A sequence (e.g., list or 3D NumPy array) of 2D image slices,
            typically representing different focal planes in a z-stack.

    Returns:
        tuple:
            - np.ndarray: The 2D image slice with the best focus.
            - int: Index of the best-focused slice in the stack.
            - list of float: Focus scores (entropy values) for all slices in the stack.
    """
    focus_scores = [entropy_focus(slice) for slice in stack]
    best_index = np.argmin(focus_scores)
    return stack[best_index], best_index, focus_scores


def nd2_to_tiff(
    files: Union[str, List[str], Path, List[Path]],
    channel_order_flip: bool = False,
    best_focus_channel: Union[int, List[int], None] = None,
    verbose: bool = False,
    z_handling: str = "max_projection",
) -> np.ndarray:
    """Converts one or multiple ND2 files to a multidimensional numpy array, ensuring CYX structure.

    Args:
        files: Path(s) to the ND2 file(s). Can be a single path or list of paths.
        channel_order_flip: If True, flips the channel order. Defaults to False.
        best_focus_channel: For "best_focus" z_handling, specify which channel(s) (0-based index) to use
                           for determining the best focal plane. If None, each channel is processed separately.
                           Can be an integer for a single channel used across all files, or a list of integers
                           with the same length as files to specify different channels for each file.
        verbose: If True, prints dimension information. Defaults to False.
        z_handling: How to handle Z-stacks. Options are "max_projection", "no_z", or "best_focus".
                   Defaults to "max_projection".
                     - "max_projection": Takes the maximum intensity projection of the Z-stack.
                     - "no_z": Keeps only the first Z slice. Used if no Z-stack is present. 
                     - "best_focus": Uses the slice with the best focus based on Shannon entropy. See 
                        find_best_focus_slice and entropy_focus functions.

    Returns:
        np.ndarray: Image data as a multidimensional numpy array in CYX format.

    Raises:
        ValueError: If files have incompatible dimensions or parameters are invalid.
    """
    # Debugging support for z_handling
    if z_handling not in ("max_projection", "no_z", "best_focus"):
        raise ValueError(
            f"Invalid z_handling: {z_handling}. Choose 'Max_projection', 'No_z', or 'Best_slice'."
        )

    # Convert input to list of Path objects
    if isinstance(files, (str, Path)):
        files = [Path(files)]
    else:
        files = [Path(f) for f in files]

    # Ensures best_focus_channel matches the number of nd2 files
    if best_focus_channel is not None:
        if isinstance(best_focus_channel, int): # If using a single channel for all nd2 files
            best_focus_channels = [best_focus_channel] * len(files) # Sets the same channel for all files
        elif isinstance(best_focus_channel, list):
            # If using different channels for each nd2 file
            if len(best_focus_channel) != len(files): #Validates that list length matches the number of files
                raise ValueError(
                    f"Length of best_focus_channel list ({len(best_focus_channel)}) "
                    f"must match the number of files ({len(files)})"
                )
            best_focus_channels = best_focus_channel
        else:
            raise ValueError(
                "best_focus_channel must be an integer, list of integers, or None"
            )
    else:
        best_focus_channels = [None] * len(files)

    # Process all files
    image_arrays = []
    for i, (file, file_focus_channel) in enumerate(zip(files, best_focus_channels), 1):
        if verbose:
            print(f"Processing file {i}/{len(files)}: {file}")
            if file_focus_channel is not None and z_handling == "best_focus":
                print(
                    f"Using channel {file_focus_channel} for focus detection in this file"
                )

        image = nd2.imread(str(file), xarray=True)

        if verbose:
            print(f"Original dimensions for {file}: {image.dims}")

        # If Z-stack present, handles Z-stack processing depending on z_handling strategy:
        if "Z" in image.dims:
            if z_handling == "max_projection": # "max_projection" → takes maximum intensity projection
                image = image.max(dim="Z")
                if "C" in image.dims:
                    img_array = image.transpose("C", "Y", "X").values
                else:
                    img_array = np.expand_dims(image.values, axis=0)

            elif z_handling == "best_focus": # "best_focus" → selects most in-focus slice using Shannon entropy
                if "C" not in image.dims: # Single channel case
                    channel_data = image.values  # shape: (Z, Y, X)
                    # Apply focus detection using find_best_focus_slice to identify the sharpest Z slice
                    best_slice, best_index, scores = find_best_focus_slice(channel_data)
                    img_array = np.expand_dims(
                        best_slice, axis=0
                    )  # Add channel dimension

                    if verbose:
                        print(f"Best Z index: {best_index}, scores min: {min(scores)}")
                else: # Multiple channels present
                    num_channels = image.sizes["C"]

                    # Validate best_focus_channel if provided
                    if file_focus_channel is not None:
                        if file_focus_channel < 0 or file_focus_channel >= num_channels:
                            raise ValueError(
                                f"best_focus_channel must be between 0 and {num_channels - 1} for file {file}"
                            )

                        # Use the specified channel to find best Z index
                        focus_data = image.isel(
                            C=file_focus_channel
                        ).values  # shape: (Z, Y, X)
                        # Apply focus detection using find_best_focus_slice to identify the sharpest Z slice
                        _, best_z_index, scores = find_best_focus_slice(focus_data)

                        if verbose:
                            print(
                                f"Using channel {file_focus_channel} for focus detection. Best Z: {best_z_index}, scores min: {min(scores)}"
                            )

                        # Extract the best Z slice for all channels
                        best_slices = []
                        for c in range(num_channels):
                            channel_data = image.isel(
                                C=c, Z=best_z_index
                            ).values  # shape: (Y, X)
                            best_slices.append(channel_data)

                        img_array = np.stack(best_slices, axis=0)  # shape: (C, Y, X)
                    else:
                        # No channel provided. Find best slice for each channel independently
                        # Warning! Each channel might come from a different Z-slice, depending on which one is sharpest for that specific signal
                        best_slices = []
                        for c in range(num_channels):
                            channel_data = image.isel(C=c).values  # shape: (Z, Y, X)
                            # Your function expects this format so this should work
                            best_slice, best_index, scores = find_best_focus_slice(
                                channel_data
                            )
                            best_slices.append(best_slice)

                            if verbose:
                                print(
                                    f"Best Z for channel {c}: {best_index}, scores min: {min(scores)}"
                                )

                        img_array = np.stack(best_slices, axis=0)  # shape: (C, Y, X)

            elif z_handling == "no_z":
                # Keep only the first Z slice instead of full stack
                if verbose:
                    print(
                        "Warning: 'No_z' option with Z-stacks will take only the first Z slice."
                    )

                if "C" in image.dims:
                    # Select the first Z slice for all channels
                    img_array = image.isel(Z=0).transpose("C", "Y", "X").values
                else:
                    # Select the first Z slice for single channel
                    img_array = np.expand_dims(image.isel(Z=0).values, axis=0)
        else:
            # No Z dimension
            if verbose and z_handling == "best_focus":
                print(f"No Z-stack present in {file}; using original data.")

            # Convert to numpy array based on dimensions present
            if "C" in image.dims:
                img_array = image.transpose("C", "Y", "X").values
            else:
                img_array = np.expand_dims(
                    image.transpose("Y", "X").values, axis=0
                )  # Add channel dimension

        # Flip channel order if needed and C dimension exists
        if (
            channel_order_flip and len(img_array.shape) > 2
        ):  # Make sure there's a channel dimension
            img_array = np.flip(img_array, axis=0)

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
