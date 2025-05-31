"""Module for aligning channels in phenotype.

Uses NumPy and scikit-image to provide image
alignment between sequencing cycles, apply percentile-based filtering, fill masked
areas with noise, and perform various transformations to enhance image data quality.
"""

import numpy as np
from lib.shared.image_utils import remove_channels
from lib.shared.align import apply_window, calculate_offsets, apply_offsets


def align_phenotype_channels(
    image_data,
    target,
    source,
    riders=[],
    upsample_factor=2,
    window=2,
    remove_channel=False,
    debug=True,  # New debug parameter
):
    """Rigid alignment of phenotype channels based on target and source channels.

    Args:
        image_data (np.ndarray): The input data containing the channels with dimensions
            (STACK, CHANNEL, I, J) if stacked, or (CHANNEL, I, J) if not.
        target (int): Index of the channel that other channels will be aligned to.
        source (int): Index of the channel to align with the target.
        riders (list[int], optional): Additional channel indices that should follow
            the same alignment as the source channel. Defaults to [].
        upsample_factor (int, optional): Subpixel alignment is done if greater than one.
            Defaults to 2.
        window (int, optional): A centered subset of data is used if greater than one.
            Defaults to 2.
        remove_channel (str or bool, optional): Specifies whether to remove channels after alignment.
            Options are {'target', 'source', False}. Defaults to False.
        debug (bool, optional): Whether to print debugging information. Defaults to False.

    Returns:
        tuple: (aligned_data, offsets_info) where:
            - aligned_data (np.ndarray): Phenotype data aligned across specified channels.
            - offsets_info (dict): Dictionary containing offset information for debugging.
    """
    # Handle stacked vs unstacked data
    if image_data.ndim == 4:
        data_ = image_data.max(axis=0)
        stack = True
    else:
        data_ = image_data.copy()
        stack = False

    # Calculate alignment offsets
    windowed = apply_window(data_[[target, source]], window)
    offsets = calculate_offsets(windowed, upsample_factor=upsample_factor)

    # Handle riders and create full offsets array
    if not isinstance(riders, list):
        riders = [riders]
    full_offsets = np.zeros((data_.shape[0], 2))
    full_offsets[[source] + riders] = offsets[1]

    # Debug: Print offsets information
    if debug:
        print(f"Target channel: {target}")
        print(f"Source channel: {source}")
        print(f"Rider channels: {riders}")
        print(f"Calculated offset for source: {offsets[1]}")
        print(f"Full offsets array: {full_offsets}")

        # Additional details about the magnitude of shifts
        shift_magnitudes = np.sqrt(np.sum(full_offsets**2, axis=1))
        print(f"Shift magnitudes (pixels): {shift_magnitudes}")
        channels_being_moved = [i for i, mag in enumerate(shift_magnitudes) if mag > 0]
        print(f"Channels being moved: {channels_being_moved}")

    # Apply alignment
    if stack:
        aligned = np.array(
            [apply_offsets(slice_, full_offsets) for slice_ in image_data]
        )
    else:
        aligned = apply_offsets(data_, full_offsets)

    # Handle channel removal if specified
    if remove_channel == "target":
        channel_order = list(range(image_data.shape[-3]))
        channel_order.remove(source)
        channel_order.insert(target + 1, source)
        aligned = aligned[..., channel_order, :, :]
        aligned = remove_channels(aligned, target)
    elif remove_channel == "source":
        aligned = remove_channels(aligned, source)

    # Create offsets info dictionary for debugging
    offsets_info = {
        "target_channel": target,
        "source_channel": source,
        "rider_channels": riders,
        "calculated_offset_for_source": offsets[1],
        "full_offsets_array": full_offsets,
        "shift_magnitudes": np.sqrt(np.sum(full_offsets**2, axis=1)),
    }

    return aligned, offsets_info
