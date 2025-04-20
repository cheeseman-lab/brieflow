"""Module for aligning cycles in SBS.

Uses NumPy and scikit-image to provide image
alignment between sequencing cycles, apply percentile-based filtering, fill masked
areas with noise, and perform various transformations to enhance image data quality.
"""

import numpy as np

from lib.shared.align import (
    apply_window,
    normalize_by_percentile,
    calculate_offsets,
    apply_offsets,
    filter_percentiles,
)


def align_cycles(
    image_data,
    channel_order=None,
    method=None,
    upsample_factor=2,
    window=2,
    cutoff=1,
    q_norm=70,
    use_align_within_cycle=True,
    cycle_files=None,
):
    """Rigid alignment of sequencing cycles and channels.

    Args:
        image_data (np.ndarray or list of np.ndarray): Unaligned SBS image with dimensions
            (CYCLE, CHANNEL, I, J) or list of single cycle SBS images, each with dimensions
            (CHANNEL, I, J).
        channel_order (list[str], optional): List of channel names in the order they are acquired.
            Example: ["DAPI", "G", "T", "A", "C"]. If None, will assume first channel is DAPI
            and remaining are bases. Defaults to None.
        method (str, optional): Method to use for alignment. Options are {'DAPI', 'sbs_mean'}.
            If None, will automatically select based on available channels. Defaults to None.
        upsample_factor (int, optional): Subpixel alignment is done if greater than one
            (can be slow). Defaults to 2.
        window (int or float, optional): A centered subset of data is used if greater than one.
            Defaults to 2.
        cutoff (int or float, optional): Cutoff for normalized data to help deal with noise in
            images. Defaults to 1.
        q_norm (int, optional): Quantile for normalization to help deal with noise in images.
            Defaults to 70.
        use_align_within_cycle (bool, optional): Align SBS channels within cycles. Defaults to True.
        cycle_files (list[int] or None, optional): Used for parsing sets of images where individual
            channels are in separate files. Defaults to None.

    Returns:
        np.ndarray: SBS image aligned across cycles.
    """
    # Handle case where cycle_files is provided
    if cycle_files is not None:
        arr = []
        current = 0
        for cycle in cycle_files:
            if cycle == 1:
                arr.append(image_data[current])
            else:
                arr.append(np.array(image_data[current : current + cycle]))
            current += cycle
        image_data = arr

    # Determine the channel structure
    base_channels = ["G", "T", "A", "C"]
    if channel_order is None:
        if isinstance(image_data, list):
            n_channels = min(x.shape[-3] if x.ndim > 2 else 1 for x in image_data)
        else:
            n_channels = image_data.shape[1]
        
        channel_order = ["DAPI"] + base_channels[:n_channels-1] if n_channels > 1 else ["DAPI"]
    
    # Identify base channels and extra channels
    base_indices = [i for i, ch in enumerate(channel_order) if ch in base_channels]
    extra_indices = [i for i, ch in enumerate(channel_order) if ch not in base_channels]
    
    # Check if channels are consistent across cycles
    channels_consistent = not isinstance(image_data, list) or all(x.shape == image_data[0].shape for x in image_data)

    # Check if number of channels varies across cycles
    if not channels_consistent:
        print("Warning: Number of channels varies across cycles.")
        
        # Process cycles with inconsistent channels
        stacked_cycles = []
        ref_cycle = image_data[0]
        ref_shape = ref_cycle.shape
        
        for cycle_idx, cycle_data in enumerate(image_data):
            current_cycle = cycle_data.copy()
            
            # Add missing extra channels from first cycle
            if cycle_data.shape[-3] != ref_shape[-3] and cycle_idx > 0:
                missing_extras = [i for i in extra_indices 
                                 if i < ref_shape[-3] and 
                                 (i >= cycle_data.shape[-3] or cycle_data.shape[-3] < ref_shape[-3])]
                
                if missing_extras:
                    # Get missing channels from first cycle
                    missing_channels = np.stack([ref_cycle[i] for i in missing_extras])
                    
                    # Add missing channels to current cycle
                    current_cycle = np.concatenate([missing_channels, current_cycle], axis=0)
            
            stacked_cycles.append(current_cycle)
        
        # Make sure all cycles have same shape
        shapes = [c.shape for c in stacked_cycles]
        if len(set(shapes)) > 1:
            print(f"Warning: After processing, cycles still have different shapes")
            
            # Find max dimensions and ensure all cycles match
            max_channels = max(s[-3] for s in shapes)
            
            for i, cycle in enumerate(stacked_cycles):
                if cycle.shape[-3] < max_channels:
                    # Add missing channels from first cycle or zeros if needed
                    pad_channels = []
                    for j in range(cycle.shape[-3], max_channels):
                        if j < ref_cycle.shape[-3]:
                            pad_channels.append(ref_cycle[j])
                        else:
                            pad_channels.append(np.zeros((cycle.shape[-2], cycle.shape[-1]), 
                                                       dtype=cycle.dtype))
                    
                    pad_array = np.stack(pad_channels)
                    stacked_cycles[i] = np.concatenate([cycle, pad_array], axis=0)
        
        stacked = np.stack(stacked_cycles)
    else:
        # All cycles have the same number of channels
        stacked = np.stack(image_data) if isinstance(image_data, list) else image_data
    
    assert stacked.ndim == 4, "Input image_data must have dimensions CYCLE, CHANNEL, I, J"

    # Automatically determine method if not provided
    if method is None:
        if channels_consistent:
            method = "DAPI"
        else:
            method = "sbs_mean"
        print(f"Method not provided. Using '{method}' for alignment based on data structure.")

    # Align between SBS channels for each cycle
    aligned = stacked.copy()
    
    if use_align_within_cycle and base_indices:
        # Only align base channels within cycle
        min_base_idx = min(base_indices)
        base_slices = slice(min_base_idx, None) if all(i >= min_base_idx for i in base_indices) else base_indices
        
        def align_it(x):
            return align_within_cycle(x, window=window, upsample_factor=upsample_factor)
        
        aligned[:, base_slices] = np.array([align_it(x) for x in aligned[:, base_slices]])

    # Align between cycles
    if method == "DAPI":
        # Only attempt DAPI alignment if DAPI channel exists
        if 0 in range(aligned.shape[1]) and (channel_order is None or channel_order[0] == "DAPI"):
            dapi_index = 0
            # Align cycles using the DAPI channel
            aligned = align_between_cycles(
                aligned, channel_index=dapi_index, window=window, upsample_factor=upsample_factor
            )
        else:
            print("Warning: 'DAPI' method selected but DAPI channel not available. Switching to 'sbs_mean'.")
            method = "sbs_mean"  # Fall back to sbs_mean method
            
    if method == "sbs_mean":
        # Calculate cycle offsets using ONLY the base channels (ignore extra channels)
        if base_indices:
            sbs_channels = base_indices
        else:
            print("Warning: No base channels found for 'sbs_mean' method. Using all channels.")
            sbs_channels = list(range(aligned.shape[1]))
            
        target = apply_window(aligned[:, sbs_channels], window=window).max(axis=1)
        normed = normalize_by_percentile(target, q_norm=q_norm)
        normed[normed > cutoff] = cutoff
        offsets = calculate_offsets(normed, upsample_factor=upsample_factor)
        
        # Apply cycle offsets to ALL channels (both base and extra)
        for channel in range(aligned.shape[1]):
            aligned[:, channel] = apply_offsets(aligned[:, channel], offsets)
    else:
        raise ValueError(f'Method "{method}" not implemented')

    return aligned


def align_within_cycle(data_, upsample_factor=4, window=1, q1=0, q2=90):
    """Align images within the same cycle.

    Args:
        data_ (np.ndarray): Image data.
        upsample_factor (int, optional): Upsampling factor for cross-correlation. Defaults to 4.
        window (int, optional): Size of the window to apply during alignment. Defaults to 1.
        q1 (int, optional): Lower percentile threshold. Defaults to 0.
        q2 (int, optional): Upper percentile threshold. Defaults to 90.

    Returns:
        np.ndarray: Aligned image data.
    """
    # Filter the input data based on percentiles
    filtered = filter_percentiles(apply_window(data_, window), q1=q1, q2=q2)
    # Calculate offsets using the filtered data
    offsets = calculate_offsets(filtered, upsample_factor=upsample_factor)
    # Apply the calculated offsets to the original data and return the result
    return apply_offsets(data_, offsets)


def align_between_cycles(
    data, channel_index, upsample_factor=4, window=1, return_offsets=False
):
    """Align images between different cycles.

    Args:
        data (np.ndarray): Image data.
        channel_index (int): Index of the channel to align between cycles.
        upsample_factor (int, optional): Upsampling factor for cross-correlation. Defaults to 4.
        window (int, optional): Size of the window to apply during alignment. Defaults to 1.
        return_offsets (bool, optional): Whether to return the calculated offsets. Defaults to False.

    Returns:
        np.ndarray: Aligned image data.
        np.ndarray, optional: Calculated offsets if return_offsets is True.
    """
    # Calculate offsets from the target channel
    target = apply_window(data[:, channel_index], window)
    offsets = calculate_offsets(target, upsample_factor=upsample_factor)

    # Apply the calculated offsets to all channels
    warped = []
    for data_ in data.transpose([1, 0, 2, 3]):
        warped += [apply_offsets(data_, offsets)]

    # Transpose the array back to its original shape
    aligned = np.array(warped).transpose([1, 0, 2, 3])

    # Return aligned data with offsets if requested
    if return_offsets:
        return aligned, offsets
    else:
        return aligned
