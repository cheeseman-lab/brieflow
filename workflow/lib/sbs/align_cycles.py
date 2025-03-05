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
    method="DAPI",
    upsample_factor=2,
    window=2,
    cutoff=1,
    q_norm=70,
    use_align_within_cycle=True,
    cycle_files=None,
    keep_extras=False,
    n=1,
    remove_for_cycle_alignment=None,
):
    """Rigid alignment of sequencing cycles and channels.

    Args:
        image_data (np.ndarray or list of np.ndarray): Unaligned SBS image with dimensions
            (CYCLE, CHANNEL, I, J) or list of single cycle SBS images, each with dimensions
            (CHANNEL, I, J).
        method (str): Method to use for alignment. Options are {'DAPI', 'sbs_mean'}.
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
            channels are in separate files, typically handled in preprocessing to combine images
            from the same cycle. Defaults to None.
        keep_extras (bool, optional): Retain channels that are not common across all cycles by
            propagating each 'extra' channel to all cycles. Ignored if the same number of channels
            exist for all cycles. Defaults to False.
        n (int, optional): Determines the first SBS channel in `data`. This should only account
            for channels in common across all cycles if `keep_extras` is False. Defaults to 1.
        remove_for_cycle_alignment (int or None, optional): Channel index to remove when finding
            cycle offsets. This should only account for channels in common across all cycles if
            `keep_extras` is False. Defaults to None.

    Returns:
        np.ndarray: SBS image aligned across cycles.
    """
    # Handle case where cycle_files is provided
    if cycle_files is not None:
        arr = []
        current = 0
        # Iterate through cycle files to de-nest list of numpy arrays
        for cycle in cycle_files:
            if cycle == 1:
                arr.append(image_data[current])
            else:
                arr.append(np.array(image_data[current : current + cycle]))
            current += cycle
        image_data = arr

    # Check if the number of channels varies across cycles
    if not all(x.shape == image_data[0].shape for x in image_data):
        # Keep only channels in common across all cycles
        channels = [x.shape[-3] if x.ndim > 2 else 1 for x in image_data]
        stacked = np.array([x[-min(channels) :] for x in image_data])

        # Add back extra channels if requested
        if keep_extras:
            extras = np.array(channels) - min(channels)
            arr = []
            for cycle, extra in enumerate(extras):
                if extra != 0:
                    arr.extend(
                        [image_data[cycle][extra_ch] for extra_ch in range(extra)]
                    )
            propagate = np.array(arr)
            stacked = np.concatenate(
                (np.array([propagate] * stacked.shape[0]), stacked), axis=1
            )
        else:
            extras = [0] * stacked.shape[0]
    else:
        stacked = np.array(image_data)
        extras = [0] * stacked.shape[0]

    assert stacked.ndim == 4, (
        "Input image_data must have dimensions CYCLE, CHANNEL, I, J"
    )

    # Align between SBS channels for each cycle
    aligned = stacked.copy()
    if use_align_within_cycle:

        def align_it(x):
            return align_within_cycle(x, window=window, upsample_factor=upsample_factor)

        aligned[:, n:] = np.array([align_it(x) for x in aligned[:, n:]])

    if method == "DAPI":
        # Align cycles using the DAPI channel
        aligned = align_between_cycles(
            aligned, channel_index=0, window=window, upsample_factor=upsample_factor
        )
    elif method == "sbs_mean":
        # Calculate cycle offsets using the average of SBS channels
        sbs_channels = list(range(n, aligned.shape[1]))
        if remove_for_cycle_alignment is not None:
            sbs_channels.remove(remove_for_cycle_alignment)
        target = apply_window(aligned[:, sbs_channels], window=window).max(axis=1)
        normed = normalize_by_percentile(target, q_norm=q_norm)
        normed[normed > cutoff] = cutoff
        offsets = calculate_offsets(normed, upsample_factor=upsample_factor)
        # Apply cycle offsets to each channel
        for channel in range(aligned.shape[1]):
            if channel >= sum(extras):
                aligned[:, channel] = apply_offsets(aligned[:, channel], offsets)
            else:
                # Don't apply offsets to extra channel in the cycle it was acquired
                extra_idx = list(np.cumsum(extras) > channel).index(True)
                extra_offsets = np.array([offsets[extra_idx]] * aligned.shape[0])
                aligned[:, channel] = apply_offsets(aligned[:, channel], extra_offsets)
    else:
        raise ValueError(f'method "{method}" not implemented')

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
