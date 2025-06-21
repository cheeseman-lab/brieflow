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
    skip_cycles=None,
    manual_background_cycle=None,
    manual_channel_mapping=None,
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
        skip_cycles (list[int] or None, optional): List of cycle indices to skip (0-based).
            These cycles will be completely excluded from alignment. Defaults to None.
        manual_background_cycle (int or None, optional): Specific cycle to use for
            background channel (0-based). Must be specified by user if needed.
            Defaults to None. If not specified, and extra channels are present,
            the cycle with the most extra channels will be used as the source for
            propagating extra channels across cycles. Only used if shapes vary across cycles.
        manual_channel_mapping (list or None, optional): List of channel orders for each cycle.
            Each element should be a list of channel names in the order they appear in that cycle's data.
            If provided, this will override automatic channel detection and enable smart channel filling.
            Example: [["DAPI", "G", "T", "A", "C"], ["DAPI", "G", "T", "A", "C"], ["DAPI", "GFP", "G", "T", "A", "C", "AF750"]]
            for a 3-cycle dataset where the third cycle has additional GFP and AF750 channels.
            Defaults to None.

    Returns:
        np.ndarray: SBS image aligned across cycles.
    """
    skip_cycles = skip_cycles or []

    # Handle cycle skipping
    if skip_cycles:
        print(f"Skipping cycles: {skip_cycles} out of {len(image_data)} total cycles")
        processed_data = []

        for i, data in enumerate(image_data):
            if i in skip_cycles:
                print(f"Skipping cycle {i} with shape {data.shape}")
            else:
                processed_data.append(data)

        if len(processed_data) == 0:
            raise ValueError("All cycles were skipped - no data to process")

        image_data = processed_data
        print(
            f"Processing {len(processed_data)} cycles after skipping {len(skip_cycles)}"
        )

    # Handle manual channel mapping if provided
    if manual_channel_mapping is not None:
        # Use user-specified channel mapping
        stacked = manual_fill_channels(
            image_data,
            current_channel_orders=manual_channel_mapping,
            target_channel_order=channel_order,
            fill_method="smart",
            source_cycle_priority=[manual_background_cycle]
            if manual_background_cycle is not None
            else None,
        )

        # Define base_indices for the target channel order
        base_channels = ["G", "T", "A", "C"]
        base_indices = [i for i, ch in enumerate(channel_order) if ch in base_channels]
        extra_indices = [
            i for i, ch in enumerate(channel_order) if ch not in base_channels
        ]

        # Set method if not provided
        if method is None:
            method = (
                "DAPI" if channel_order and channel_order[0] == "DAPI" else "sbs_mean"
            )
            print(f"Method not provided. Using '{method}' for manual channel mapping.")

    # If no manual mapping is provided, determine channel structure automatically
    else:
        # Determine the channel structure
        base_channels = ["G", "T", "A", "C"]
        if channel_order is None:
            if isinstance(image_data, list):
                n_channels = min(x.shape[-3] if x.ndim > 2 else 1 for x in image_data)
            else:
                n_channels = image_data.shape[1]

            channel_order = (
                ["DAPI"] + base_channels[: n_channels - 1]
                if n_channels > 1
                else ["DAPI"]
            )

        # Identify base channels and extra channels
        base_indices = [i for i, ch in enumerate(channel_order) if ch in base_channels]
        extra_indices = [
            i for i, ch in enumerate(channel_order) if ch not in base_channels
        ]

        # Handle channel inconsistencies - simplified approach
        if not all(x.shape == image_data[0].shape for x in image_data):
            print("Warning: Number of channels varies across cycles.")

            # Keep only channels in common across all cycles
            channels = [x.shape[-3] if x.ndim > 2 else 1 for x in image_data]
            min_channels = min(channels)
            print(f"Channel counts: {channels}, using minimum: {min_channels}")

            stacked = np.array([x[-min_channels:] for x in image_data])

            # Automatically add back extra channels (propagate to all cycles)
            extras = np.array(channels) - min_channels
            if any(extras > 0):
                print("Propagating extra channels to all cycles...")
                arr = []

                # Find the cycle with extra channels (manual_background_cycle or cycle with most extras)
                source_cycle_idx = None
                if manual_background_cycle is not None:
                    # Convert to processed cycle index after skipping
                    adjusted_idx = manual_background_cycle
                    for skip_idx in sorted(skip_cycles):
                        if skip_idx <= manual_background_cycle:
                            adjusted_idx -= 1
                    if 0 <= adjusted_idx < len(image_data) and extras[adjusted_idx] > 0:
                        source_cycle_idx = adjusted_idx
                        print(
                            f"Using user-specified segmentation background cycle {manual_background_cycle} (processed index {adjusted_idx})"
                        )

                if source_cycle_idx is None:
                    # Find cycle with the most extra channels
                    max_extra_cycle = np.argmax(extras)
                    if extras[max_extra_cycle] > 0:
                        source_cycle_idx = max_extra_cycle
                        print(
                            f"Auto-selected cycle {max_extra_cycle} as source (has {extras[max_extra_cycle]} extra channels)"
                        )

                if source_cycle_idx is not None:
                    # Get ALL extra channels from the source cycle
                    for extra_ch in range(int(extras[source_cycle_idx])):
                        arr.append(image_data[source_cycle_idx][extra_ch])

                    propagate = np.array(arr)
                    print(
                        f"Propagating {len(arr)} extra channels with shapes: {[ch.shape for ch in arr]}"
                    )

                    # Add extra channels to the beginning of all cycles
                    stacked = np.concatenate(
                        (np.array([propagate] * stacked.shape[0]), stacked), axis=1
                    )
        else:
            # All cycles have the same number of channels
            stacked = (
                np.array(image_data) if isinstance(image_data, list) else image_data
            )

        # Debug print before final stacking
        print(f"Final stacked shape before alignment: {stacked.shape}")

        assert stacked.ndim == 4, (
            "Input image_data must have dimensions CYCLE, CHANNEL, I, J"
        )

        # Automatically determine method if not provided
        if method is None:
            # Use DAPI if we have consistent channels, sbs_mean if inconsistent
            if all(x.shape == image_data[0].shape for x in image_data):
                method = "DAPI"
            else:
                method = "sbs_mean"
            print(
                f"Method not provided. Using '{method}' for alignment based on data structure."
            )

    # Align between SBS channels for each cycle
    aligned = stacked.copy()

    if use_align_within_cycle and base_indices:
        # Only align base channels within cycle
        min_base_idx = min(base_indices)
        base_slices = (
            slice(min_base_idx, None)
            if all(i >= min_base_idx for i in base_indices)
            else base_indices
        )

        def align_it(x):
            return align_within_cycle(x, window=window, upsample_factor=upsample_factor)

        aligned[:, base_slices] = np.array(
            [align_it(x) for x in aligned[:, base_slices]]
        )

    # Align between cycles
    if method == "DAPI":
        # Only attempt DAPI alignment if DAPI channel exists
        if 0 in range(aligned.shape[1]) and (
            channel_order is None or channel_order[0] == "DAPI"
        ):
            dapi_index = 0
            # Align cycles using the DAPI channel
            aligned = align_between_cycles(
                aligned,
                channel_index=dapi_index,
                window=window,
                upsample_factor=upsample_factor,
            )
        else:
            print(
                "Warning: 'DAPI' method selected but DAPI channel not available. Switching to 'sbs_mean'."
            )
            method = "sbs_mean"  # Fall back to sbs_mean method

    elif method == "sbs_mean":
        # Calculate cycle offsets using ONLY the base channels (ignore extra channels)
        if base_indices:
            sbs_channels = base_indices
        else:
            print(
                "Warning: No base channels found for 'sbs_mean' method. Using all channels."
            )
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


def manual_fill_channels(
    image_data,
    current_channel_orders,
    target_channel_order,
    fill_method="smart",
    source_cycle_priority=None,
):
    """Fill cycles to match target channel order by mapping channels by name.

    Args:
        image_data: List of cycle arrays, each with shape (CHANNEL, I, J)
        current_channel_orders: List of channel names for each cycle
        target_channel_order: Final desired channel order
        fill_method: 'zeros', 'smart' (copy from other cycles), or specific fill value
        source_cycle_priority: List of cycle indices to prioritize when copying channels

    Returns:
        np.ndarray: Stacked array with shape (CYCLE, len(target_channel_order), I, J)
    """
    n_cycles = len(image_data)
    target_n_channels = len(target_channel_order)
    spatial_shape = image_data[0].shape[1:]

    aligned_data = np.zeros(
        (n_cycles, target_n_channels) + spatial_shape, dtype=image_data[0].dtype
    )

    # Build a map of which cycles have which channels
    channel_sources = {}  # {channel_name: [cycle_indices_that_have_it]}
    for cycle_idx, current_order in enumerate(current_channel_orders):
        for channel_name in current_order:
            if channel_name not in channel_sources:
                channel_sources[channel_name] = []
            channel_sources[channel_name].append(cycle_idx)

    # Fill each cycle
    for cycle_idx, (cycle_data, current_order) in enumerate(
        zip(image_data, current_channel_orders)
    ):
        print(f"Cycle {cycle_idx}: {current_order} -> {target_channel_order}")

        for target_idx, channel_name in enumerate(target_channel_order):
            if channel_name in current_order:
                # Copy from current cycle
                source_idx = current_order.index(channel_name)
                aligned_data[cycle_idx, target_idx] = cycle_data[source_idx]
                print(
                    f"  {channel_name}: copied from current cycle position {source_idx}"
                )
            elif fill_method == "smart" and channel_name in channel_sources:
                # Copy from another cycle that has this channel
                source_cycles = channel_sources[channel_name]

                # Choose source cycle (prioritize user preference, then first available)
                if source_cycle_priority:
                    chosen_cycle = next(
                        (c for c in source_cycle_priority if c in source_cycles),
                        source_cycles[0],
                    )
                else:
                    chosen_cycle = source_cycles[0]

                source_order = current_channel_orders[chosen_cycle]
                source_idx = source_order.index(channel_name)
                aligned_data[cycle_idx, target_idx] = image_data[chosen_cycle][
                    source_idx
                ]
                print(
                    f"  {channel_name}: copied from cycle {chosen_cycle} position {source_idx}"
                )
            else:
                # Fill with zeros or specified value
                fill_val = 0 if fill_method == "smart" else fill_method
                print(f"  {channel_name}: filled with {fill_val}")

    return aligned_data
