"""Module for aligning cycles in SBS.

Uses NumPy and scikit-image to provide image
alignment between sequencing cycles, apply percentile-based filtering, fill masked
areas with noise, and perform various transformations to enhance image data quality.
"""

import numpy as np
from scipy import ndimage
from skimage.registration import phase_cross_correlation

from lib.shared.align import (
    apply_window,
    normalize_by_percentile,
    calculate_offsets,
    apply_offsets,
    filter_percentiles,
    offsets_to_metrics,
)
from lib.shared.alignment_overlay import center_crop, plot_overlay_grid

# a cycle or channel shifted by at least this many pixels is reported as off
ALIGNMENT_PASS_PX = 1.0
# spot images are stretched between these percentiles so background noise stays dark
SPOT_DISPLAY_PERCENTILES = (99, 99.95)


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
    verbose=False,
    return_metrics=False,
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
        verbose (bool, optional): If True, print detailed alignment information including
            calculated offsets for each cycle. Useful for debugging alignment issues.
            Defaults to False.
        return_metrics (bool, optional): If True, also return a dict of per-cycle alignment
            offset metrics keyed offset_y_cycle{i}/offset_x_cycle{i}. Defaults to False.

    Returns:
        np.ndarray: SBS image aligned across cycles.
        dict, optional: Per-cycle alignment offset metrics if return_metrics is True.
    """
    skip_cycles = skip_cycles or []
    n_input_cycles = len(image_data)

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

    # Track the source cycle for extra channels (for proper offset application)
    extra_channel_source_cycle = None

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

        # Track source cycle for extra channels
        if manual_background_cycle is not None:
            extra_channel_source_cycle = manual_background_cycle

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

                    # Track the source cycle for proper offset application later
                    extra_channel_source_cycle = source_cycle_idx
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

    # Track per-cycle offsets from whichever alignment branch runs
    cycle_offsets = None

    # Align between cycles
    if method == "DAPI":
        # Only attempt DAPI alignment if DAPI channel exists
        if 0 in range(aligned.shape[1]) and (
            channel_order is None or channel_order[0] == "DAPI"
        ):
            dapi_index = 0
            # Align cycles using the DAPI channel
            aligned, offsets = align_between_cycles(
                aligned,
                channel_index=dapi_index,
                window=window,
                upsample_factor=upsample_factor,
                return_offsets=True,
            )
            cycle_offsets = offsets

            if verbose:
                print("\n=== Cycle Alignment Offsets (DAPI method) ===")
                for cycle_idx, offset in enumerate(offsets):
                    print(f"  Cycle {cycle_idx}: shift = {offset} pixels (y, x)")
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
        offsets, _ = calculate_offsets(normed, upsample_factor=upsample_factor)
        cycle_offsets = offsets

        if verbose:
            print("\n=== Cycle Alignment Offsets (sbs_mean method) ===")
            for cycle_idx, offset in enumerate(offsets):
                print(f"  Cycle {cycle_idx}: shift = {offset} pixels (y, x)")

        # Apply cycle offsets conditionally based on channel type
        for channel in range(aligned.shape[1]):
            if channel in extra_indices and extra_channel_source_cycle is not None:
                # Extra channels: use ONLY the offset from the cycle they were acquired in
                # This prevents misalignment when the same image is propagated across cycles
                source_offset = np.array(
                    [offsets[extra_channel_source_cycle]] * aligned.shape[0]
                )
                aligned[:, channel] = apply_offsets(aligned[:, channel], source_offset)
                if (
                    channel == extra_indices[0]
                ):  # Print once for the first extra channel
                    print(
                        f"Applying source cycle {extra_channel_source_cycle} offset to {len(extra_indices)} extra channel(s)"
                    )
            else:
                # Base channels: apply cycle-specific offsets
                aligned[:, channel] = apply_offsets(aligned[:, channel], offsets)
    else:
        raise ValueError(f'Method "{method}" not implemented')

    # Alignment QC: per-cycle DAPI residual and per-cycle, per-channel spot shifts
    cycle_labels = [i + 1 for i in range(n_input_cycles) if i not in skip_cycles]
    report_alignment_qc(
        aligned,
        channel_order,
        base_indices,
        cycle_labels=cycle_labels,
        upsample_factor=upsample_factor,
    )

    if return_metrics:
        return aligned, (
            offsets_to_metrics(cycle_offsets, "cycle")
            if cycle_offsets is not None
            else {}
        )

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
    offsets, _ = calculate_offsets(filtered, upsample_factor=upsample_factor)
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
    offsets, _ = calculate_offsets(target, upsample_factor=upsample_factor)

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


def report_alignment_qc(
    aligned, channel_order, base_indices, cycle_labels=None, upsample_factor=2
):
    """Print alignment QC for aligned SBS data, naming any cycle or channel that is off.

    Two metrics are printed in a fixed format: `cycle_dapi_shift_residual_max_px` (largest
    DAPI shift of any cycle against the first cycle) and
    `intra_cycle_channel_shift_residual_max_px` (largest shift of a base channel against
    the other base channels of its cycle, see `channel_shift_residuals`). A per-cycle table
    follows, and every cycle or channel at or above `ALIGNMENT_PASS_PX` is named in a
    warning. One bad cycle or channel is reported, and left out of the intra-cycle value,
    so that it reads as "check or skip this cycle" rather than "all of SBS is misaligned";
    when most cycles are off the value includes them.

    Args:
        aligned (np.ndarray): Aligned SBS data (CYCLE, CHANNEL, I, J).
        channel_order (list[str] or None): Channel names; DAPI must be first for the DAPI metric.
        base_indices (list[int]): Indices of the base channels.
        cycle_labels (list[int], optional): 1-based cycle number of each row of aligned,
            after skipped cycles are removed. Defaults to 1..n.
        upsample_factor (int, optional): Subpixel factor for phase correlation. Defaults to 2.

    Returns:
        dict: `cycle_dapi_shift_residual_max_px`, `intra_cycle_channel_shift_residual_max_px`,
            `dapi_shifts` (CYCLE, 2), `channel_shifts` (CYCLE, BASE, 2, nan = unestimable),
            `channel_residuals` (CYCLE, BASE, 2) and `warnings` (list[str]).
    """
    n_cycles = aligned.shape[0]
    cycle_labels = list(cycle_labels or range(1, n_cycles + 1))
    base_names = [channel_order[i] if channel_order else f"ch{i}" for i in base_indices]
    warnings = []

    has_dapi = aligned.shape[1] > 0 and (
        channel_order is None or channel_order[0] == "DAPI"
    )
    if has_dapi and n_cycles > 1:
        dapi_shifts, _ = calculate_offsets(
            aligned[:, 0], upsample_factor=upsample_factor
        )
        dapi_shifts = np.asarray(dapi_shifts, dtype=float)
        cycle_dapi_max = float(np.max(np.abs(dapi_shifts)))
        for c in range(1, n_cycles):
            if np.max(np.abs(dapi_shifts[c])) >= ALIGNMENT_PASS_PX:
                warnings.append(
                    f"cycle {cycle_labels[c]}: DAPI is shifted {_fmt(dapi_shifts[c])} px "
                    f"from cycle {cycle_labels[0]}"
                )
    else:
        dapi_shifts = np.full((n_cycles, 2), np.nan)
        cycle_dapi_max = 0.0 if has_dapi else float("nan")

    if base_indices and (n_cycles > 1 or len(base_indices) > 1):
        qc = channel_shift_residuals(aligned, base_indices, upsample_factor)
        shifts, residuals = qc["shifts"], qc["residuals"]
        res_max = np.max(np.abs(residuals), axis=-1)
        off = res_max >= ALIGNMENT_PASS_PX
        off_cycles = np.flatnonzero(off.any(axis=1))
        measured_cycles = np.flatnonzero(np.isfinite(res_max).any(axis=1))
        keep = np.isfinite(res_max)
        if len(off_cycles) * 2 <= len(measured_cycles):
            keep[off_cycles] = False
        intra_max = float(np.max(res_max[keep])) if keep.any() else float("nan")
        for c in range(n_cycles):
            for b, name in enumerate(base_names):
                if off[c, b]:
                    warnings.append(
                        f"cycle {cycle_labels[c]}: channel {name} is shifted "
                        f"{_fmt(residuals[c, b])} px from the other channels of its cycle"
                    )
                elif not np.isfinite(shifts[c, b, 0]):
                    warnings.append(
                        f"cycle {cycle_labels[c]}: channel {name} shares too few spots "
                        "with the other cycles to estimate a shift (not measured)"
                    )
            if np.max(np.abs(qc["cycle_shifts"][c])) >= ALIGNMENT_PASS_PX:
                warnings.append(
                    f"cycle {cycle_labels[c]}: base channels are shifted "
                    f"{_fmt(qc['cycle_shifts'][c])} px from the spots of the other cycles"
                )
    else:
        shifts = residuals = np.full((n_cycles, len(base_indices), 2), np.nan)
        intra_max = float("nan")

    print("Alignment QC:")
    print(
        f"  cycle_dapi_shift_residual_max_px:           {cycle_dapi_max:.4f}  (pass < {ALIGNMENT_PASS_PX})"
    )
    print(
        f"  intra_cycle_channel_shift_residual_max_px:  {intra_max:.4f}  (pass < {ALIGNMENT_PASS_PX})"
    )
    if np.isfinite(shifts).any():
        print(
            "  Per-cycle shifts (dy, dx px; channels vs the spots of the other cycles):"
        )
        header = (
            "    cycle  " + "DAPI".ljust(14) + "".join(n.ljust(14) for n in base_names)
        )
        print(header)
        for c in range(n_cycles):
            row = f"    {cycle_labels[c]:<7}" + _fmt(dapi_shifts[c]).ljust(14)
            row += "".join(_fmt(shifts[c, b]).ljust(14) for b in range(len(base_names)))
            print(row)
    for warning in warnings:
        print(f"  Warning: {warning}")
    if warnings and np.isfinite(intra_max) and intra_max < ALIGNMENT_PASS_PX:
        print(
            "  The rest of the tile is aligned; a single off cycle can be dropped with "
            "skip_cycles (check mapping with and without it)."
        )

    return {
        "cycle_dapi_shift_residual_max_px": cycle_dapi_max,
        "intra_cycle_channel_shift_residual_max_px": intra_max,
        "dapi_shifts": dapi_shifts,
        "channel_shifts": shifts,
        "channel_residuals": residuals,
        "warnings": warnings,
    }


def channel_shift_residuals(aligned, base_indices, upsample_factor=2):
    """Measure each base channel's shift against a spot map every channel shares.

    In 4-color SBS a spot is bright in only one base channel per cycle, so two base
    channels of the same cycle can share almost no structure and phase correlation
    between them returns an arbitrary peak. Instead, each channel's spot image
    (Laplacian of Gaussian) is registered to the maximum spot image of the other cycles:
    every spot of the channel is a sequencing spot that is bright in some channel of
    every other cycle. With one cycle the other channels of that cycle are used. The
    shift is estimated on the top and bottom halves of the tile separately; a channel
    whose two estimates disagree by more than `ALIGNMENT_PASS_PX` has too little shared
    structure to estimate a shift and is reported as nan instead of as a spurious shift.

    Args:
        aligned (np.ndarray): Aligned SBS data (CYCLE, CHANNEL, I, J).
        base_indices (list[int]): Indices of the base channels.
        upsample_factor (int, optional): Subpixel factor for phase correlation. Defaults to 2.

    Returns:
        dict: `shifts` (CYCLE, BASE, 2) shift (dy, dx) of each channel, nan where it
            cannot be estimated; `cycle_shifts` (CYCLE, 2) median shift of the channels of
            each cycle; `residuals` (CYCLE, BASE, 2) shift of each channel minus its
            cycle's median, nan where fewer than two channels of the cycle are measured.
    """
    n_cycles, n_bases = aligned.shape[0], len(base_indices)
    spot_max = np.stack(
        [
            np.max([_spot_image(aligned[c, b]) for b in base_indices], axis=0)
            for c in range(n_cycles)
        ]
    )
    shifts = np.full((n_cycles, n_bases, 2), np.nan)
    for c in range(n_cycles):
        others = [o for o in range(n_cycles) if o != c]
        cycle_ref = spot_max[others].max(axis=0) if others else None
        for k, b in enumerate(base_indices):
            moving = _spot_image(aligned[c, b])
            if cycle_ref is not None:
                reference = cycle_ref
            else:
                reference = np.max(
                    [_spot_image(aligned[c, o]) for o in base_indices if o != b], axis=0
                )
            shifts[c, k] = _split_half_shift(reference, moving, upsample_factor)

    measured = np.isfinite(shifts[..., 0])
    cycle_shifts = np.full((n_cycles, 2), np.nan)
    residuals = np.full_like(shifts, np.nan)
    for c in range(n_cycles):
        if measured[c].any():
            cycle_shifts[c] = np.median(shifts[c, measured[c]], axis=0)
        if measured[c].sum() >= 2:
            residuals[c] = shifts[c] - cycle_shifts[c]
    return {"shifts": shifts, "cycle_shifts": cycle_shifts, "residuals": residuals}


def visualize_sbs_alignment(
    aligned_data, channel_names, dapi_cycle, viz_channels, crop_size=300
):
    """Visualize SBS cycle alignment with DAPI reference and RGB base channel overlay.

    Shows 3 locations (corner, center, random) with:
    - Grayscale DAPI background (anatomical reference)
    - RGB overlay of base channels from different cycles
    Color fringing in bases indicates misalignment across cycles.

    Args:
        aligned_data (np.ndarray): Aligned image array (CYCLE, CHANNEL, Y, X).
        channel_names (list): List of channel names.
        dapi_cycle (int): Cycle index for DAPI reference.
        viz_channels (list): List of (cycle_idx, channel_name) tuples for RGB overlay.
            Must have exactly 3 elements for R, G, B channels.
        crop_size (int, optional): Size of zoomed crops in pixels. Defaults to 300.

    Returns:
        matplotlib.figure.Figure: Figure with 3 panels showing alignment at different locations,
            or None if there's an error.

    Example:
        >>> fig = visualize_sbs_alignment(
        ...     aligned,
        ...     ["DAPI", "G", "T", "A", "C"],
        ...     dapi_cycle=0,
        ...     viz_channels=[(0, "G"), (5, "T"), (10, "A")],
        ...     crop_size=300
        ... )
        >>> plt.show()
    """
    import matplotlib.pyplot as plt

    if len(viz_channels) != 3:
        print(
            f"Error: Need exactly 3 channels for RGB overlay, got {len(viz_channels)}"
        )
        return None

    n_cycles, n_channels, height, width = aligned_data.shape

    # Get DAPI reference
    if dapi_cycle >= n_cycles:
        print(f"Error: DAPI cycle {dapi_cycle} out of range (max {n_cycles - 1})")
        return None
    if "DAPI" not in channel_names:
        print(f"Error: DAPI channel not found in {channel_names}")
        return None

    dapi_idx = channel_names.index("DAPI")
    dapi_data = aligned_data[dapi_cycle, dapi_idx]

    # Parse base channels for RGB overlay
    rgb_data = []
    rgb_labels = []
    for cycle_idx, ch_name in viz_channels:
        if cycle_idx >= n_cycles:
            print(f"Error: Cycle {cycle_idx} out of range (max {n_cycles - 1})")
            return None
        if ch_name not in channel_names:
            print(f"Error: Channel '{ch_name}' not found in {channel_names}")
            return None
        ch_idx = channel_names.index(ch_name)
        rgb_data.append(aligned_data[cycle_idx, ch_idx])
        rgb_labels.append(f"C{cycle_idx + 1}-{ch_name}")

    # Define 3 crop locations
    np.random.seed(42)
    locations = [
        ("Top-Left Corner", 50, 50),
        ("Center", (height - crop_size) // 2, (width - crop_size) // 2),
        (
            "Random Location",
            np.random.randint(50, height - crop_size - 50),
            np.random.randint(50, width - crop_size - 50),
        ),
    ]

    # Create figure with 1 row x 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for col_idx, (location_name, y_start, x_start) in enumerate(locations):
        y_end = y_start + crop_size
        x_end = x_start + crop_size

        # Create composite: DAPI (grayscale) + RGB overlay (bases)
        composite = np.zeros((crop_size, crop_size, 3))

        # Add DAPI as grayscale background
        dapi_crop = dapi_data[y_start:y_end, x_start:x_end]
        p2, p98 = np.percentile(dapi_crop, [2, 98])
        dapi_norm = np.clip((dapi_crop - p2) / (p98 - p2 + 1e-8), 0, 1)
        # Set DAPI in all RGB channels for grayscale
        composite[:, :, 0] = dapi_norm
        composite[:, :, 1] = dapi_norm
        composite[:, :, 2] = dapi_norm

        # Overlay base channels as RGB
        for i, img_data in enumerate(rgb_data):
            crop = img_data[y_start:y_end, x_start:x_end]
            p2, p98 = np.percentile(crop, [2, 98])
            crop_norm = np.clip((crop - p2) / (p98 - p2 + 1e-8), 0, 1)
            # Add to composite (additive blending)
            composite[:, :, i] = np.clip(composite[:, :, i] + crop_norm * 0.7, 0, 1)

        axes[col_idx].imshow(composite)
        axes[col_idx].set_title(
            f"{location_name}\n"
            + f"DAPI: C{dapi_cycle + 1} (gray) | "
            + f"R={rgb_labels[0]}, G={rgb_labels[1]}, B={rgb_labels[2]}",
            fontsize=10,
        )
        axes[col_idx].axis("off")

    plt.tight_layout()
    return fig


def plot_cycle_alignment_overlay(
    aligned, channel_names, crop_size=300, cycle_labels=None, upsample_factor=2
):
    """Overlay every cycle's DAPI (green) on the first cycle's DAPI (magenta), one panel per cycle.

    Aligned cycles look white or grey; a misaligned cycle shows every nucleus twice, once
    magenta and once green, so a single off cycle stands out. Each title gives the cycle's
    measured shift (dy, dx px) against the first cycle and the colored fraction (see
    `colored_fraction`). Without a DAPI channel the maximum over the base channels of each
    cycle is used.

    Args:
        aligned (np.ndarray): Aligned SBS data (CYCLE, CHANNEL, I, J).
        channel_names (list[str]): Channel names of aligned.
        crop_size (int, optional): Side of the centered crop shown, in pixels. Defaults to 300.
        cycle_labels (list[int], optional): Cycle number shown for each row of aligned.
            Defaults to 1..n.
        upsample_factor (int, optional): Subpixel factor for the shift estimate. Defaults to 2.

    Returns:
        matplotlib.figure.Figure: The figure, or None with fewer than two cycles.
    """
    n_cycles = aligned.shape[0]
    cycle_labels = list(cycle_labels or range(1, n_cycles + 1))
    if "DAPI" in channel_names:
        images = aligned[:, channel_names.index("DAPI")]
        what = "DAPI"
    else:
        bases = [i for i, ch in enumerate(channel_names) if ch in ("G", "T", "A", "C")]
        images = aligned[:, bases].max(axis=1)
        what = "max of base channels"
    shifts, _ = calculate_offsets(images, upsample_factor=upsample_factor)
    images = center_crop(images, crop_size)
    panels = [
        (
            images[0],
            images[c],
            f"cycle {cycle_labels[c]} vs {cycle_labels[0]}: {_fmt(shifts[c])}",
        )
        for c in range(1, n_cycles)
    ]
    return plot_overlay_grid(
        panels,
        ncols=4,
        suptitle=f"Between cycles: {what}, cycle {cycle_labels[0]} magenta, "
        "cycle k green (white = aligned)",
    )


def plot_channel_alignment_overlay(
    aligned,
    channel_names,
    crop_size=100,
    cycle_labels=None,
    upsample_factor=2,
):
    """Overlay each base channel's spots (green) on the shared spot map (magenta).

    One row per cycle and one column per base channel. The magenta reference is the spot
    map of the other cycles (the reference used by `channel_shift_residuals`), so every
    green spot of an aligned channel sits on a magenta spot and looks white; spots of
    other sequences stay magenta. A shifted channel or cycle shows green spots beside
    their magenta partners. Each panel title gives the measured shift (dy, dx px), or
    n/a when it cannot be estimated, and the fraction of green spot pixels with no
    magenta partner.

    Args:
        aligned (np.ndarray): Aligned SBS data (CYCLE, CHANNEL, I, J).
        channel_names (list[str]): Channel names of aligned.
        crop_size (int, optional): Side of the centered crop shown, in pixels. Defaults to 100.
        cycle_labels (list[int], optional): Cycle number shown for each row of aligned.
            Defaults to 1..n.
        upsample_factor (int, optional): Subpixel factor for the shift estimate. Defaults to 2.

    Returns:
        matplotlib.figure.Figure: The figure, or None without base channels.
    """
    base_indices = [
        i for i, ch in enumerate(channel_names) if ch in ("G", "T", "A", "C")
    ]
    if not base_indices:
        return None
    n_cycles = aligned.shape[0]
    cycle_labels = list(cycle_labels or range(1, n_cycles + 1))
    shifts = channel_shift_residuals(aligned, base_indices, upsample_factor)["shifts"]
    spots = np.stack(
        [
            [_spot_image(center_crop(aligned[c, b], crop_size)) for b in base_indices]
            for c in range(n_cycles)
        ]
    )
    panels = []
    for c in range(n_cycles):
        others = [o for o in range(n_cycles) if o != c]
        for k, b in enumerate(base_indices):
            if others:
                reference = spots[others].max(axis=(0, 1))
            else:
                reference = np.delete(spots[c], k, axis=0).max(axis=0)
            panels.append(
                (
                    reference,
                    spots[c, k],
                    f"cycle {cycle_labels[c]} {channel_names[b]}: {_fmt(shifts[c, k])}",
                )
            )
    return plot_overlay_grid(
        panels,
        ncols=len(base_indices),
        panel_size=2.6,
        percentiles=SPOT_DISPLAY_PERCENTILES,
        colored="moving",
        fraction_percentiles=(0, 100),
        suptitle="Within cycles: channel spots green, spots of the other cycles magenta "
        "(white = aligned; shift dy, dx px)",
    )


def _spot_image(image, sigma=1.0):
    """Laplacian-of-Gaussian spot image scaled to [0, 1] by its 99.9th percentile."""
    image = np.asarray(image, dtype=np.float32)
    if np.ptp(image) == 0:
        return np.zeros_like(image)
    log = -ndimage.gaussian_laplace(image, sigma)
    log = np.clip(log, 0, None)
    scale = np.percentile(log, 99.9)
    if scale <= 0:
        return np.zeros_like(log)
    return np.clip(log / scale, 0, 1)


def _split_half_shift(reference, moving, upsample_factor):
    """Shift (dy, dx) of moving against reference, nan unless both tile halves agree."""
    if not moving.any() or not reference.any():
        return np.array([np.nan, np.nan])
    half = reference.shape[0] // 2
    estimates = []
    for rows in (slice(0, half), slice(half, None)):
        shift, _, _ = phase_cross_correlation(
            moving[rows],
            reference[rows],
            upsample_factor=upsample_factor,
            normalization=None,
        )
        estimates.append(shift)
    if np.max(np.abs(estimates[0] - estimates[1])) > ALIGNMENT_PASS_PX:
        return np.array([np.nan, np.nan])
    return (estimates[0] + estimates[1]) / 2


def _fmt(shift):
    """Format a (dy, dx) shift, or n/a when it is not measured."""
    if not np.all(np.isfinite(shift)):
        return "n/a"
    return f"({shift[0]:+.1f}, {shift[1]:+.1f})"
