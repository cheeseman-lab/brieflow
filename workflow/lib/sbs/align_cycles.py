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
from lib.shared.alignment_overlay import colored_fraction, magenta_green_overlay

# a cycle or channel shifted by at least this many pixels is reported as off
ALIGNMENT_PASS_PX = 1.0
# display normalization windows (px), about one nucleus and one spot wide
DAPI_WINDOW = 31
SPOT_WINDOW = 7
# a cycle is flagged when its matched spot fraction is this far below the median cycle
SPOT_MATCH_DROP = 0.2
# or when it has fewer spots than this fraction of the median cycle
SPOT_COUNT_RATIO = 0.5
BASE_CHANNELS = ("G", "T", "A", "C")


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
    compute_qc=False,
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
        compute_qc (bool, optional): If True, print the alignment QC report
            (report_alignment_qc). Off by default because it costs a second alignment
            pass and only produces printed diagnostics. Defaults to False.

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
    if compute_qc:
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


def cycle_spot_match(aligned, channel_names, threshold=0.4, tolerance=1.0):
    """Match each cycle's spots to the spots of all other cycles.

    Per cycle and base channel, the background is removed with a white top-hat and the
    channel is scaled by its own 99.9th percentile (dye balance); spots are local maxima
    above threshold. The union over G/T/A/C, with spots within 2 px merged, is the cycle's
    spot set. Every sequencing spot is lit in some channel every cycle, so in an aligned
    tile most spots of a cycle lie within tolerance of a spot in another cycle; dim spots
    are not detected in every cycle, so the fraction is high but not 1. A cycle is flagged
    when its matched fraction is more than `SPOT_MATCH_DROP` below the median of the cycles,
    or its spot count is below `SPOT_COUNT_RATIO` times the median count.

    Args:
        aligned (np.ndarray): Aligned SBS data (CYCLE, CHANNEL, I, J).
        channel_names (list[str]): Channel names of aligned.
        threshold (float, optional): Spot threshold as a fraction of each channel's 99.9th
            percentile. Defaults to 0.4.
        tolerance (float, optional): Matching distance in pixels. Defaults to 1.0.

    Returns:
        dict: `spots` (list of (N, 2) arrays per cycle), `counts`, `matched` (fraction of
            each cycle's spots within tolerance of a spot in another cycle) and `flagged`
            (bool per cycle); None without base channels or with one cycle.
    """
    from scipy.spatial import cKDTree

    bases = [i for i, ch in enumerate(channel_names) if ch in BASE_CHANNELS]
    n_cycles = aligned.shape[0]
    if not bases or n_cycles < 2:
        return None
    spots = [_union_spots(aligned[c, bases], threshold) for c in range(n_cycles)]
    counts = np.array([len(points) for points in spots])
    matched = np.zeros(n_cycles)
    for c in range(n_cycles):
        others = np.concatenate([spots[o] for o in range(n_cycles) if o != c])
        if len(spots[c]) and len(others):
            distance, _ = cKDTree(others).query(spots[c])
            matched[c] = np.mean(distance <= tolerance)
    flagged = (matched < np.median(matched) - SPOT_MATCH_DROP) | (
        counts < SPOT_COUNT_RATIO * np.median(counts)
    )
    return {"spots": spots, "counts": counts, "matched": matched, "flagged": flagged}


def plot_cycle_alignment_overlay(
    aligned,
    channel_names,
    cycles=None,
    cycle_labels=None,
    upsample_factor=2,
    crop_size=300,
):
    """Show each cycle's alignment in one compact row: DAPI and sequencing spots.

    DAPI column (when DAPI was imaged in every cycle): cycle k (green) on the first cycle
    (magenta), brightness-matched for display (see `magenta_green_overlay`); aligned nuclei
    read white or grey and a shift leaves fringes. Spots column: cycle k's spots (green dots)
    on the spots of all other cycles (magenta dots), from `cycle_spot_match`; a matched spot
    is white and a rolony not detected in cycle k stays magenta. Titles give the measured
    shift (dy, dx px), the DAPI colored fraction (see `colored_fraction`), and the share of
    cycle k's spots within 1 px of a spot in another cycle with the spot count. A cycle is
    marked off when a shift reaches `ALIGNMENT_PASS_PX` or `cycle_spot_match` flags it.
    Crops show the same nucleus- and spot-rich region in every row.

    The numbers are computed and printed as a table for every cycle; only the rows drawn
    are limited to cycles, and off cycles are always drawn.

    Args:
        aligned (np.ndarray): Aligned SBS data (CYCLE, CHANNEL, I, J).
        channel_names (list[str]): Channel names of aligned.
        cycles (list[int] or str, optional): Cycle numbers (as in cycle_labels) to draw, or
            "all". Defaults to None: the second, middle and last cycle.
        cycle_labels (list[int], optional): Cycle number shown for each row of aligned.
            Defaults to 1..n.
        upsample_factor (int, optional): Subpixel factor for the shift estimates. Defaults to 2.
        crop_size (int, optional): Side of each crop, in pixels. Defaults to 300.

    Returns:
        matplotlib.figure.Figure: The figure, or None with fewer than two cycles.
    """
    import matplotlib.pyplot as plt

    n_cycles = aligned.shape[0]
    if n_cycles < 2:
        return None
    cycle_labels = list(cycle_labels or range(1, n_cycles + 1))
    off = np.zeros(n_cycles, dtype=bool)
    dapi = _per_cycle_dapi(aligned, channel_names)
    if dapi is not None:
        dapi_shifts, _ = calculate_offsets(dapi, upsample_factor=upsample_factor)
        off |= np.abs(dapi_shifts).max(axis=1) >= ALIGNMENT_PASS_PX
        dapi_window = _busiest_crop(dapi[0], crop_size)
    match = cycle_spot_match(aligned, channel_names)
    if match is not None:
        base_shifts, _ = calculate_offsets(
            _merged_base_spots(aligned, channel_names), upsample_factor=upsample_factor
        )
        off |= np.abs(base_shifts).max(axis=1) >= ALIGNMENT_PASS_PX
        off |= match["flagged"]
        dots = np.stack([_dot_image(p, aligned.shape[-2:]) for p in match["spots"]])
        spot_window = _busiest_crop(dots.sum(axis=0), crop_size)
    columns = (dapi is not None) + (match is not None)
    print("Cycle alignment (shifts vs first cycle; spot match vs all other cycles):")
    for c in range(n_cycles):
        dapi_text = _fmt(dapi_shifts[c]) if dapi is not None else "-"
        spot_text = (
            f"{_fmt(base_shifts[c])}  {match['matched'][c]:.0%} matched  "
            f"n={match['counts'][c]}"
            if match is not None
            else "-"
        )
        status = "  OFF" if off[c] else ""
        print(f"  cycle {cycle_labels[c]}: DAPI {dapi_text}  spots {spot_text}{status}")
    shown = _selected_cycles(cycles, cycle_labels, off)

    fig, axes = plt.subplots(
        len(shown),
        columns,
        figsize=(max(3.6 * columns, 7.0), 3.5 * len(shown) + 1.2),
        squeeze=False,
        layout="constrained",
    )
    for row, c in enumerate(shown):
        color = "red" if off[c] else "black"
        mark = "  OFF" if off[c] else ""
        panels = list(axes[row])
        if dapi is not None:
            ax = panels.pop(0)
            ax.axis("off")
            if c == 0:
                ax.set_title(f"cycle {cycle_labels[0]} DAPI: reference", fontsize=9)
            else:
                overlay = magenta_green_overlay(
                    dapi[0][dapi_window], dapi[c][dapi_window], DAPI_WINDOW
                )
                ax.imshow(overlay, interpolation="nearest")
                ax.set_title(
                    f"cycle {cycle_labels[c]} DAPI: {_fmt(dapi_shifts[c])}{mark}\n"
                    f"{colored_fraction(overlay):.0%} in one cycle only",
                    fontsize=9,
                    color=color,
                )
        if match is not None:
            ax = panels.pop(0)
            others = np.delete(dots, c, axis=0).max(axis=0)
            overlay = np.stack(
                [others[spot_window], dots[c][spot_window], others[spot_window]],
                axis=-1,
            )
            ax.imshow(overlay, interpolation="nearest")
            ax.set_title(
                f"cycle {cycle_labels[c]} spots: {_fmt(base_shifts[c])}{mark}\n"
                f"{match['matched'][c]:.0%} within 1 px of another cycle, "
                f"n={match['counts'][c]}",
                fontsize=9,
                color=color,
            )
            ax.axis("off")
    flagged = [cycle_labels[c] for c in np.flatnonzero(off)]
    status = (
        f"off: cycle {', '.join(map(str, flagged))}"
        if flagged
        else f"all cycles within {ALIGNMENT_PASS_PX:g} px, spots matched"
    )
    shown_labels = ", ".join(str(cycle_labels[c]) for c in shown)
    fig.suptitle(
        f"Showing cycles {shown_labels} of {n_cycles}; off cycles always shown\n"
        f"DAPI: cycle k green on cycle {cycle_labels[0]} magenta, white = aligned\n"
        "Spots: cycle k green on all other cycles magenta, white = matched,\n"
        f"magenta only = not detected in cycle k. {status}",
        fontsize=10,
    )
    return fig


def plot_flagged_channel_overlays(
    aligned,
    channel_names,
    cycles=None,
    channels=None,
    cycle_labels=None,
    upsample_factor=2,
    crop_size=300,
    max_panels=4,
):
    """Overlay base channels on the spots of the other cycles: flagged ones always, plus a selection.

    Each channel's spots (green) are shown on the spots of the other cycles (magenta),
    cropped to a spot-rich region; a shifted channel's spots appear beside their magenta
    partners. Titles give the channel's shift (dy, dx px) against the other channels of its
    cycle. Channels that `report_alignment_qc` flags within their cycle are always shown
    (largest shift first, at most max_panels); channels and cycles add more panels.

    Args:
        aligned (np.ndarray): Aligned SBS data (CYCLE, CHANNEL, I, J).
        channel_names (list[str]): Channel names of aligned.
        cycles (list[int] or str, optional): Cycle numbers (as in cycle_labels) for the
            extra panels, or "all". Defaults to None: all cycles when channels is given.
        channels (list[str] or str, optional): Base channels for the extra panels, or
            "all". Defaults to None: flagged channels only.
        cycle_labels (list[int], optional): Cycle number shown for each row of aligned.
            Defaults to 1..n.
        upsample_factor (int, optional): Subpixel factor for the shift estimates. Defaults to 2.
        crop_size (int, optional): Side of each crop, in pixels. Defaults to 300.
        max_panels (int, optional): Most panels shown, largest shift first. Defaults to 4.

    Returns:
        matplotlib.figure.Figure: The figure, or None when no channel is flagged or
            selected.
    """
    import matplotlib.pyplot as plt

    base_indices = [i for i, ch in enumerate(channel_names) if ch in BASE_CHANNELS]
    n_cycles = aligned.shape[0]
    if not base_indices or (n_cycles < 2 and len(base_indices) < 2):
        return None
    cycle_labels = list(cycle_labels or range(1, n_cycles + 1))
    residuals = channel_shift_residuals(aligned, base_indices, upsample_factor)[
        "residuals"
    ]
    size = np.nan_to_num(np.abs(residuals).max(axis=-1), nan=0.0)
    flagged = np.argwhere(size >= ALIGNMENT_PASS_PX).tolist()
    flagged = sorted(flagged, key=lambda ck: -size[ck[0], ck[1]])[:max_panels]
    extra = []
    if channels is not None:
        names = [channel_names[b] for b in base_indices]
        wanted = names if channels == "all" else list(channels)
        rows = _selected_cycles(cycles or "all", cycle_labels, np.zeros(n_cycles, bool))
        extra = [[c, names.index(ch)] for c in rows for ch in wanted]
    panels = flagged + [ck for ck in extra if ck not in flagged]
    if not panels:
        return None

    channels = np.stack(
        [
            [_background_subtracted(aligned[c, b]) for b in base_indices]
            for c in range(n_cycles)
        ]
    )
    ncols = min(len(panels), len(base_indices))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(max(3.6 * ncols, 6.0), 3.8 * nrows + 0.6),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (c, k) in zip(axes.ravel(), panels):
        is_flagged = [c, k] in flagged
        others = [o for o in range(n_cycles) if o != c]
        if others:
            reference = channels[others].max(axis=(0, 1))
        else:
            reference = np.delete(channels[c], k, axis=0).max(axis=0)
        window = _busiest_crop(channels[c, k], crop_size)
        overlay = magenta_green_overlay(
            _grow(reference[window]), _grow(channels[c, k][window]), SPOT_WINDOW
        )
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(
            f"cycle {cycle_labels[c]} {channel_names[base_indices[k]]}: "
            f"{_fmt(residuals[c, k])}{'  OFF' if is_flagged else ''}",
            fontsize=9,
            color="red" if is_flagged else "black",
        )
        ax.axis("off")
    fig.suptitle(
        "Flagged channels always shown. Channel spots green, spots of the other cycles "
        "magenta;\n"
        "green beside magenta = shifted (other sequences stay magenta)",
        fontsize=10,
    )
    return fig


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


def _per_cycle_dapi(aligned, channel_names):
    """DAPI of every cycle, or None without DAPI or when it was imaged in one cycle only."""
    if "DAPI" not in channel_names:
        return None
    dapi = aligned[:, channel_names.index("DAPI")]
    if all(np.array_equal(dapi[c], dapi[0]) for c in range(1, len(dapi))):
        return None
    return dapi


def _merged_base_spots(aligned, channel_names):
    """Per-cycle maximum of the base channels' spot images, or None without base channels."""
    bases = [i for i, ch in enumerate(channel_names) if ch in BASE_CHANNELS]
    if not bases:
        return None
    return np.stack(
        [
            np.max([_spot_image(aligned[c, b]) for b in bases], axis=0)
            for c in range(len(aligned))
        ]
    )


def _selected_cycles(cycles, cycle_labels, off):
    """Row indices to draw: the requested cycles (default second, middle, last) plus off ones."""
    n_cycles = len(cycle_labels)
    if cycles == "all":
        chosen = set(range(n_cycles))
    elif cycles is None:
        chosen = {min(1, n_cycles - 1), n_cycles // 2, n_cycles - 1}
    else:
        unknown = set(cycles) - set(cycle_labels)
        if unknown:
            raise ValueError(
                f"Unknown cycles {sorted(unknown)}; cycles are {cycle_labels}"
            )
        chosen = {cycle_labels.index(label) for label in cycles}
    return sorted(chosen | set(np.flatnonzero(off).tolist()))


def _union_spots(channels, threshold):
    """Spots of one cycle: per-channel local maxima after top-hat and dye balance, merged."""
    peaks = np.zeros(channels.shape[-2:], dtype=bool)
    for channel in channels:
        tophat = ndimage.white_tophat(np.asarray(channel, dtype=np.float32), size=9)
        scale = np.percentile(tophat, 99.9)
        if scale <= 0:
            continue
        smooth = ndimage.gaussian_filter(tophat / scale, 1.0)
        peaks |= (smooth == ndimage.maximum_filter(smooth, size=5)) & (
            smooth >= threshold
        )
    labels, n = ndimage.label(ndimage.binary_dilation(peaks))
    if not n:
        return np.zeros((0, 2))
    return np.array(ndimage.center_of_mass(peaks, labels, range(1, n + 1)))


def _dot_image(points, shape):
    """Image with a small dot (3x3 px) at each point."""
    image = np.zeros(shape, dtype=np.float32)
    rows, cols = np.round(points).astype(int).T if len(points) else ([], [])
    image[rows, cols] = 1.0
    return ndimage.maximum_filter(image, size=3)


def _background_subtracted(image, sigma=10.0):
    """Image minus a broad Gaussian background, clipped at zero."""
    image = np.asarray(image, dtype=np.float32)
    return np.clip(image - ndimage.gaussian_filter(image, sigma), 0, None)


def _grow(image, size=3):
    """Grey dilation, so a spot shifted by a pixel between cycles still overlaps itself."""
    return ndimage.maximum_filter(image, size=size)


def _busiest_crop(image, crop_size):
    """Slices of the crop_size window with the most signal."""
    size = min(crop_size, *image.shape)
    density = ndimage.uniform_filter(np.asarray(image, dtype=np.float32), size=size)
    half = size // 2
    inner = density[
        half : image.shape[0] - size + half + 1, half : image.shape[1] - size + half + 1
    ]
    y, x = np.unravel_index(np.argmax(inner), inner.shape)
    return slice(y, y + size), slice(x, x + size)
