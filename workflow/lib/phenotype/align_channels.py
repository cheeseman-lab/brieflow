"""Module for aligning channels in phenotype.

Uses NumPy and scikit-image to provide image alignment between sequencing cycles.
"""

import numpy as np
from lib.shared.image_utils import remove_channels
from lib.shared.align import (
    apply_window,
    calculate_offsets,
    apply_offsets,
)


def align_phenotype_channels(
    image_data,
    target,
    source,
    riders=[],
    upsample_factor=2,
    window=2,
    remove_channel=False,
    verbose=False,
    return_metrics=False,
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
        verbose (bool, optional): If True, print detailed alignment information including
            calculated offsets for source and rider channels. Useful for debugging alignment issues.
            Defaults to False.
        return_metrics (bool, optional): If True, return alignment quality metrics in addition
            to aligned data. Defaults to False.

    Returns:
        np.ndarray: Phenotype data aligned across specified channels.
            If return_metrics=True, returns tuple of (aligned_data, metrics_dict) where
            metrics_dict contains:
            - 'offset': list, the [y, x] offset that was applied
    """
    # Handle stacked vs unstacked data
    if image_data.ndim == 4:
        data_ = image_data.max(axis=0)
        stack = True
    else:
        data_ = image_data.copy()
        stack = False

    # Calculate alignment offsets using phase cross-correlation
    final_offset = phenotype_channel_shift(
        data_, target, source, window, upsample_factor
    )

    # Handle riders and create full offsets array
    if not isinstance(riders, list):
        riders = [riders]
    full_offsets = np.zeros((data_.shape[0], 2))
    full_offsets[[source] + riders] = final_offset

    if verbose:
        print("\n=== Phenotype Channel Alignment Offsets ===")
        print(f"  Target channel (index {target}): no shift (reference)")
        print(
            f"  Source channel (index {source}): shift = {final_offset} pixels (y, x)"
        )
        if riders:
            for rider_idx in riders:
                print(
                    f"  Rider channel (index {rider_idx}): shift = {final_offset} pixels (y, x)"
                )

    # Apply alignment
    if stack:
        aligned = np.array(
            [apply_offsets(slice_, full_offsets) for slice_ in image_data]
        )
    else:
        aligned = apply_offsets(data_, full_offsets)

    # Alignment QC — residual on aligned target/source (+riders)
    to_check = [target, source] + list(riders)
    if aligned.ndim == 4:
        check_data = aligned.max(axis=0)
    else:
        check_data = aligned
    residual, _ = calculate_offsets(
        apply_window(check_data[to_check], window),
        upsample_factor=upsample_factor,
    )
    phenotype_channel_shift_residual_max_px = (
        float(np.max(np.abs(residual[1:]))) if len(residual) > 1 else float("nan")
    )

    print("Alignment QC:")
    print(
        f"  phenotype_channel_shift_residual_max_px:    {phenotype_channel_shift_residual_max_px:.4f}  (pass < 5.0)"
    )

    # Handle channel removal if specified
    if remove_channel == "target":
        channel_order = list(range(image_data.shape[-3]))
        channel_order.remove(source)
        channel_order.insert(target + 1, source)
        aligned = aligned[..., channel_order, :, :]
        aligned = remove_channels(aligned, target)
    elif remove_channel == "source":
        aligned = remove_channels(aligned, source)

    # Return with metrics if requested
    if return_metrics:
        metrics_dict = {
            "offset": final_offset.tolist()
            if hasattr(final_offset, "tolist")
            else list(final_offset),
        }
        return aligned, metrics_dict

    return aligned


def phenotype_channel_shift(data, target, source, window=2, upsample_factor=2):
    """Shift (dy, dx) of the source channel against the target, as `align_phenotype_channels` measures it.

    Args:
        data (np.ndarray): Phenotype image (CHANNEL, I, J).
        target (int): Index of the target channel.
        source (int): Index of the source channel.
        window (int, optional): Alignment window. Defaults to 2.
        upsample_factor (int, optional): Subpixel factor. Defaults to 2.

    Returns:
        np.ndarray: The (dy, dx) shift.
    """
    windowed = apply_window(data[[target, source]], window)
    offsets, _ = calculate_offsets(windowed, upsample_factor=upsample_factor)
    return np.asarray(offsets[1], dtype=float)


def plot_phenotype_alignment_overlay(
    image,
    target,
    source,
    channel_names,
    riders=None,
    window=2,
    upsample_factor=2,
    crop_size=300,
):
    """Overlay the source channel (green) on the target channel (magenta) before and after alignment.

    The shift is measured and applied exactly as `align_phenotype_channels` does, on the
    image before alignment, so the view does not depend on channels removed afterwards.
    Both channels are brightness-matched for display (see `magenta_green_overlay`): after
    alignment objects should read white or grey, and a shift leaves magenta and green
    fringes. Titles give the measured shift and the remaining shift (dy, dx px) and the
    share of signal in one channel only; riders follow the source's shift.

    Args:
        image (np.ndarray): Phenotype image before alignment, (CHANNEL, I, J) or
            (STACK, CHANNEL, I, J).
        target (int): Index of the target (reference) channel.
        source (int): Index of the source channel aligned to the target.
        channel_names (list[str] or dict[int, str]): Channel names by index; a dict needs
            only the target, source and riders.
        riders (list[int], optional): Channels that follow the source's shift. Defaults to None.
        window (int, optional): Alignment window, as in `align_phenotype_channels`. Defaults to 2.
        upsample_factor (int, optional): Subpixel factor, as in `align_phenotype_channels`.
            Defaults to 2.
        crop_size (int, optional): Side of the centered crop shown, in pixels. Defaults to 300.

    Returns:
        matplotlib.figure.Figure: Two panels, before and after alignment.
    """
    from lib.shared.alignment_overlay import center_crop, plot_overlay_grid

    data = image.max(axis=0) if image.ndim == 4 else image
    shift = phenotype_channel_shift(data, target, source, window, upsample_factor)
    pair = data[[target, source]]
    after = apply_offsets(pair, np.array([[0, 0], shift]))
    residual = phenotype_channel_shift(after, 0, 1, window, upsample_factor)
    names = [channel_names[i] for i in riders or []]
    carried = f" (riders {', '.join(names)} follow the source)" if names else ""
    return plot_overlay_grid(
        [
            (
                center_crop(pair[0], crop_size),
                center_crop(pair[1], crop_size),
                f"before: shift {_fmt_shift(shift)}",
            ),
            (
                center_crop(after[0], crop_size),
                center_crop(after[1], crop_size),
                f"after: residual {_fmt_shift(residual)}",
            ),
        ],
        ncols=2,
        panel_size=5,
        window=61,
        suptitle=f"{channel_names[target]} magenta, {channel_names[source]} green{carried}:"
        " white/grey = aligned, magenta/green fringes = shift",
    )


def plot_phenotype_channel_overlay(
    image, reference, moving, channel_names, crop_size=300
):
    """Overlay one phenotype channel (green) on another (magenta) as a sanity check.

    Without channel alignment, DAPI against the cell-boundary channel shows whether nuclei
    sit inside their cells. The stains differ, so the overlay is not expected to be white and
    no shift is measured.

    Args:
        image (np.ndarray): Phenotype image, (CHANNEL, I, J) or (STACK, CHANNEL, I, J).
        reference (int): Index of the channel shown in magenta.
        moving (int): Index of the channel shown in green.
        channel_names (list[str]): Channel names of image.
        crop_size (int, optional): Side of the centered crop shown, in pixels. Defaults to 300.

    Returns:
        matplotlib.figure.Figure: One panel.
    """
    from lib.shared.alignment_overlay import center_crop, plot_overlay_grid

    data = image.max(axis=0) if image.ndim == 4 else image
    return plot_overlay_grid(
        [
            (
                center_crop(data[reference], crop_size),
                center_crop(data[moving], crop_size),
                f"{channel_names[reference]} magenta, {channel_names[moving]} green",
            )
        ],
        ncols=1,
        panel_size=5,
        colored=False,
        suptitle="No channel alignment configured: nuclei (magenta) should sit inside "
        "cells (green)",
    )


def _fmt_shift(shift):
    """Format a (dy, dx) shift."""
    return f"({shift[0]:+.1f}, {shift[1]:+.1f})"
