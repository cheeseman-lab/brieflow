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
    normalize_percentile=False,
    lower_percentile=1,
    upper_percentile=99,
    verbose=False,
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
        normalize_percentile (bool, optional): If True, apply percentile normalization before
            cross-correlation. Improves alignment when intensity varies across cycles. Defaults to False.
        lower_percentile (float, optional): Lower percentile for normalization. Defaults to 1.
        upper_percentile (float, optional): Upper percentile for normalization. Defaults to 99.
        verbose (bool, optional): If True, print detailed alignment information including
            calculated offsets for source and rider channels. Useful for debugging alignment issues.
            Defaults to False.

    Returns:
        np.ndarray: Phenotype data aligned across specified channels.
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
    offsets = calculate_offsets(
        windowed,
        upsample_factor=upsample_factor,
        normalize=normalize_percentile,
        lower_percentile=lower_percentile,
        upper_percentile=upper_percentile,
    )

    # Handle riders and create full offsets array
    if not isinstance(riders, list):
        riders = [riders]
    full_offsets = np.zeros((data_.shape[0], 2))
    full_offsets[[source] + riders] = offsets[1]

    if verbose:
        print("\n=== Phenotype Channel Alignment Offsets ===")
        if normalize_percentile:
            print(f"  Percentile normalization: [{lower_percentile}, {upper_percentile}]")
        print(f"  Target channel (index {target}): no shift (reference)")
        print(f"  Source channel (index {source}): shift = {offsets[1]} pixels (y, x)")
        if riders:
            for rider_idx in riders:
                print(
                    f"  Rider channel (index {rider_idx}): shift = {offsets[1]} pixels (y, x)"
                )

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

    return aligned


def visualize_phenotype_alignment(
    aligned_data, channel_names, viz_channels, crop_size=300
):
    """Visualize phenotype channel alignment with grayscale and RGB overlays.

    Shows 16 locations (4x4 grid). First channel shown in grayscale with
    remaining 3 channels as RGB composite overlaid. Color fringing indicates misalignment.

    Args:
        aligned_data (np.ndarray): Aligned image array (CHANNEL, Y, X).
        channel_names (list): List of all channel names.
        viz_channels (list): List of 4 channel names to visualize
            (1st=grayscale, 2nd-4th=RGB overlay).
        crop_size (int, optional): Size of zoomed crops in pixels. Defaults to 300.

    Returns:
        matplotlib.figure.Figure: Figure with 4x4 grid of alignment visualizations,
            or None if there's an error.

    Example:
        >>> fig = visualize_phenotype_alignment(
        ...     aligned_image,
        ...     ["DAPI", "TUBULIN", "GH2AX", "PHALLOIDIN"],
        ...     ["DAPI", "TUBULIN", "GH2AX", "PHALLOIDIN"],
        ...     crop_size=300
        ... )
        >>> plt.show()
    """
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec

    if len(viz_channels) != 4:
        print(
            f"Error: Need exactly 4 channels (1 grayscale + 3 RGB), got {len(viz_channels)}"
        )
        return None

    _, height, width = aligned_data.shape

    # Get channel indices
    channel_indices = []
    for ch_name in viz_channels:
        if ch_name not in channel_names:
            print(f"Error: Channel '{ch_name}' not found in {channel_names}")
            return None
        channel_indices.append(channel_names.index(ch_name))

    # Define 16 crop locations (4 corners, 4 edges, 4 quadrant centers, 4 random)
    margin = 50
    mid_y = (height - crop_size) // 2
    mid_x = (width - crop_size) // 2

    locations = [
        # Row 1: Corners and top edge
        ("Top-Left Corner", margin, margin),
        ("Top Edge", margin, mid_x),
        ("Top-Right Corner", margin, width - crop_size - margin),
        (
            "Random 1",
            np.random.randint(margin, height - crop_size - margin),
            np.random.randint(margin, width - crop_size - margin),
        ),
        # Row 2: Left edge, quadrant centers
        ("Left Edge", mid_y, margin),
        ("Top-Left Quadrant", mid_y // 2, mid_x // 2),
        ("Top-Right Quadrant", mid_y // 2, mid_x + mid_x // 2),
        (
            "Random 2",
            np.random.randint(margin, height - crop_size - margin),
            np.random.randint(margin, width - crop_size - margin),
        ),
        # Row 3: More quadrant centers and right edge
        ("Bottom-Left Quadrant", mid_y + mid_y // 2, mid_x // 2),
        ("Center", mid_y, mid_x),
        ("Bottom-Right Quadrant", mid_y + mid_y // 2, mid_x + mid_x // 2),
        ("Right Edge", mid_y, width - crop_size - margin),
        # Row 4: Bottom edge, corner, and random
        (
            "Random 3",
            np.random.randint(margin, height - crop_size - margin),
            np.random.randint(margin, width - crop_size - margin),
        ),
        ("Bottom-Left Corner", height - crop_size - margin, margin),
        ("Bottom Edge", height - crop_size - margin, mid_x),
        (
            "Bottom-Right Corner",
            height - crop_size - margin,
            width - crop_size - margin,
        ),
    ]

    # Create figure with 4 rows x 4 columns
    fig = plt.figure(figsize=(20, 20))
    gs = GridSpec(4, 4, figure=fig, hspace=0.3, wspace=0.2)

    for idx, (location_name, y_start, x_start) in enumerate(locations):
        y_end = y_start + crop_size
        x_end = x_start + crop_size

        # Create combined RGBA image
        rgba = np.zeros((crop_size, crop_size, 3))

        # Add grayscale (first channel) as base layer
        gray_crop = aligned_data[channel_indices[0], y_start:y_end, x_start:x_end]
        p2, p98 = np.percentile(gray_crop, [2, 98])
        gray_norm = np.clip((gray_crop - p2) / (p98 - p2 + 1e-8), 0, 1)

        # Add RGB composite (channels 2-4) overlaid on grayscale
        rgb = np.zeros((crop_size, crop_size, 3))
        for i, ch_idx in enumerate(channel_indices[1:]):
            crop = aligned_data[ch_idx, y_start:y_end, x_start:x_end]
            p2, p98 = np.percentile(crop, [2, 98])
            crop_norm = np.clip((crop - p2) / (p98 - p2 + 1e-8), 0, 1)
            rgb[:, :, i] = crop_norm

        # Blend: 50% grayscale, 50% RGB
        for i in range(3):
            rgba[:, :, i] = 0.5 * gray_norm + 0.5 * rgb[:, :, i]

        ax = fig.add_subplot(gs[idx // 4, idx % 4])
        ax.imshow(rgba)
        ax.set_title(
            f"{location_name}\n"
            + f"Gray: {viz_channels[0]} | RGB: R={viz_channels[1]}, G={viz_channels[2]}, B={viz_channels[3]}",
            fontsize=8,
        )
        ax.axis("off")

    plt.tight_layout()
    return fig
