"""Magenta/green overlays for checking image alignment by eye.

The reference image is shown in magenta and the moving image in green, blended
additively. Where the two images agree the overlay is white or grey; where they are
misaligned every object appears twice, once magenta and once green, so the size and
direction of a shift can be read off the image.
"""

import numpy as np


def magenta_green_overlay(reference, moving, percentiles=(1, 99.5)):
    """Blend a reference (magenta) and a moving (green) image into an RGB overlay.

    Each image is contrast-stretched between its own percentiles, so both colors are
    comparable even when the two images have different intensity ranges.

    Args:
        reference (np.ndarray): 2D reference image, shown in magenta.
        moving (np.ndarray): 2D moving image, shown in green; same shape as reference.
        percentiles (tuple[float, float], optional): Lower and upper percentiles used to
            stretch each image to [0, 1]. Defaults to (1, 99.5).

    Returns:
        np.ndarray: RGB image (Y, X, 3) with values in [0, 1].
    """
    ref = _stretch(reference, percentiles)
    mov = _stretch(moving, percentiles)
    return np.stack([ref, mov, ref], axis=-1)


def colored_fraction(overlay, mask=None, basis="both", min_signal=0.25):
    """Fraction of signal pixels in an overlay that are colored rather than white or grey.

    A pixel is colored when one image is less than half as bright as the other there. Aligned
    copies of the same structure give a value near 0; a shift makes both copies colored.
    Images of different stains or resolutions are not white even when aligned, so the
    fraction is only meaningful for two images of the same structure.

    Args:
        overlay (np.ndarray): RGB overlay from `magenta_green_overlay`.
        mask (np.ndarray, optional): Boolean mask of the pixels to count. Defaults to None (all).
        basis (str, optional): "both" counts pixels where either image is bright; "moving"
            counts only pixels where the green image is bright and asks whether magenta is
            missing there, for references that hold more objects than the moving image.
            Defaults to "both".
        min_signal (float, optional): Stretched intensity a pixel needs to count as signal.
            Defaults to 0.25.

    Returns:
        float: Colored fraction in [0, 1], or nan when no pixel has signal.
    """
    ref, mov = overlay[..., 0], overlay[..., 1]
    if basis == "moving":
        keep = mov >= min_signal
        colored = ref < 0.5 * mov
    else:
        signal = np.maximum(ref, mov)
        keep = signal >= min_signal
        colored = np.minimum(ref, mov) < 0.5 * signal
    if mask is not None:
        keep &= mask
    if not keep.any():
        return float("nan")
    return float(colored[keep].mean())


def center_crop(image, crop_size):
    """Return the centered crop_size x crop_size window of the last two axes."""
    height, width = image.shape[-2:]
    crop_size = min(crop_size, height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    return image[..., y0 : y0 + crop_size, x0 : x0 + crop_size]


def plot_overlay_grid(
    panels,
    ncols=4,
    panel_size=3.5,
    suptitle=None,
    percentiles=(1, 99.5),
    colored="both",
    fraction_percentiles=None,
):
    """Plot a grid of magenta/green overlays with one title per panel.

    Args:
        panels (list[tuple]): (reference, moving, title) or (reference, moving, title, mask)
            per panel; the mask limits the colored fraction to part of the panel.
        ncols (int, optional): Panels per row. Defaults to 4.
        panel_size (float, optional): Width and height of each panel in inches.
            Defaults to 3.5.
        suptitle (str, optional): Figure title. Defaults to None.
        percentiles (tuple[float, float], optional): Contrast percentiles passed to
            `magenta_green_overlay`. Defaults to (1, 99.5).
        colored (str or None, optional): Basis of the colored fraction appended to each
            title (see `colored_fraction`), or None to leave it out. Defaults to "both".
        fraction_percentiles (tuple[float, float], optional): Contrast percentiles of the
            overlay the colored fraction is computed on, when it should differ from the
            display. Defaults to None (same as percentiles).

    Returns:
        matplotlib.figure.Figure: The figure, or None if there are no panels.
    """
    import matplotlib.pyplot as plt

    if not panels:
        return None
    ncols = max(1, min(ncols, len(panels)))
    nrows = int(np.ceil(len(panels) / ncols))
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(panel_size * ncols, panel_size * nrows + (0.6 if suptitle else 0)),
        squeeze=False,
        layout="constrained",
    )
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (reference, moving, title, *mask) in zip(axes.ravel(), panels):
        overlay = magenta_green_overlay(reference, moving, percentiles)
        ax.imshow(overlay, interpolation="nearest")
        if colored:
            if fraction_percentiles is not None:
                overlay = magenta_green_overlay(reference, moving, fraction_percentiles)
            fraction = colored_fraction(overlay, mask[0] if mask else None, colored)
            title = f"{title}\n{fraction:.0%} colored"
        ax.set_title(title, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return fig


def _stretch(image, percentiles):
    """Scale an image to [0, 1] between its own lower and upper percentiles."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, percentiles)
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)
