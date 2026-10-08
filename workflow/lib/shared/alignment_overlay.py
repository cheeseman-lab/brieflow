"""Magenta/green overlays for checking image alignment by eye.

The reference image is magenta and the moving image green, added together. For display
only, each image is normalized by its local maximum over a window about one object wide,
and the moving image is histogram-matched to the reference, so objects of different
brightness in the two images (fading stain, different scopes) look alike. Aligned objects
then read white or grey, a shift leaves magenta and green fringes, and an object present in
only one image stays fully magenta or green.
"""

import numpy as np


def magenta_green_overlay(reference, moving, window=31):
    """Overlay a reference (magenta) and a moving (green) image, brightness-matched for display.

    Args:
        reference (np.ndarray): 2D reference image.
        moving (np.ndarray): 2D moving image, same shape as reference.
        window (int, optional): Local-normalization window in pixels, about one object
            wide (see `normalize_for_display`). Defaults to 31.

    Returns:
        np.ndarray: RGB image (Y, X, 3) with values in [0, 1].
    """
    from skimage.exposure import match_histograms

    ref = normalize_for_display(reference, window)
    mov = match_histograms(normalize_for_display(moving, window), ref)
    return np.stack([ref, mov, ref], axis=-1)


def normalize_for_display(image, window=31):
    """Scale an image to [0, 1] by its local maximum, for display only.

    The image is lightly smoothed and its background (5th percentile) removed, then each
    pixel is divided by the maximum within window pixels, so every object peaks near 1
    whatever its brightness. Otsu's threshold is the smallest divisor, so background noise
    is not stretched up.

    Args:
        image (np.ndarray): 2D image.
        window (int, optional): Window in pixels, about one object wide. Defaults to 31.

    Returns:
        np.ndarray: Normalized image in [0, 1].
    """
    from scipy import ndimage
    from skimage.filters import threshold_otsu

    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=np.float32), 1.0)
    smooth = np.clip(smooth - np.percentile(smooth, 5), 0, None)
    if not smooth.any():
        return smooth
    local_max = ndimage.maximum_filter(smooth, size=window)
    return np.clip(smooth / np.maximum(local_max, threshold_otsu(smooth)), 0, 1)


def colored_fraction(overlay, mask=None, min_signal=0.25):
    """Share of signal in an overlay that is in only one of the two images.

    A pixel counts as signal when either normalized image reaches min_signal, and as one
    image only when the other is less than half as bright there. Near 0 when the objects
    coincide; a shift raises it. Objects present in only one image also count.

    Args:
        overlay (np.ndarray): RGB overlay from `magenta_green_overlay`.
        mask (np.ndarray, optional): Boolean mask of the pixels to count. Defaults to None (all).
        min_signal (float, optional): Normalized value a pixel needs to count as signal.
            Defaults to 0.25.

    Returns:
        float: Fraction in [0, 1], or nan when there is no signal.
    """
    ref, mov = overlay[..., 0], overlay[..., 1]
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
    panels, ncols=4, panel_size=3.5, suptitle=None, colored=True, window=31
):
    """Plot a grid of magenta/green overlays with one title per panel.

    Args:
        panels (list[tuple]): (reference, moving, title) or (reference, moving, title, mask)
            per panel; the mask limits the colored fraction to part of the panel.
        ncols (int, optional): Panels per row. Defaults to 4.
        panel_size (float, optional): Width and height of each panel in inches.
            Defaults to 3.5.
        suptitle (str, optional): Figure title. Defaults to None.
        colored (bool, optional): Append the colored fraction (see `colored_fraction`) to
            each title. Defaults to True.
        window (int, optional): Display normalization window passed to
            `magenta_green_overlay`. Defaults to 31.

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
        overlay = magenta_green_overlay(reference, moving, window)
        ax.imshow(overlay, interpolation="nearest")
        if colored:
            fraction = colored_fraction(overlay, mask[0] if mask else None)
            title = f"{title}\n{fraction:.0%} in one image only"
        ax.set_title(title, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return fig
