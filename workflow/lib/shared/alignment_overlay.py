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


def plot_overlay_grid(
    panels, ncols=4, panel_size=3.5, suptitle=None, percentiles=(1, 99.5)
):
    """Plot a grid of magenta/green overlays with one title per panel.

    Args:
        panels (list[tuple[np.ndarray, np.ndarray, str]]): (reference, moving, title)
            per panel.
        ncols (int, optional): Panels per row. Defaults to 4.
        panel_size (float, optional): Width and height of each panel in inches.
            Defaults to 3.5.
        suptitle (str, optional): Figure title. Defaults to None.
        percentiles (tuple[float, float], optional): Contrast percentiles passed to
            `magenta_green_overlay`. Defaults to (1, 99.5).

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
    for ax, (reference, moving, title) in zip(axes.ravel(), panels):
        overlay = magenta_green_overlay(reference, moving, percentiles)
        ax.imshow(overlay, interpolation="nearest")
        ax.set_title(title, fontsize=9)
    if suptitle:
        fig.suptitle(suptitle, fontsize=11)
    return fig


def center_crop(image, crop_size):
    """Return the centered crop_size x crop_size window of the last two axes."""
    height, width = image.shape[-2:]
    crop_size = min(crop_size, height, width)
    y0 = (height - crop_size) // 2
    x0 = (width - crop_size) // 2
    return image[..., y0 : y0 + crop_size, x0 : x0 + crop_size]


def _stretch(image, percentiles):
    """Scale an image to [0, 1] between its own lower and upper percentiles."""
    image = np.asarray(image, dtype=np.float32)
    low, high = np.percentile(image, percentiles)
    if high <= low:
        return np.zeros_like(image)
    return np.clip((image - low) / (high - low), 0, 1)
