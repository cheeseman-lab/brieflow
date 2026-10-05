"""Every magenta/green alignment overlay recovers an injected shift and shows it in color.

For each path (SBS cycles, SBS channel grid, merge, phenotype channels) an image is rolled
by a known (dy, dx): the shift in the panel title must match it, and the panel's colored
fraction (printed in the title, except for merge) must be clearly higher than for the
aligned images.
"""

import re
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import pytest

matplotlib.use("Agg")

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.merge.merge_utils import plot_merge_alignment_overlay  # noqa: E402
from lib.phenotype.align_channels import (  # noqa: E402
    align_phenotype_channels,
    plot_phenotype_alignment_overlay,
    plot_phenotype_channel_overlay,
)
from lib.sbs.align_cycles import (  # noqa: E402
    plot_channel_alignment_overlay,
    plot_cycle_alignment_overlay,
)
from lib.shared.alignment_overlay import (  # noqa: E402
    colored_fraction,
    magenta_green_overlay,
)

SHIFT = (3, -2)
CHANNELS = ["DAPI", "G", "T", "A", "C"]


def _blobs(shape, points, sigma):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    img = np.zeros(shape, dtype=np.float32)
    for y, x in points:
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2))
    return img


def _title_shift(title):
    """The first (dy, dx) pair in a panel title."""
    dy, dx = re.search(r"\(([+-][\d.]+), ([+-][\d.]+)\)", title).groups()
    return float(dy), float(dx)


def _title_fraction(title):
    return float(re.search(r"(\d+)% colored", title).group(1)) / 100


def _panel(fig, i):
    ax = fig.axes[i]
    return ax.get_title(), ax.images[0].get_array()


@pytest.fixture(scope="module")
def sbs_cycles():
    rng = np.random.default_rng(0)
    shape, n_cycles = (192, 192), 4
    spots = rng.uniform(8, shape[0] - 8, size=(250, 2))
    dapi = _blobs(shape, rng.uniform(15, shape[0] - 15, size=(25, 2)), 6.0)
    data = np.zeros((n_cycles, len(CHANNELS)) + shape, dtype=np.float32)
    for c in range(n_cycles):
        data[c, 0] = dapi
        base = rng.integers(0, 4, size=len(spots))
        for k in range(4):
            data[c, 1 + k] = _blobs(shape, spots[base == k], 1.2)
    data += rng.normal(0, 0.02, size=data.shape).astype(np.float32)
    return data * 1000 + 100


def test_colored_fraction_white_vs_doubled():
    img = _blobs((64, 64), [(32, 32)], 2.0)
    assert colored_fraction(magenta_green_overlay(img, img)) == 0.0
    moved = magenta_green_overlay(img, np.roll(img, 6, axis=1))
    assert colored_fraction(moved) > 0.8
    assert np.isnan(colored_fraction(magenta_green_overlay(img * 0, img * 0)))


def test_sbs_cycle_overlay(sbs_cycles):
    data = sbs_cycles.copy()
    data[2] = np.roll(sbs_cycles[2], SHIFT, axis=(-2, -1))
    fig = plot_cycle_alignment_overlay(data, CHANNELS, crop_size=160)
    aligned_title, _ = _panel(fig, 0)
    shifted_title, overlay = _panel(fig, 1)
    assert "cycle 3" in shifted_title
    assert np.allclose(_title_shift(shifted_title), SHIFT, atol=0.6)
    assert np.allclose(_title_shift(aligned_title), (0, 0), atol=0.6)
    assert _title_fraction(shifted_title) > _title_fraction(aligned_title) + 0.1
    assert colored_fraction(overlay) == pytest.approx(
        _title_fraction(shifted_title), abs=0.01
    )


def test_sbs_channel_overlay(sbs_cycles):
    data = sbs_cycles.copy()
    data[0, 2] = np.roll(sbs_cycles[0, 2], SHIFT, axis=(0, 1))
    fig = plot_channel_alignment_overlay(data, CHANNELS, crop_size=160)
    aligned_title, _ = _panel(fig, 0)
    shifted_title, _ = _panel(fig, 1)
    assert "cycle 1 T" in shifted_title
    assert np.allclose(_title_shift(shifted_title), SHIFT, atol=0.6)
    assert _title_fraction(shifted_title) > _title_fraction(aligned_title) + 0.3


def test_merge_overlay():
    rng = np.random.default_rng(1)
    ph = _blobs((240, 240), rng.uniform(10, 230, size=(60, 2)), 4.0)
    rotation, translation = np.eye(2) * 0.5, np.array([40.0, 30.0])
    yy, xx = np.indices((200, 200)).astype(float)
    from scipy import ndimage

    sbs = ndimage.map_coordinates(ph, [(yy - 40) / 0.5, (xx - 30) / 0.5], order=1)
    good = pd.DataFrame(
        {"tile": [5], "site": [0], "rotation": [rotation], "translation": [translation]}
    )
    bad = good.assign(translation=[translation + np.array(SHIFT, dtype=float)])
    figs = [
        plot_merge_alignment_overlay(sbs, ph, df, 5, 0, crop_size=100)
        for df in (good, bad)
    ]
    (good_title, good_overlay), (bad_title, bad_overlay) = (_panel(f, 1) for f in figs)
    assert np.allclose(_title_shift(good_title), (0, 0), atol=0.6)
    assert np.allclose(_title_shift(bad_title), SHIFT, atol=0.6)
    assert colored_fraction(bad_overlay) > colored_fraction(good_overlay) + 0.3


def test_phenotype_overlay():
    rng = np.random.default_rng(2)
    shape = (256, 256)
    dapi = _blobs(shape, rng.uniform(15, 241, size=(30, 2)), 5.0)
    other = _blobs(shape, rng.uniform(15, 241, size=(30, 2)), 8.0)
    image = (
        np.stack([dapi, other, np.roll(dapi, SHIFT, axis=(0, 1)), other]) * 1000 + 100
    )
    names = ["DAPI", "WGA", "DAPI2", "GFP"]
    fig = plot_phenotype_alignment_overlay(
        image, 0, 2, names, riders=[3], crop_size=200
    )
    (before, _), (after, _) = _panel(fig, 0), _panel(fig, 1)
    assert np.allclose(_title_shift(before), SHIFT, atol=0.6)
    assert np.allclose(_title_shift(after), (0, 0), atol=0.6)
    assert _title_fraction(before) > _title_fraction(after) + 0.3
    assert "GFP" in fig._suptitle.get_text()
    _, metrics = align_phenotype_channels(image, 0, 2, riders=[3], return_metrics=True)
    assert np.allclose(metrics["offset"], _title_shift(before))
    assert len(plot_phenotype_channel_overlay(image, 0, 1, names).axes) == 1
