"""Every magenta/green alignment overlay recovers an injected shift and shows it in color.

For each path (SBS cycles, SBS channel grid, merge, phenotype channels) an image is rolled
by a known (dy, dx): the shift in the panel title must match it, the shifted cycle or
channel must be flagged, and the colored fraction (DAPI and phenotype titles, merge image)
must be higher than for the aligned images.
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
    cycle_spot_match,
    plot_cycle_alignment_overlay,
    plot_flagged_channel_overlays,
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
    return float(re.search(r"(\d+)% in one", title).group(1)) / 100


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
    data[:, 4] *= 0.2
    data += rng.normal(0, 0.02, size=data.shape).astype(np.float32)
    return data * 1000 + 100


def test_overlay_white_when_aligned_despite_brightness():
    rng = np.random.default_rng(3)
    centers = [(30, 30), (60, 66), (100, 40)]
    img = _blobs((128, 128), centers, 5.0)
    dim = sum(
        w * _blobs((128, 128), [p], 5.0) for w, p in zip((0.1, 0.4, 0.25), centers)
    )
    noise = rng.normal(0, 0.01, img.shape)
    assert colored_fraction(magenta_green_overlay(img + noise, dim + noise)) < 0.05
    moved = magenta_green_overlay(img + noise, np.roll(dim, 5, axis=1) + noise)
    assert colored_fraction(moved) > 0.3
    assert np.isnan(colored_fraction(magenta_green_overlay(img * 0, img * 0)))


def _titles(fig):
    return [ax.get_title() for ax in fig.axes]


def _row(titles, c, columns=2):
    """Titles of the panels in the row of cycle index c."""
    return titles[c * columns : (c + 1) * columns]


def _matched(title):
    return float(re.search(r"(\d+)% within 1 px", title).group(1)) / 100


def test_spot_match_aligned_with_dim_channel(sbs_cycles):
    match = cycle_spot_match(sbs_cycles, CHANNELS)
    assert (match["matched"] >= 0.9).all()
    assert not match["flagged"].any()
    assert match["counts"].min() >= 0.8 * match["counts"].max()


def test_sbs_cycle_overlay_aligned(sbs_cycles):
    fig = plot_cycle_alignment_overlay(sbs_cycles, CHANNELS, crop_size=96, cycles="all")
    titles = _titles(fig)
    assert len(titles) == 2 * len(sbs_cycles)
    assert not any("OFF" in t for t in titles)
    assert all(_title_fraction(t) <= 0.05 for t in titles if " DAPI: (" in t)
    assert all(_matched(t) >= 0.9 for t in titles if " spots: " in t)
    assert "all cycles within 1 px" in fig._suptitle.get_text()
    assert plot_flagged_channel_overlays(sbs_cycles, CHANNELS) is None


def test_sbs_cycle_overlay_shifted_cycle(sbs_cycles):
    data = sbs_cycles.copy()
    data[2] = np.roll(sbs_cycles[2], SHIFT, axis=(-2, -1))
    match = cycle_spot_match(data, CHANNELS)
    assert match["flagged"].tolist() == [False, False, True, False]
    assert match["matched"][2] <= 0.2
    fig = plot_cycle_alignment_overlay(data, CHANNELS, crop_size=96, cycles="all")
    titles = _titles(fig)
    assert [t for t in titles if "OFF" in t] == _row(titles, 2)
    dapi, spots = _row(titles, 2)
    assert np.allclose(_title_shift(dapi), SHIFT, atol=0.6)
    assert np.allclose(_title_shift(spots), SHIFT, atol=0.6)
    assert _title_fraction(dapi) >= 0.1
    assert _matched(spots) <= 0.2
    assert "off: cycle 3" in fig._suptitle.get_text()


def test_sbs_cycle_overlay_default_selection(sbs_cycles, capsys):
    fig = plot_cycle_alignment_overlay(sbs_cycles, CHANNELS, crop_size=96)
    titles = _titles(fig)
    assert [t.split(":")[0] for t in titles] == [
        f"cycle {c} {v}" for c in (2, 3, 4) for v in ("DAPI", "spots")
    ]
    assert "Showing cycles 2, 3, 4 of 4" in fig._suptitle.get_text()
    table = capsys.readouterr().out
    assert all(f"cycle {c}:" in table for c in range(1, 5))


def test_sbs_cycle_overlay_adds_off_cycles(sbs_cycles):
    data = sbs_cycles.copy()
    data[2] = np.roll(sbs_cycles[2], SHIFT, axis=(-2, -1))
    fig = plot_cycle_alignment_overlay(data, CHANNELS, crop_size=96, cycles=[2])
    titles = _titles(fig)
    assert [t.split(":")[0] for t in titles] == [
        "cycle 2 DAPI",
        "cycle 2 spots",
        "cycle 3 DAPI",
        "cycle 3 spots",
    ]
    assert "OFF" in titles[2] and "Showing cycles 2, 3 of 4" in fig._suptitle.get_text()
    with pytest.raises(ValueError, match="Unknown cycles"):
        plot_cycle_alignment_overlay(data, CHANNELS, cycles=[9])


def test_channel_overlays_selection_keeps_flagged(sbs_cycles):
    data = sbs_cycles.copy()
    data[0, 2] = np.roll(sbs_cycles[0, 2], SHIFT, axis=(0, 1))
    fig = plot_flagged_channel_overlays(data, CHANNELS, cycles=[2], channels=["G"])
    titles = _titles(fig)
    assert titles[0].startswith("cycle 1 T") and "OFF" in titles[0]
    assert titles[1].startswith("cycle 2 G") and "OFF" not in titles[1]
    every = plot_flagged_channel_overlays(
        sbs_cycles, CHANNELS, cycles="all", channels="all"
    )
    assert len([ax for ax in every.axes if ax.images]) == 4 * len(sbs_cycles)


def test_spot_count_flags_collapsed_cycle(sbs_cycles):
    data = sbs_cycles.copy()
    rng = np.random.default_rng(4)
    keep = _blobs(data.shape[-2:], rng.uniform(8, 184, size=(30, 2)), 1.2) * 1000
    data[1, 1:] = 100 + rng.normal(0, 20, size=data[1, 1:].shape) + keep
    match = cycle_spot_match(data, CHANNELS)
    assert match["flagged"][1]
    assert match["counts"][1] < 0.5 * np.median(match["counts"])


def test_sbs_flagged_channel_overlay(sbs_cycles):
    data = sbs_cycles.copy()
    data[0, 2] = np.roll(sbs_cycles[0, 2], SHIFT, axis=(0, 1))
    titles = _titles(plot_flagged_channel_overlays(data, CHANNELS, crop_size=96))
    assert len(titles) == 1 and titles[0].startswith("cycle 1 T")
    assert np.allclose(_title_shift(titles[0]), SHIFT, atol=0.6)


def test_sbs_flagged_channel_overlays_capped(sbs_cycles):
    data = sbs_cycles.copy()
    data[:, 4] = np.roll(sbs_cycles[:, 4], (2, 2), axis=(-2, -1))
    titles = _titles(plot_flagged_channel_overlays(data, CHANNELS, max_panels=2))
    assert len(titles) == 2 and all(" C: " in t for t in titles)


def test_sbs_cycle_overlay_without_per_cycle_dapi(sbs_cycles):
    data = sbs_cycles.copy()
    data[:, 0] = sbs_cycles[0, 0]
    titles = _titles(plot_cycle_alignment_overlay(data, CHANNELS, crop_size=96))
    assert len(titles) == 3
    assert all(" spots: " in t for t in titles)


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
    figs = [plot_merge_alignment_overlay(sbs, ph, df, 5, 0) for df in (good, bad)]
    (good_title, good_overlay), (bad_title, bad_overlay) = (_panel(f, 1) for f in figs)
    context = figs[0].axes[0].images[0].get_array()
    outside = np.asarray(context[:38])
    assert outside[..., 1].max() == 0 and np.allclose(outside[..., 0], outside[..., 2])
    assert np.allclose(_title_shift(good_title), (0, 0), atol=0.6)
    assert np.allclose(_title_shift(bad_title), SHIFT, atol=0.6)
    assert colored_fraction(good_overlay) <= 0.05
    assert colored_fraction(bad_overlay) >= 0.3


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
    assert _title_fraction(after) <= 0.05 and _title_fraction(before) >= 0.3
    assert "GFP" in fig._suptitle.get_text()
    _, metrics = align_phenotype_channels(image, 0, 2, riders=[3], return_metrics=True)
    assert np.allclose(metrics["offset"], _title_shift(before))
    assert len(plot_phenotype_channel_overlay(image, 0, 1, names).axes) == 1
