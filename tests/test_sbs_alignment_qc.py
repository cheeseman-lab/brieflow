"""Alignment QC on synthetic 4-color SBS cycles, where each spot is bright in one channel per cycle.

Phase correlation between two base channels of a cycle has no shared spots to lock on to,
so the old intra-cycle metric could report shifts of hundreds of pixels on aligned data.
The QC must stay below one pixel on aligned data, name a single shifted channel or cycle
without failing the whole tile, fail when a channel is off in every cycle, and report an
empty channel as not measured.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.sbs.align_cycles import (  # noqa: E402
    channel_shift_residuals,
    plot_channel_alignment_overlay,
    plot_cycle_alignment_overlay,
    report_alignment_qc,
)
from lib.shared.alignment_overlay import magenta_green_overlay  # noqa: E402

CHANNELS = ["DAPI", "G", "T", "A", "C"]
BASES = [1, 2, 3, 4]


def _blobs(shape, points, sigma):
    yy, xx = np.mgrid[: shape[0], : shape[1]]
    img = np.zeros(shape, dtype=np.float32)
    for y, x in points:
        img += np.exp(-((yy - y) ** 2 + (xx - x) ** 2) / (2 * sigma**2))
    return img


@pytest.fixture(scope="module")
def cycles():
    rng = np.random.default_rng(0)
    shape, n_cycles = (192, 192), 5
    spots = rng.uniform(8, shape[0] - 8, size=(250, 2))
    nuclei = rng.uniform(15, shape[0] - 15, size=(25, 2))
    dapi = _blobs(shape, nuclei, 6.0)
    data = np.zeros((n_cycles, len(CHANNELS)) + shape, dtype=np.float32)
    for c in range(n_cycles):
        data[c, 0] = dapi
        base = rng.integers(0, 4, size=len(spots))
        for k in range(4):
            data[c, 1 + k] = _blobs(shape, spots[base == k], 1.2)
    data += rng.normal(0, 0.02, size=data.shape).astype(np.float32)
    return (data * 1000 + 100).astype(np.float32)


def _qc(data):
    return report_alignment_qc(data, CHANNELS, BASES, upsample_factor=2)


def test_aligned_cycles_pass(cycles):
    qc = _qc(cycles)
    assert qc["intra_cycle_channel_shift_residual_max_px"] < 1.0
    assert qc["cycle_dapi_shift_residual_max_px"] < 1.0
    assert np.isfinite(qc["channel_shifts"]).all()
    assert qc["warnings"] == []


def test_one_shifted_channel_is_named_not_failed(cycles):
    data = cycles.copy()
    data[0, 2] = np.roll(cycles[0, 2], (3, -2), axis=(0, 1))
    qc = _qc(data)
    assert np.allclose(qc["channel_residuals"][0, 1], (3, -2), atol=0.6)
    assert qc["intra_cycle_channel_shift_residual_max_px"] < 1.0
    assert any("cycle 1: channel T" in w for w in qc["warnings"])
    assert not any("channel G" in w or "channel A" in w for w in qc["warnings"])


def test_one_shifted_cycle_is_named(cycles):
    data = cycles.copy()
    data[3] = np.roll(cycles[3], (3, -2), axis=(-2, -1))
    qc = _qc(data)
    assert np.allclose(qc["dapi_shifts"][3], (3, -2), atol=0.6)
    assert any(w.startswith("cycle 4: DAPI") for w in qc["warnings"])
    assert any(w.startswith("cycle 4: base channels") for w in qc["warnings"])
    assert qc["intra_cycle_channel_shift_residual_max_px"] < 1.0


def test_channel_off_in_every_cycle_fails(cycles):
    data = cycles.copy()
    data[:, 4] = np.roll(cycles[:, 4], (2, 2), axis=(-2, -1))
    qc = _qc(data)
    assert qc["intra_cycle_channel_shift_residual_max_px"] >= 1.0


@pytest.mark.parametrize("empty", ["flat", "noise"])
def test_empty_channel_is_not_measured(cycles, empty):
    data = cycles.copy()
    rng = np.random.default_rng(1)
    flat = np.full(data.shape[-2:], 100.0, dtype=np.float32)
    data[2, 4] = flat if empty == "flat" else flat + rng.normal(0, 20, flat.shape)
    shifts = channel_shift_residuals(data, BASES)["shifts"]
    assert np.isnan(shifts[2, 3]).all()
    qc = _qc(data)
    assert qc["intra_cycle_channel_shift_residual_max_px"] < 1.0
    assert any("cycle 3: channel C shares too few spots" in w for w in qc["warnings"])


def test_overlay_is_white_when_equal_and_split_when_shifted():
    img = _blobs((64, 64), [(32, 32)], 2.0)
    same = magenta_green_overlay(img, img)
    assert np.allclose(same[..., 0], same[..., 1])
    moved = magenta_green_overlay(img, np.roll(img, 6, axis=1))
    assert moved[32, 32, 0] > 0.9 and moved[32, 32, 1] < 0.2
    assert moved[32, 38, 1] > 0.9 and moved[32, 38, 0] < 0.2


def test_overlay_figures(cycles):
    import matplotlib

    matplotlib.use("Agg")
    assert len(plot_cycle_alignment_overlay(cycles, CHANNELS, crop_size=96).axes) == 4
    assert (
        len(plot_channel_alignment_overlay(cycles, CHANNELS, crop_size=96).axes) == 20
    )
