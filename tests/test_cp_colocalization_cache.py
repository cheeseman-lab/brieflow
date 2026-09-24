"""Tests for the per-channel otsu/rank cache in cp_colocalization_all_channels.

The multichannel branch hoists otsu() and rankdata() out of the channel-pair loop,
computing them once per channel instead of once per pair. Two things have to hold:

1. The cached path returns exactly what the per-pair path returned, so the
   optimization cannot move a published phenotype feature.
2. The cache is only valid for otsu. costes thresholds are pair-dependent and a
   fractional threshold is not what the cache holds, so either must fail loudly
   rather than silently yielding otsu numbers.
"""

import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.external.cp_emulator import (  # noqa: E402
    cp_colocalization,
    cp_colocalization_all_channels,
)

CHANNELS = 4


class FakeRegion:
    """Minimal stand-in for the regionprops object the emulator measures."""

    def __init__(self, intensity_image, image):
        self.intensity_image = intensity_image
        self.image = image


@pytest.fixture
def region():
    rng = np.random.default_rng(0)
    intensity_image = rng.integers(0, 4096, size=(24, 24, CHANNELS)).astype(np.uint16)
    image = rng.random((24, 24)) > 0.3
    return FakeRegion(intensity_image, image)


def per_pair_reference(r):
    """The pre-optimization path: one measure_colocalization call per channel pair."""
    return np.array(
        [
            cp_colocalization(r, first, second, mode="multichannel", threshold="otsu")
            for first, second in combinations(range(CHANNELS), 2)
        ]
    ).flatten(order="F")


def test_cached_path_matches_per_pair_path(region):
    cached = cp_colocalization_all_channels(
        region, mode="multichannel", threshold="otsu"
    )
    assert np.allclose(cached, per_pair_reference(region), equal_nan=True)


def test_default_threshold_is_the_otsu_path(region):
    assert np.allclose(
        cp_colocalization_all_channels(region, mode="multichannel"),
        per_pair_reference(region),
        equal_nan=True,
    )


@pytest.mark.parametrize("threshold", ["costes", 0.15])
def test_non_otsu_threshold_raises_instead_of_returning_otsu(region, threshold):
    with pytest.raises(ValueError, match="otsu"):
        cp_colocalization_all_channels(region, mode="multichannel", threshold=threshold)


def test_unexpected_kwarg_raises(region):
    with pytest.raises(ValueError):
        cp_colocalization_all_channels(region, mode="multichannel", bogus=1)
