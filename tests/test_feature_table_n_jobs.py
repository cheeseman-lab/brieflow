"""Tests for the n_jobs paths of feature_table_multichannel.

n_jobs is a pure throughput knob, so the threaded path must return exactly what
the sequential path returns for scalar, length-1-iterable and multi-element
features alike. The default is 1 because region-level threading measured ~0.76x
the throughput of tile-level parallelism; a caller that raises it must still get
identical numbers.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.shared.feature_table_utils import (  # noqa: E402
    feature_table_multichannel,
)

FEATURES = {
    "area": lambda r: r.area,  # scalar
    "mean": lambda r: float(np.asarray(r.image_intensity).mean()),  # scalar, intensity
    "bbox": lambda r: r.bbox,  # 4-element iterable
    "height": lambda r: (r.bbox[2] - r.bbox[0],),  # length-1 iterable
}


@pytest.fixture
def data_and_labels():
    rng = np.random.default_rng(0)
    labels = np.zeros((32, 32), dtype=int)
    labels[2:10, 2:10] = 1
    labels[12:22, 4:18] = 2
    labels[24:30, 20:28] = 3
    data = rng.random((3, 32, 32))  # channel-first, as the emulator passes it
    return data, labels


def test_default_is_sequential():
    """The signature default must stay 1 -- -1 oversubscribes under a tile pool."""
    import inspect

    sig = inspect.signature(feature_table_multichannel)
    assert sig.parameters["n_jobs"].default == 1


@pytest.mark.parametrize("n_jobs", [2, 4, -1])
def test_threaded_path_matches_sequential(data_and_labels, n_jobs):
    data, labels = data_and_labels
    sequential = feature_table_multichannel(data, labels, FEATURES, n_jobs=1)
    threaded = feature_table_multichannel(data, labels, FEATURES, n_jobs=n_jobs)

    pd.testing.assert_frame_equal(sequential, threaded)
    # Guard against a fixture that silently produces nothing to compare.
    assert len(sequential) == 3
    assert "bbox_0" in sequential and "bbox_3" in sequential  # multi-element expanded
    assert "height" in sequential and "area" in sequential


def test_zero_regions_returns_empty_frame_on_both_paths():
    data = np.zeros((3, 8, 8))
    labels = np.zeros((8, 8), dtype=int)
    for n_jobs in (1, 4):
        out = feature_table_multichannel(data, labels, FEATURES, n_jobs=n_jobs)
        assert out.empty
