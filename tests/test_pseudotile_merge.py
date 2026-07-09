# tests/test_pseudotile_merge.py
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.merge.pseudotile_merge import register_stage_frames


def _grid_meta(n_rows, n_cols, spacing, dy=0.0, dx=0.0):
    """Build a regular grid metadata DataFrame with optional offset."""
    records = []
    tile = 0
    for r in range(n_rows):
        for c in range(n_cols):
            records.append(
                {"tile": tile, "y_pos": r * spacing + dy, "x_pos": c * spacing + dx}
            )
            tile += 1
    return pd.DataFrame(records)


@pytest.mark.unit
def test_register_stage_frames_recovers_known_translation():
    ph_meta = _grid_meta(3, 3, spacing=500.0)
    sbs_meta = _grid_meta(3, 3, spacing=500.0, dy=300.0, dx=645.0)

    result = register_stage_frames(sbs_meta, ph_meta)

    assert np.allclose(result["translation"], [300.0, 645.0], atol=1e-6)
    assert np.allclose(result["rotation"], np.eye(2))


@pytest.mark.unit
def test_register_stage_frames_y_then_x_ordering():
    ph_meta = _grid_meta(3, 3, spacing=500.0)
    sbs_meta = _grid_meta(3, 3, spacing=500.0, dy=100.0, dx=700.0)

    result = register_stage_frames(sbs_meta, ph_meta)

    assert np.isclose(result["translation"][0], 100.0, atol=1e-6), "index 0 must be dy"
    assert np.isclose(result["translation"][1], 700.0, atol=1e-6), "index 1 must be dx"


@pytest.mark.unit
def test_register_stage_frames_robust_to_extra_tiles():
    # ph_meta: 5x5 grid centred at (1000, 1000) + offset of (200, 400)
    ph_meta = _grid_meta(5, 5, spacing=500.0, dy=0.0, dx=0.0)
    # Shift ph_meta so its centre matches after applying known (dy, dx)
    # sbs: 3x3 subset — symmetric, so median == centre of 3x3 == (500, 500)
    # ph:  5x5 — median == centre of 5x5 == (1000, 1000)
    # We want translation = [300.0, 500.0]
    # => sbs_center - ph_center = [300.0, 500.0]
    # => sbs grid must be centred at (1300, 1500) relative to ph origin
    ph_meta_shifted = ph_meta.copy()
    # ph grid: y in {0,500,1000,1500,2000}, x in {0,500,1000,1500,2000}; median=(1000,1000)
    # sbs grid: y in {1300,1800,2300}, x in {1500,2000,2500}; median=(1800,2000)
    # translation = (1800-1000, 2000-1000) = (800, 1000) — just verify within atol=5
    sbs_meta = _grid_meta(3, 3, spacing=500.0, dy=1300.0, dx=1500.0)

    result = register_stage_frames(sbs_meta, ph_meta_shifted)

    expected = np.array([800.0, 1000.0])
    assert np.allclose(result["translation"], expected, atol=5.0)
    assert np.allclose(result["rotation"], np.eye(2))
