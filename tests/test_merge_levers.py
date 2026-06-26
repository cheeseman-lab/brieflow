"""Backward-compatibility tests for the newly-parameterized fast-merge levers.

The hardcoded knobs in the fast-merge path (triangle/point/region thresholds, RANSAC
kwargs, polynomial warp degree/iterations/min_correspondences) were promoted to keyword
arguments whose DEFAULTS equal the previous literals. These tests assert that calling the
functions with the new kwargs set to the old literals is identical to calling them with
no kwargs — i.e. existing screens are unaffected.

RANSAC is nondeterministic by default, so wherever RANSAC is involved we pin
`random_state` on both sides; the test then isolates the *threshold* defaults.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from workflow.lib.merge.hash import find_triangles, evaluate_match
from workflow.lib.merge.fast_merge import merge_triangle_hash


def _synthetic_pair(n=250, seed=0):
    """Two cell tables related by a known rotation+scale+translation (so triangles match)."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(0, 1000, size=(n, 2))
    theta = np.deg2rad(1.1)
    scale = 0.27
    R = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    Y = X @ R.T + np.array([40.0, -15.0])
    df0 = pd.DataFrame({"i": X[:, 0], "j": X[:, 1], "cell": np.arange(n), "tile": 0,
                        "well": "A1", "plate": 1})
    df1 = pd.DataFrame({"i": Y[:, 0], "j": Y[:, 1], "cell": np.arange(n), "tile": 0,
                        "well": "A1", "plate": 1})
    return df0, df1


def test_evaluate_match_threshold_defaults_unchanged():
    df0, df1 = _synthetic_pair()
    t0, t1 = find_triangles(df0[["i", "j"]]), find_triangles(df1[["i", "j"]])
    seed = {"random_state": 0}
    r_default = evaluate_match(t0, t1, ransac_kwargs=seed)
    r_explicit = evaluate_match(
        t0, t1, threshold_triangle=0.3, threshold_point=2, threshold_region=50,
        ransac_kwargs=seed,
    )
    assert r_default[0] is not None, "synthetic pair should match"
    np.testing.assert_allclose(r_default[0], r_explicit[0])          # rotation
    np.testing.assert_allclose(r_default[1], r_explicit[1])          # translation
    assert r_default[2] == r_explicit[2]                            # score


def test_warp_kwargs_defaults_unchanged():
    """merge with local_refinement on: no warp_kwargs == warp_kwargs at the old literals."""
    df0, df1 = _synthetic_pair()
    # alignment that maps df0 (phenotype) into df1 (sbs) space: invert the synthetic transform
    theta = np.deg2rad(1.1); scale = 0.27
    R = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    alignment = {"rotation": R, "translation": np.array([40.0, -15.0])}

    m_default = merge_triangle_hash(df0, df1, alignment, threshold=2, local_refinement="polynomial")
    m_explicit = merge_triangle_hash(
        df0, df1, alignment, threshold=2, local_refinement="polynomial",
        warp_kwargs={"degree": 2, "iterations": 2, "min_correspondences": 30},
    )
    pd.testing.assert_frame_equal(m_default, m_explicit)


def test_local_refinement_off_ignores_warp_kwargs():
    """With refinement off, warp_kwargs must have no effect (and the off-path is unchanged)."""
    df0, df1 = _synthetic_pair()
    theta = np.deg2rad(1.1); scale = 0.27
    R = scale * np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    alignment = {"rotation": R, "translation": np.array([40.0, -15.0])}

    m_off = merge_triangle_hash(df0, df1, alignment, threshold=2)
    m_off_kwargs = merge_triangle_hash(
        df0, df1, alignment, threshold=2, warp_kwargs={"degree": 3, "iterations": 5},
    )
    pd.testing.assert_frame_equal(m_off, m_off_kwargs)
