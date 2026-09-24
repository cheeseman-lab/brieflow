"""Regression test for the phenotype coordinate transform in `plot_merge_example`.

The scaled panels used to place phenotype cells with a per-axis min-max rescale of the
raw `i`/`j` coordinates onto the predicted bounding box:

    X_norm   = (X - X.min(0)) / (X.max(0) - X.min(0))
    X_scaled = X_norm * (Y_pred.max(0) - Y_pred.min(0)) + Y_pred.min(0)

That is a diagonal affine — independent x and y scale plus shift — so it cannot represent
the rotation component of the fitted model. Under a real rotation the bounding box of the
rotated cloud is strictly larger than the rotated bounding box, which introduces a
spurious anisotropic stretch: with a 30 degree rotation the plotted points land a median
~256 px from where the pipeline actually put them, against a 2 px match threshold. A
well-aligned tile therefore rendered as badly offset. The mapping was also driven by the
two extreme cells on each axis, and divided by zero on a tile whose cells share one row.

The panels now plot `Y_pred` directly, which is the position the pipeline matches on
(fitted affine, plus the local warp when `local_refinement` is enabled).
"""

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.merge.merge_utils import plot_merge_example  # noqa: E402

PH_TILE, SBS_SITE = 1, 7


def _rotation(degrees):
    theta = np.deg2rad(degrees)
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])


def _render(X, rotation, translation, threshold=2):
    """Draw one merge preview and hand back its axes without opening a window."""
    Y = X @ rotation.T + translation
    df_ph = pd.DataFrame(X, columns=["i", "j"]).assign(tile=PH_TILE)
    df_sbs = pd.DataFrame(Y, columns=["i", "j"]).assign(tile=SBS_SITE)
    alignment_vec = {
        "tile": PH_TILE,
        "site": SBS_SITE,
        "rotation": rotation,
        "translation": translation,
    }
    plot_merge_example(df_ph, df_sbs, alignment_vec, threshold=threshold)
    return plt.gcf(), Y


@pytest.fixture(autouse=True)
def _headless(monkeypatch):
    monkeypatch.setattr(plt, "show", lambda *a, **k: None)
    yield
    plt.close("all")


def test_phenotype_plotted_at_fitted_affine_position_under_rotation():
    """Scatter coordinates must be the model prediction, not a bounding-box rescale."""
    X = np.random.default_rng(0).uniform(0, 1000, size=(300, 2))
    rotation, translation = _rotation(30), np.array([120.0, -45.0])
    fig, Y_pred = _render(X, rotation, translation)

    # Collection 0 on each panel is the gray SBS backdrop; the rest are phenotype points.
    for ax in fig.axes:
        plotted = np.vstack([c.get_offsets() for c in ax.collections[1:]])
        assert plotted.shape == X.shape
        np.testing.assert_allclose(
            np.sort(plotted, axis=0), np.sort(Y_pred, axis=0), atol=1e-6
        )

    # Guard the test itself: the old mapping is wildly different, so this is not vacuous.
    X_norm = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
    old = X_norm * (Y_pred.max(axis=0) - Y_pred.min(axis=0)) + Y_pred.min(axis=0)
    assert np.median(np.linalg.norm(old - Y_pred, axis=1)) > 100


def test_single_row_tile_does_not_produce_nan_coordinates():
    """A tile whose cells share one coordinate used to divide by zero and plot nothing."""
    X = np.column_stack([np.full(5, 50.0), np.arange(5.0)])
    fig, _ = _render(X, _rotation(30), np.array([120.0, -45.0]))

    for ax in fig.axes:
        plotted = np.vstack([c.get_offsets() for c in ax.collections])
        assert np.isfinite(plotted).all()
