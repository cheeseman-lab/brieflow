"""Regression tests for TVN with and without per-batch correction.

`tvn_batch_correction: false` passes batch_col=None to `tvn_on_controls`, which fits
one centering, rotation and scaling on the pooled controls and skips the per-batch
CORAL loop. That path must not depend on the batch labels at all, and must still leave
the controls centered and whitened.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`),
# so align.py's own `from lib.aggregate...` imports resolve too.
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.align import tvn_on_controls  # noqa: E402

PERT_COL = "gene_symbol_0"


def _data(n=400, n_features=5, seed=0):
    rng = np.random.default_rng(seed)
    embeddings = rng.normal(size=(n, n_features))
    metadata = pd.DataFrame(
        {
            PERT_COL: np.where(np.arange(n) % 2 == 0, "nontargeting", "GENE_1"),
            "batch": np.where(np.arange(n) < n // 2, "b0", "b1"),
        }
    )
    embeddings[metadata["batch"] == "b1"] += 3.0
    return embeddings, metadata


def test_global_tvn_ignores_batch_labels():
    embeddings, metadata = _data()
    relabelled = metadata.assign(batch="one_batch")

    result = tvn_on_controls(embeddings, metadata, PERT_COL, "nontargeting", None)
    relabelled_result = tvn_on_controls(
        embeddings, relabelled, PERT_COL, "nontargeting", None
    )

    np.testing.assert_allclose(result, relabelled_result)


def test_global_tvn_whitens_the_pooled_controls():
    embeddings, metadata = _data()
    is_control = (metadata[PERT_COL] == "nontargeting").to_numpy()

    result = tvn_on_controls(embeddings, metadata, PERT_COL, "nontargeting", None)

    np.testing.assert_allclose(result[is_control].mean(axis=0), 0.0, atol=1e-8)
    np.testing.assert_allclose(result[is_control].std(axis=0), 1.0, atol=1e-8)


def test_global_tvn_keeps_the_batch_offset_that_per_batch_tvn_removes():
    embeddings, metadata = _data()
    is_control = (metadata[PERT_COL] == "nontargeting").to_numpy()
    b1 = (metadata["batch"] == "b1").to_numpy()

    global_result = tvn_on_controls(
        embeddings, metadata, PERT_COL, "nontargeting", None
    )
    batch_result = tvn_on_controls(
        embeddings, metadata, PERT_COL, "nontargeting", "batch"
    )

    def control_offset(result):
        b0_mean = result[is_control & ~b1].mean(axis=0)
        return np.abs(result[is_control & b1].mean(axis=0) - b0_mean).max()

    assert control_offset(batch_result) < 1e-6
    assert control_offset(global_result) > 1.0
