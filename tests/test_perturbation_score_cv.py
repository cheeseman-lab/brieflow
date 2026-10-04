"""Perturbation AUC must be estimated with feature selection inside cross-validation.

Selecting the top-k features on all cells before cross_val_predict leaks the held-out
labels and inflates the AUC on noise, most for units with few cells.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.perturbation_score import calculate_perturbation_scores  # noqa: E402


def _cells(n, n_features=1000, shift=0.0, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, n_features)).astype(np.float32)
    label = np.array(["GENE"] * (n // 2) + ["nontargeting"] * (n - n // 2))
    X[label == "GENE", :5] += shift
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(n_features)])
    df["gene_symbol_0"] = label
    df.index = df.index + 1000
    return df, [f"f{i}" for i in range(n_features)]


@pytest.mark.parametrize("n", [200, 2000])
def test_noise_auc_is_unbiased(n):
    df, cols = _cells(n)
    _, auc = calculate_perturbation_scores(df, "GENE", cols, "gene_symbol_0")
    assert abs(auc - 0.5) < 0.05


def test_real_shift_is_detected():
    df, cols = _cells(400, shift=1.5)
    _, auc = calculate_perturbation_scores(df, "GENE", cols, "gene_symbol_0")
    assert auc > 0.8


def test_output_shape_and_index():
    df, cols = _cells(300, n_features=50)
    scores, auc = calculate_perturbation_scores(df, "GENE", cols, "gene_symbol_0")
    assert scores.index.equals(df.index)
    assert scores.between(0, 1).all()
    assert 0 <= auc <= 1
    scores, auc = calculate_perturbation_scores(
        df.iloc[:50], "GENE", cols, "gene_symbol_0"
    )
    assert scores.isna().all() and np.isnan(auc)
