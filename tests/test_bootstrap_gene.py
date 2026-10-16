"""Regression tests for the per-gene bootstrap, which keeps exceed counts instead of null arrays.

The p-values must follow the stored-null definition (the share of null medians above the
observed median, folded to two tails), the gene null must be the per-simulation median
of its constructs' nulls, and a rerun must reproduce the same p-values.
"""

import sys
from pathlib import Path

import numpy as np

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.bootstrap import (  # noqa: E402
    _median_inplace,
    bootstrap_seed,
    calculate_pvals,
    pvals_from_exceed_counts,
    run_gene_bootstrap,
)


def _controls(rng, sizes, n_features):
    ids = np.repeat([f"nt_{i}" for i in range(len(sizes))], sizes)
    features = rng.normal(size=(len(ids), n_features))
    return np.column_stack([ids, features]).astype(object)


def test_median_matches_numpy_including_ties_and_even_lengths():
    rng = np.random.default_rng(0)
    for n in [1, 2, 3, 4, 7, 10, 101]:
        for values in (rng.normal(size=n), rng.integers(0, 3, size=n).astype(float)):
            assert _median_inplace(values.copy()) == np.median(values)


def test_exceed_counts_give_the_stored_null_pvalues():
    rng = np.random.default_rng(1)
    nulls = rng.normal(size=(500, 4))
    observed = np.array([-2.0, 0.0, 0.3, np.nan])
    counts = (nulls > observed).sum(axis=0)

    pvals = pvals_from_exceed_counts(counts, len(nulls))

    one_tail = (nulls > observed).mean(axis=0)
    expected = np.array([1 - x if x > 0.5 else x for x in one_tail]) * 2
    np.testing.assert_array_equal(pvals, expected)
    np.testing.assert_array_equal(calculate_pvals(nulls, observed), expected)


def test_gene_bootstrap_is_reproducible_and_scores_an_extreme_gene():
    rng = np.random.default_rng(2)
    controls = _controls(rng, [30, 12], 3)
    observed = np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]])

    first = run_gene_bootstrap(
        observed, [controls, controls], [4, 6], observed[0], 3000, bootstrap_seed("A")
    )
    again = run_gene_bootstrap(
        observed, [controls, controls], [4, 6], observed[0], 3000, bootstrap_seed("A")
    )

    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)
    construct_pvals, gene_pvals = first
    assert construct_pvals.shape == (2, 3)
    assert np.all(construct_pvals[:, 0] == 0) and gene_pvals[0] == 0
    assert np.all(construct_pvals[:, 1:] > 0.2) and np.all(gene_pvals[1:] > 0.2)
