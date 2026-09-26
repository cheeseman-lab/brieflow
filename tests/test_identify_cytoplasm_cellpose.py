"""Bit-identity tests for the vectorized identify_cytoplasm_cellpose.

The per-label Python loop was replaced by a single `np.where`. The loop had an
order-dependent quirk -- it wrote cell label C over `cells == C` and then zeroed
`nuclei == C`, in ascending label order, so a pixel covered by a *smaller*-labelled
nucleus was re-written by the later cell pass and survived as C. The vectorized
expression has to reproduce that quirk, not the intent, or masks shift.

The reference below is a verbatim copy of the pre-refactor loop. If someone
"cleans up" the `>=` to `==`, or reorders it, these fail.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.phenotype.identify_cytoplasm_cellpose import (  # noqa: E402
    identify_cytoplasm_cellpose,
)


def _reference_loop(nuclei, cells):
    """Verbatim pre-refactor implementation, minus the compatibility gate."""
    cytoplasms = np.zeros(cells.shape)
    for cell_label in np.unique(cells):
        if cell_label == 0:
            continue
        nucleus_label = cell_label
        nucleus_coords = np.argwhere(nuclei == nucleus_label)
        cell_coords = np.argwhere(cells == cell_label)
        cytoplasms[cell_coords[:, 0], cell_coords[:, 1]] = cell_label
        cytoplasms[nucleus_coords[:, 0], nucleus_coords[:, 1]] = 0
    return cytoplasms.astype(int)


def _aligned():
    """The ordinary case: each nucleus sits inside its own same-labelled cell."""
    nuclei = np.zeros((20, 20), dtype=int)
    cells = np.zeros((20, 20), dtype=int)
    cells[2:9, 2:9] = 1
    nuclei[4:7, 4:7] = 1
    cells[11:19, 3:12] = 2
    nuclei[13:16, 5:9] = 2
    cells[3:9, 12:19] = 3
    nuclei[5:8, 14:17] = 3
    return nuclei, cells


def _smaller_nucleus_over_larger_cell():
    """Exercise the N < C quirk: nucleus 1 overlaps cell 2's territory.

    The loop processes cell 1 (zeroing nucleus 1's pixels), then cell 2, which
    re-writes 2 over the pixels nucleus 1 shares with cell 2. Those pixels
    therefore survive as 2 -- an `np.where(nuclei > 0, 0, cells)` would wrongly
    zero them.
    """
    nuclei = np.zeros((20, 20), dtype=int)
    cells = np.zeros((20, 20), dtype=int)
    cells[2:10, 2:10] = 1
    cells[10:18, 2:10] = 2
    nuclei[4:12, 4:8] = 1  # straddles the cell 1 / cell 2 boundary
    nuclei[13:16, 4:8] = 2
    return nuclei, cells


@pytest.mark.parametrize(
    "fixture", [_aligned, _smaller_nucleus_over_larger_cell], ids=lambda f: f.__name__
)
def test_matches_the_pre_refactor_loop(fixture):
    nuclei, cells = fixture()
    assert set(np.unique(nuclei).tolist()) == set(np.unique(cells).tolist()), (
        "fixture must satisfy the shared-label-set precondition"
    )

    out = identify_cytoplasm_cellpose(nuclei, cells)
    np.testing.assert_array_equal(out, _reference_loop(nuclei, cells))
    assert out.dtype == _reference_loop(nuclei, cells).dtype


def test_the_quirk_fixture_actually_exercises_the_quirk():
    """Guard the guard: a naive `nuclei > 0` mask must disagree on this fixture."""
    nuclei, cells = _smaller_nucleus_over_larger_cell()
    naive = np.where(nuclei > 0, 0, cells).astype(int)
    assert not np.array_equal(naive, _reference_loop(nuclei, cells)), (
        "fixture no longer distinguishes the ascending-label semantics"
    )


def test_mismatched_label_sets_return_none():
    """The gate is set equality, not count equality -- same count, different labels."""
    nuclei = np.zeros((10, 10), dtype=int)
    cells = np.zeros((10, 10), dtype=int)
    cells[1:4, 1:4] = 1
    cells[6:9, 6:9] = 2
    nuclei[2:3, 2:3] = 1
    nuclei[7:8, 7:8] = 3  # same number of unique labels, different label set

    assert len(np.unique(nuclei)) == len(np.unique(cells))
    assert identify_cytoplasm_cellpose(nuclei, cells) is None
