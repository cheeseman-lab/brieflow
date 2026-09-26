"""Tests for the single-pass label map in reconcile_nuclei_cells.

The function used to run three extra regionprops scans per tile purely to print
diagnostics, and a second regionprops(cells, ...) scan to build the consensus
cell->nucleus map. Two things have to hold:

1. The consensus map derived from the keep_multiple=True map is byte-identical
   to what a separate keep_multiple=False pass produced, so the optimization
   cannot move a published label.
2. verbose is print-only: the returned masks are identical either way.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from skimage.measure import regionprops  # noqa: E402

from lib.shared.segmentation_utils import (  # noqa: E402
    center_pixels,
    reconcile_nuclei_cells,
)


def _masks():
    """Cells 1-3 hold one nucleus each; cell 4 holds two; cell 5 holds none."""
    cells = np.zeros((40, 40), dtype=int)
    nuclei = np.zeros((40, 40), dtype=int)

    layout = {
        1: ((2, 12), [1]),
        2: ((14, 24), [2]),
        3: ((26, 36), [3]),
    }
    for cell_label, ((r0, r1), nuc_labels) in layout.items():
        cells[r0:r1, 2:12] = cell_label
        for i, n in enumerate(nuc_labels):
            nuclei[r0 + 2 + i * 3 : r0 + 5 + i * 3, 4:8] = n

    # cell 4: two nuclei (4, 5) -> dropped by consensus, merged by contained_in_cells
    cells[2:18, 20:34] = 4
    nuclei[4:8, 22:26] = 4
    nuclei[10:14, 28:32] = 5

    # cell 5: no nucleus at all
    cells[24:36, 20:32] = 5

    return nuclei, cells


def _get_unique_label_map(regions, keep_multiple=False):
    """Pre-optimization reference implementation (verbatim from the old body)."""
    label_map = {}
    for region in regions:
        intensity_image = region.intensity_image[region.intensity_image > 0]
        labels = np.unique(intensity_image)
        if keep_multiple:
            label_map[region.label] = labels
        elif len(labels) == 1:
            label_map[region.label] = labels[0]
    return label_map


def test_consensus_map_derivation_matches_separate_pass():
    """{k: v[0] for len(v)==1} over the multi-map == a keep_multiple=False pass."""
    nuclei, cells = _masks()
    nuclei_eroded = center_pixels(nuclei)

    reference = _get_unique_label_map(
        regionprops(cells, intensity_image=nuclei_eroded), keep_multiple=False
    )
    multiple = _get_unique_label_map(
        regionprops(cells, intensity_image=nuclei_eroded), keep_multiple=True
    )
    derived = {k: v[0] for k, v in multiple.items() if len(v) == 1}

    assert derived.keys() == reference.keys()
    assert all(derived[k] == reference[k] for k in reference)
    # The fixture must actually exercise the dropped cases, or this proves nothing.
    assert any(len(v) > 1 for v in multiple.values())  # cell 4
    assert any(len(v) == 0 for v in multiple.values())  # cell 5


@pytest.mark.parametrize("how", ["consensus", "contained_in_cells"])
def test_verbose_does_not_change_masks(how):
    nuclei, cells = _masks()
    quiet = reconcile_nuclei_cells(nuclei.copy(), cells.copy(), how=how)
    loud = reconcile_nuclei_cells(nuclei.copy(), cells.copy(), how=how, verbose=True)

    for a, b in zip(quiet, loud):
        np.testing.assert_array_equal(a, b)


def test_consensus_keeps_only_unique_pairs():
    nuclei, cells = _masks()
    out_nuclei, out_cells = reconcile_nuclei_cells(
        nuclei.copy(), cells.copy(), how="consensus"
    )
    # Cells 1-3 survive; the two-nucleus cell 4 and the empty cell 5 do not.
    assert sorted(np.unique(out_cells)) == [0, 1, 2, 3]
    assert sorted(np.unique(out_nuclei)) == [0, 1, 2, 3]
