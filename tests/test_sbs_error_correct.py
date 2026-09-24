"""Tests for barcode error correction in the SBS call_cells path.

error_correct_reads was changed from an O(n x library) distance scan to an O(1)
hamming-1 lookup index. The correction contract has to be identical either way:

1. An exact library match is returned unchanged.
2. A read within max_distance of exactly one barcode is corrected to it.
3. A read beyond max_distance of every barcode is returned unchanged.
4. A read equidistant from two barcodes is ambiguous and returned unchanged --
   silently picking one would assign reads to the wrong perturbation.
"""

import sys
from pathlib import Path

import pandas as pd

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.sbs.call_cells import error_correct_reads  # noqa: E402

LIBRARY = pd.Series(["AAAAAAAAAAAA", "CCCCCCCCCCCC", "GGGGGGGGGGGG"])


def correct(reads, library=LIBRARY, max_distance=1):
    return error_correct_reads(
        pd.Series(reads), library, max_distance=max_distance, distance_metric="hamming"
    )


def test_exact_matches_pass_through_unchanged():
    out = correct(["AAAAAAAAAAAA", "CCCCCCCCCCCC"])

    assert out.iloc[0] == "AAAAAAAAAAAA"
    assert out.iloc[1] == "CCCCCCCCCCCC"


def test_reads_within_max_distance_are_corrected():
    out = correct(["AAAAAAAAAAAC", "AAAAAAAAAAAT"])  # both 1 edit from barcode 0

    assert out.iloc[0] == "AAAAAAAAAAAA"
    assert out.iloc[1] == "AAAAAAAAAAAA"


def test_reads_beyond_max_distance_are_left_alone():
    out = correct(["AAAAAAAAAAGG"])  # 2 edits from barcode 0, 10 from the rest

    assert out.iloc[0] == "AAAAAAAAAAGG"


def test_ambiguous_reads_are_not_corrected():
    """Equidistant from two barcodes -- correcting would invent a perturbation call."""
    library = pd.Series(["AAAAAAAAAAAC", "AAAAAAAAAAAG"])
    out = correct(["AAAAAAAAAAAA"], library=library)  # 1 edit from both

    assert out.iloc[0] == "AAAAAAAAAAAA"
