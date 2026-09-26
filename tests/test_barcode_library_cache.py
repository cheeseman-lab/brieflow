"""Tests for the per-process barcode-library cache in lib.sbs.call_cells.

The barcode library is identical for every tile in a run, so load_barcode_library
parses it once per worker process. The cache is only safe if callers cannot
corrupt it, so two things have to hold:

1. Repeated loads of the same path parse the file exactly once.
2. Each caller gets an independent frame — mutating one must not leak into the
   next load, or one tile's edit becomes every later tile's barcode library.
"""

import sys
from pathlib import Path

import pandas as pd

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.sbs import call_cells as cc  # noqa: E402


def _library(tmp_path):
    fp = tmp_path / "barcode_library.tsv"
    pd.DataFrame({"sgRNA": ["AAAA", "CCCC"], "gene_symbol": ["g1", "g2"]}).to_csv(
        fp, sep="\t", index=False
    )
    return str(fp)


def test_parses_once_and_returns_equal_frames(tmp_path, monkeypatch):
    fp = _library(tmp_path)
    cc._read_barcode_library_cached.cache_clear()

    calls = []
    real_read_csv = pd.read_csv
    monkeypatch.setattr(
        pd, "read_csv", lambda *a, **k: (calls.append(a[0]), real_read_csv(*a, **k))[1]
    )

    first = cc.load_barcode_library(fp)
    second = cc.load_barcode_library(fp)

    assert len(calls) == 1
    pd.testing.assert_frame_equal(first, second)
    assert first is not second


def test_mutating_a_returned_frame_does_not_poison_the_cache(tmp_path):
    fp = _library(tmp_path)
    cc._read_barcode_library_cached.cache_clear()

    first = cc.load_barcode_library(fp)
    first["gene_symbol"] = "clobbered"
    first.drop(index=first.index[0], inplace=True)

    second = cc.load_barcode_library(fp)
    assert list(second["gene_symbol"]) == ["g1", "g2"]
