"""filter_samples_to_wells keeps the samples of the wells a table lists, from TSV or CSV."""

import sys
from pathlib import Path

import pandas as pd
import pytest

_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.shared.configuration_utils import filter_samples_to_wells  # noqa: E402

SAMPLES = pd.DataFrame(
    {
        "sample_fp": ["a", "b", "c", "d"],
        "plate": [1, 1, 1, 2],
        "well": ["A1", "A1", "A2", "A1"],
    }
)


@pytest.mark.parametrize("sep,suffix", [("\t", ".tsv"), (",", ".csv")])
def test_keeps_listed_wells(tmp_path, sep, suffix):
    fp = tmp_path / f"wells{suffix}"
    pd.DataFrame(
        {"plate": [1, 2], "well": ["A1", "A1"], "treatment": ["x", "y"]}
    ).to_csv(fp, sep=sep, index=False)
    out = filter_samples_to_wells(SAMPLES, fp)
    assert out["sample_fp"].tolist() == ["a", "b", "d"]


def test_no_match_raises(tmp_path):
    fp = tmp_path / "wells.tsv"
    pd.DataFrame({"plate": [3], "well": ["B1"]}).to_csv(fp, sep="\t", index=False)
    with pytest.raises(ValueError, match="No samples well"):
        filter_samples_to_wells(SAMPLES, fp)
