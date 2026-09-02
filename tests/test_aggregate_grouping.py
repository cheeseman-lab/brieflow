"""Regression tests for aggregating by an arbitrary label column (split_col/group_cols).

Covers the three bug classes that appear once an aggregated point is
perturbation x group rather than perturbation alone:

1. Grouping must be inert by default and exact when enabled: group_cols=[] must
   reproduce the pre-change single-key output row for row, and group_cols=["treatment"]
   must emit one point per (perturbation, treatment) that still accounts for every input
   cell. The composite label folded into pert_col is what keeps AnnData obs_names unique
   once a perturbation appears under several groups, so it must be unique and must not
   consume the literal group column it was built from.
2. Control identification must read only the perturbation half of a composite key.
   The old whole-string `pert.str.contains(control_key)` let a group value satisfy
   control_key, so a control_key naming a treatment silently relabelled every perturbed
   cell in that treatment as a control, poisoning centering and the bootstrap null.
3. `join_well_annotations` must fail loudly instead of corrupting the cell/feature
   correspondence: an unmapped (plate, well) becomes a NaN group whose cells groupby
   drops, a repeated (plate, well) multiplies cells through the left join, and the
   merge's fresh RangeIndex desyncs the class mask from the separately held features
   frame that split_datasets.py masks with it.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`),
# so aggregate.py's own `from lib.aggregate...` imports resolve too.
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.aggregate import aggregate  # noqa: E402
from lib.aggregate.cell_data_utils import (  # noqa: E402
    GROUP_KEY_SEP,
    control_mask,
    join_well_annotations,
)

PERT_COL = "gene_symbol_0"


def _cells(perturbations, treatments):
    return pd.DataFrame({PERT_COL: perturbations, "treatment": treatments})


def _annotations_tsv(tmp_path, rows):
    fp = tmp_path / "well_annotations.tsv"
    pd.DataFrame(rows).to_csv(fp, sep="\t", index=False)
    return fp


# --- Bug class 1: grouping keys --------------------------------------------------


def test_group_cols_empty_reproduces_single_key_aggregate():
    """Pinned to the literal pre-group_cols output, so the new grouping path stays inert."""
    embeddings = np.arange(12, dtype=float).reshape(6, 2)
    metadata = _cells(
        ["MYC", "TP53", "MYC", "nontargeting", "TP53", "nontargeting"],
        ["DMSO", "DMSO", "Cort", "Cort", "Cort", "DMSO"],
    )
    expected_meta = pd.DataFrame(
        {PERT_COL: ["MYC", "TP53", "nontargeting"], "cell_count": [2, 2, 2]}
    )

    emb, meta = aggregate(embeddings, metadata, PERT_COL, group_cols=[], method="mean")

    assert np.array_equal(emb, np.array([[2.0, 3.0], [5.0, 6.0], [8.0, 9.0]]))
    pd.testing.assert_frame_equal(meta, expected_meta)

    emb_default, meta_default = aggregate(embeddings, metadata, PERT_COL, method="mean")
    assert np.array_equal(emb, emb_default)
    pd.testing.assert_frame_equal(meta, meta_default)


def test_positional_call_still_binds_method_not_group_cols():
    """group_cols must sit after method: 8_aggregate.py calls aggregate() positionally.

    Inserting it before method silently bound AGG_METHOD ("median") to group_cols, and
    list("median") splatted into single characters -> KeyError: 'm'.
    """
    metadata = pd.DataFrame({"gene": ["A", "A", "B", "B"]})
    embeddings = np.array([[1.0], [3.0], [10.0], [30.0]])

    _, meta = aggregate(embeddings, metadata, "gene", "median")

    assert list(meta["gene"]) == ["A", "B"]
    assert "group_cols" not in meta.columns


def test_group_cols_yields_one_row_per_perturbation_and_group():
    embeddings = np.arange(14, dtype=float).reshape(7, 2)
    metadata = _cells(
        ["MYC", "MYC", "MYC", "TP53", "TP53", "nontargeting", "nontargeting"],
        ["DMSO", "DMSO", "Cort", "DMSO", "Cort", "DMSO", "Cort"],
    )

    emb, meta = aggregate(
        embeddings, metadata, PERT_COL, group_cols=["treatment"], method="mean"
    )

    assert len(meta) == 6
    assert emb.shape == (6, 2)
    assert dict(zip(meta[PERT_COL], meta["cell_count"])) == {
        "MYC=DMSO": 2,
        "MYC=Cort": 1,
        "TP53=DMSO": 1,
        "TP53=Cort": 1,
        "nontargeting=DMSO": 1,
        "nontargeting=Cort": 1,
    }
    assert meta["cell_count"].sum() == len(metadata)


def test_composite_pert_col_is_unique_and_keeps_literal_group_col():
    """The composite is the AnnData obs name: MYC under two treatments must not collide,
    while downstream grouping still needs the literal treatment column beside it."""
    embeddings = np.arange(8, dtype=float).reshape(4, 2)
    metadata = _cells(["MYC", "MYC", "TP53", "TP53"], ["DMSO", "Cort", "DMSO", "Cort"])

    _, meta = aggregate(
        embeddings, metadata, PERT_COL, group_cols=["treatment"], method="mean"
    )

    assert meta[PERT_COL].is_unique
    assert set(meta[PERT_COL]) == {"MYC=DMSO", "MYC=Cort", "TP53=DMSO", "TP53=Cort"}
    assert sorted(meta["treatment"]) == ["Cort", "Cort", "DMSO", "DMSO"]
    for perturbation, treatment in zip(meta[PERT_COL], meta["treatment"]):
        assert perturbation.endswith(f"{GROUP_KEY_SEP}{treatment}")


# --- Bug class 2: control identification under composite keys --------------------


@pytest.mark.parametrize(
    "keys,expected",
    [
        (["nontargeting=Cort", "MYC=Cort"], [True, False]),
        (["nontargeting_1=DMSO", "TP53=DMSO"], [True, False]),
        (["nontargeting_1", "TP53"], [True, False]),  # no grouping: unchanged behaviour
    ],
)
def test_control_mask_flags_controls_under_grouping(keys, expected):
    assert list(control_mask(pd.Series(keys), "nontargeting")) == expected


def test_control_mask_ignores_a_control_key_naming_a_group_value():
    """A control_key matching the group half must flag nothing; the old whole-string
    contains() flagged 3 of these 4 perturbed rows as controls."""
    keys = pd.Series(["MYC=DMSO", "TP53=DMSO", "nontargeting_1=DMSO", "MYC=Cort"])

    assert list(control_mask(keys, "DMSO")) == [False, False, False, False]


# --- Bug class 3: well-annotation join -------------------------------------------


def test_control_mask_list_survives_the_control_rename():
    """prepare_alignment_data uniquifies controls to <name>_<pert_id>.

    An exact-only list match stops seeing them after that rename, which silently
    skips TVN normalization instead of failing. The suffixed form must still match,
    while a longer name sharing the prefix (EGFP_10) must not.
    """
    values = pd.Series(
        ["EGFP_1", "EGFP_1_CCTCCGGC", "H2B-EGFP_1_CCTCGCGT", "NES-EGFP_1", "EGFP_10"]
    )

    flagged = values[control_mask(values, ["EGFP_1", "H2B-EGFP_1"])].tolist()

    assert flagged == ["EGFP_1", "EGFP_1_CCTCCGGC", "H2B-EGFP_1_CCTCGCGT"]


def test_join_well_annotations_raises_on_unmapped_well(tmp_path):
    """An unmapped well becomes a NaN group that groupby drops without a word."""
    metadata = pd.DataFrame({"plate": [1, 1], "well": ["A1", "A2"], "cell": [10, 11]})
    fp = _annotations_tsv(tmp_path, [{"plate": 1, "well": "A1", "treatment": "DMSO"}])

    with pytest.raises(ValueError, match="absent from"):
        join_well_annotations(metadata, fp)


def test_join_well_annotations_raises_on_repeated_well(tmp_path):
    """A repeated (plate, well) multiplies cells through the left join, desyncing the
    metadata from the features frame it is carried alongside."""
    metadata = pd.DataFrame({"plate": [1], "well": ["A1"], "cell": [10]})
    fp = _annotations_tsv(
        tmp_path,
        [
            {"plate": 1, "well": "A1", "treatment": "DMSO"},
            {"plate": 1, "well": "A1", "treatment": "Cort"},
        ],
    )

    with pytest.raises(ValueError, match="repeats"):
        join_well_annotations(metadata, fp)


def test_join_well_annotations_preserves_index_for_feature_masking(tmp_path):
    """split_datasets.py masks a separately held features frame with a mask built from
    the joined metadata, so a merge-fresh RangeIndex misaligns cells from their features
    whenever the incoming metadata does not already carry a trivial index."""
    index = [7, 9, 11]
    metadata = pd.DataFrame(
        {"plate": [1, 1, 1], "well": ["A1", "A2", "A1"]}, index=index
    )
    features = pd.DataFrame({"feature_0": [0.1, 0.2, 0.3]}, index=index)
    fp = _annotations_tsv(
        tmp_path,
        [
            {"plate": 1, "well": "A1", "treatment": "DMSO"},
            {"plate": 1, "well": "A2", "treatment": "Cort"},
        ],
    )

    joined = join_well_annotations(metadata, fp)

    assert joined.index.equals(pd.Index(index))
    assert list(joined["treatment"]) == ["DMSO", "Cort", "DMSO"]

    mask = joined["treatment"] == "DMSO"
    assert list(joined[mask]["well"]) == ["A1", "A1"]
    assert list(features[mask]["feature_0"]) == [0.1, 0.3]
