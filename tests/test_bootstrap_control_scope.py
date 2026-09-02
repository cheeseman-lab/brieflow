"""Regression tests for which controls the construct bootstrap draws its null from.

`bootstrap_control_scope` picks the control pool once an aggregated point is
perturbation x group. "pooled" and "within_group" are the historical scopes and must
stay byte-identical. "reference_group" is for a library where the group is a treatment
acting on the perturbation itself: an over-expressed receptor moved by its ligand has
to be tested against the unliganded control state, so the null is pinned to the vehicle
group rather than to controls that saw the same ligand. Pinning the wrong pool is
silent — it returns p-values, just against the treated baseline — so the scopes must be
demonstrably different pools, and a mis-set reference group must raise.
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`),
# so bootstrap.py's own `from lib.aggregate...` imports resolve too.
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.bootstrap import select_control_pool  # noqa: E402
from lib.aggregate.cell_data_utils import GROUP_KEY_SEP  # noqa: E402

PERT_COL = "gene_symbol_0"
GROUP_COLS = ["treatment"]


def _controls(keys):
    """Controls array as bootstrap_construct.py reads it: key column, then features."""
    return pd.DataFrame(
        {PERT_COL: keys, "feature_0": [float(i) for i in range(len(keys))]}
    )


CONTROLS = _controls(
    [
        "EGFP_1=Ethanol",
        "H2B-EGFP_1=Ethanol",
        "EGFP_1=Corticosterone",
        "H2B-EGFP_1=Corticosterone",
        "EGFP_1=Progesterone",
    ]
)


def _keys(control_pool):
    return list(control_pool[PERT_COL])


# --- Historical scopes ------------------------------------------------------------


def test_pooled_keeps_every_control():
    pool = select_control_pool(CONTROLS, "NR3C1_5=Corticosterone", "pooled")

    pd.testing.assert_frame_equal(pool, CONTROLS)


def test_within_group_keeps_the_constructs_own_group():
    pool = select_control_pool(CONTROLS, "NR3C1_5=Corticosterone", "within_group")

    assert _keys(pool) == ["EGFP_1=Corticosterone", "H2B-EGFP_1=Corticosterone"]


def test_within_group_is_inert_without_a_composite_key():
    """No group_cols means no group half to split on, so the pool stays pooled."""
    controls = _controls(["EGFP_1", "H2B-EGFP_1"])

    pool = select_control_pool(controls, "NR3C1_5", "within_group")

    pd.testing.assert_frame_equal(pool, controls)


def test_within_group_raises_on_a_group_with_no_controls():
    with pytest.raises(
        ValueError, match="No control cells found for group 'Estradiol'"
    ):
        select_control_pool(CONTROLS, "NR3C1_5=Estradiol", "within_group")


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown bootstrap_control_scope: vehicle"):
        select_control_pool(CONTROLS, "NR3C1_5=Ethanol", "vehicle")


# --- Reference group scope --------------------------------------------------------


def test_reference_group_pins_the_pool_across_groups():
    for construct_id in ["NR3C1_5=Corticosterone", "NR3C1_5=Progesterone"]:
        pool = select_control_pool(
            CONTROLS,
            construct_id,
            "reference_group",
            reference_group="Ethanol",
            group_cols=GROUP_COLS,
        )

        assert _keys(pool) == ["EGFP_1=Ethanol", "H2B-EGFP_1=Ethanol"]


def test_reference_group_pool_differs_from_within_group_pool():
    """The whole point of the scope: same construct, a different set of control cells."""
    construct_id = "NR3C1_5=Corticosterone"

    reference_pool = select_control_pool(
        CONTROLS,
        construct_id,
        "reference_group",
        reference_group="Ethanol",
        group_cols=GROUP_COLS,
    )
    within_pool = select_control_pool(CONTROLS, construct_id, "within_group")

    assert set(_keys(reference_pool)).isdisjoint(_keys(within_pool))
    assert all(k.endswith(f"{GROUP_KEY_SEP}Ethanol") for k in _keys(reference_pool))


def test_reference_group_matches_within_group_inside_the_reference_group():
    """A construct already in the reference group must see the same null either way."""
    construct_id = "NR3C1_5=Ethanol"

    reference_pool = select_control_pool(
        CONTROLS,
        construct_id,
        "reference_group",
        reference_group="Ethanol",
        group_cols=GROUP_COLS,
    )

    pd.testing.assert_frame_equal(
        reference_pool, select_control_pool(CONTROLS, construct_id, "within_group")
    )


def test_reference_group_absent_from_the_control_pool_raises():
    with pytest.raises(ValueError, match="'DMSO' is absent from the control pool"):
        select_control_pool(
            CONTROLS,
            "NR3C1_5=Corticosterone",
            "reference_group",
            reference_group="DMSO",
            group_cols=GROUP_COLS,
        )


def test_reference_group_error_lists_the_groups_that_are_present():
    with pytest.raises(ValueError) as excinfo:
        select_control_pool(
            CONTROLS,
            "NR3C1_5=Corticosterone",
            "reference_group",
            reference_group="Etahnol",
            group_cols=GROUP_COLS,
        )

    assert "Corticosterone" in str(excinfo.value)
    assert "Ethanol" in str(excinfo.value)


def test_reference_group_raises_without_group_cols():
    """Ungrouped controls carry no group, so silently falling back to pooled would
    swap the null the operator asked for without a word."""
    controls = _controls(["EGFP_1", "H2B-EGFP_1"])

    with pytest.raises(ValueError, match="needs aggregate group_cols"):
        select_control_pool(
            controls, "NR3C1_5", "reference_group", reference_group="Ethanol"
        )


def test_reference_group_raises_without_a_reference_group_value():
    with pytest.raises(ValueError, match="requires bootstrap_reference_group"):
        select_control_pool(
            CONTROLS, "NR3C1_5=Corticosterone", "reference_group", group_cols=GROUP_COLS
        )


def test_reference_group_spans_several_group_cols():
    """Several group_cols fold into one key joined by GROUP_KEY_SEP, so the reference
    value is the whole joined key, not just the first column's value."""
    controls = _controls(
        ["EGFP_1=Ethanol=6h", "EGFP_1=Ethanol=24h", "EGFP_1=Corticosterone=6h"]
    )

    pool = select_control_pool(
        controls,
        "NR3C1_5=Corticosterone=6h",
        "reference_group",
        reference_group=f"Ethanol{GROUP_KEY_SEP}6h",
        group_cols=["treatment", "timepoint"],
    )

    assert _keys(pool) == ["EGFP_1=Ethanol=6h"]
