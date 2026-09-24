"""Regression tests for which controls the construct bootstrap draws its null from.

`bootstrap_control_scope` picks the control pool once an aggregated point is
perturbation x group. "pooled" and "within_group" are the historical scopes and must
stay byte-identical. "reference_group" is for a library where the group is a treatment
acting on the perturbation itself: a perturbation moved by the treatment has to be
tested against the untreated control state, so the null is pinned to the vehicle group
rather than to controls that saw the same treatment. Pinning the wrong pool is
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

from lib.aggregate.bootstrap import (  # noqa: E402
    bootstrap_control_mask,
    select_control_pool,
    within_perturbation_construct_mask,
)
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
        "EGFP_1=Vehicle",
        "H2B-EGFP_1=Vehicle",
        "EGFP_1=TreatA",
        "H2B-EGFP_1=TreatA",
        "EGFP_1=TreatB",
    ]
)


def _keys(control_pool):
    return list(control_pool[PERT_COL])


# --- Historical scopes ------------------------------------------------------------


def test_pooled_keeps_every_control():
    pool = select_control_pool(CONTROLS, "TARGET_5=TreatA", "pooled")

    pd.testing.assert_frame_equal(pool, CONTROLS)


def test_within_group_keeps_the_constructs_own_group():
    pool = select_control_pool(CONTROLS, "TARGET_5=TreatA", "within_group")

    assert _keys(pool) == ["EGFP_1=TreatA", "H2B-EGFP_1=TreatA"]


def test_within_group_is_inert_without_a_composite_key():
    """No group_cols means no group half to split on, so the pool stays pooled."""
    controls = _controls(["EGFP_1", "H2B-EGFP_1"])

    pool = select_control_pool(controls, "TARGET_5", "within_group")

    pd.testing.assert_frame_equal(pool, controls)


def test_within_group_raises_on_a_group_with_no_controls():
    with pytest.raises(ValueError, match="No control cells found for group 'TreatC'"):
        select_control_pool(CONTROLS, "TARGET_5=TreatC", "within_group")


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown bootstrap_control_scope: vehicle"):
        select_control_pool(CONTROLS, "TARGET_5=Vehicle", "vehicle")


# --- Reference group scope --------------------------------------------------------


def test_reference_group_pins_the_pool_across_groups():
    for construct_id in ["TARGET_5=TreatA", "TARGET_5=TreatB"]:
        pool = select_control_pool(
            CONTROLS,
            construct_id,
            "reference_group",
            reference_group="Vehicle",
            group_cols=GROUP_COLS,
        )

        assert _keys(pool) == ["EGFP_1=Vehicle", "H2B-EGFP_1=Vehicle"]


def test_reference_group_pool_differs_from_within_group_pool():
    """The whole point of the scope: same construct, a different set of control cells."""
    construct_id = "TARGET_5=TreatA"

    reference_pool = select_control_pool(
        CONTROLS,
        construct_id,
        "reference_group",
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )
    within_pool = select_control_pool(CONTROLS, construct_id, "within_group")

    assert set(_keys(reference_pool)).isdisjoint(_keys(within_pool))
    assert all(k.endswith(f"{GROUP_KEY_SEP}Vehicle") for k in _keys(reference_pool))


def test_reference_group_matches_within_group_inside_the_reference_group():
    """A construct already in the reference group must see the same null either way."""
    construct_id = "TARGET_5=Vehicle"

    reference_pool = select_control_pool(
        CONTROLS,
        construct_id,
        "reference_group",
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )

    pd.testing.assert_frame_equal(
        reference_pool, select_control_pool(CONTROLS, construct_id, "within_group")
    )


def test_reference_group_absent_from_the_control_pool_raises():
    with pytest.raises(ValueError, match="'DMSO' is absent from the control pool"):
        select_control_pool(
            CONTROLS,
            "TARGET_5=TreatA",
            "reference_group",
            reference_group="DMSO",
            group_cols=GROUP_COLS,
        )


def test_reference_group_error_lists_the_groups_that_are_present():
    with pytest.raises(ValueError) as excinfo:
        select_control_pool(
            CONTROLS,
            "TARGET_5=TreatA",
            "reference_group",
            reference_group="Vehicel",
            group_cols=GROUP_COLS,
        )

    assert "TreatA" in str(excinfo.value)
    assert "Vehicle" in str(excinfo.value)


def test_reference_group_raises_without_group_cols():
    """Ungrouped controls carry no group, so silently falling back to pooled would
    swap the null the operator asked for without a word."""
    controls = _controls(["EGFP_1", "H2B-EGFP_1"])

    with pytest.raises(ValueError, match="needs aggregate group_cols"):
        select_control_pool(
            controls, "TARGET_5", "reference_group", reference_group="Vehicle"
        )


def test_reference_group_raises_without_a_reference_group_value():
    with pytest.raises(ValueError, match="requires bootstrap_reference_group"):
        select_control_pool(
            CONTROLS, "TARGET_5=TreatA", "reference_group", group_cols=GROUP_COLS
        )


def test_reference_group_spans_several_group_cols():
    """Several group_cols fold into one key joined by GROUP_KEY_SEP, so the reference
    value is the whole joined key, not just the first column's value."""
    controls = _controls(
        ["EGFP_1=Vehicle=6h", "EGFP_1=Vehicle=24h", "EGFP_1=TreatA=6h"]
    )

    pool = select_control_pool(
        controls,
        "TARGET_5=TreatA=6h",
        "reference_group",
        reference_group=f"Vehicle{GROUP_KEY_SEP}6h",
        group_cols=["treatment", "timepoint"],
    )

    assert _keys(pool) == ["EGFP_1=Vehicle=6h"]


# --- Within perturbation scope ----------------------------------------------------

REFERENCE_POOL = _controls(
    [
        "TARGET_5=Vehicle",
        "TARGET_5=Vehicle",
        "OTHER_2=Vehicle",
        "EGFP_1=Vehicle",
    ]
)


def test_within_perturbation_draws_from_the_constructs_own_reference_arm():
    for construct_id in ["sgT1=TreatA", "sgT2=TreatB"]:
        pool = select_control_pool(
            REFERENCE_POOL,
            construct_id,
            "within_perturbation",
            reference_group="Vehicle",
            group_cols=GROUP_COLS,
            perturbation_id="TARGET_5=TreatA",
        )

        assert _keys(pool) == ["TARGET_5=Vehicle", "TARGET_5=Vehicle"]


def test_within_perturbation_raises_on_a_perturbation_with_no_reference_arm():
    with pytest.raises(
        ValueError, match="No control cells found for perturbation 'ORPHAN_3'"
    ):
        select_control_pool(
            REFERENCE_POOL,
            "sgO1=TreatA",
            "within_perturbation",
            reference_group="Vehicle",
            group_cols=GROUP_COLS,
            perturbation_id="ORPHAN_3=TreatA",
        )


def test_within_perturbation_requires_a_perturbation_id():
    """Construct ids carry a guide or barcode, so matching on them finds nothing."""
    with pytest.raises(ValueError, match="needs the construct's perturbation_id"):
        select_control_pool(
            REFERENCE_POOL,
            "TARGET_5=TreatA",
            "within_perturbation",
            reference_group="Vehicle",
            group_cols=GROUP_COLS,
        )


CELLS = pd.DataFrame(
    {
        PERT_COL: [
            "TARGET_5=Vehicle",
            "TARGET_5=TreatA",
            "EGFP_1=Vehicle",
            "EGFP_1=TreatA",
        ],
        "treatment": ["Vehicle", "TreatA", "Vehicle", "TreatA"],
    }
)


def test_control_mask_for_within_perturbation_is_the_reference_group():
    mask = bootstrap_control_mask(
        CELLS,
        PERT_COL,
        "EGFP",
        "within_perturbation",
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )

    assert list(mask) == [True, False, True, False]


def test_control_mask_for_other_scopes_is_the_control_key():
    for scope in ["pooled", "within_group", "reference_group"]:
        mask = bootstrap_control_mask(
            CELLS,
            PERT_COL,
            "EGFP",
            scope,
            reference_group="Vehicle",
            group_cols=GROUP_COLS,
        )

        assert list(mask) == [False, False, True, True]


def test_control_mask_raises_on_an_absent_reference_group():
    with pytest.raises(ValueError, match="'DMSO' matches no cells"):
        bootstrap_control_mask(
            CELLS,
            PERT_COL,
            "EGFP",
            "within_perturbation",
            reference_group="DMSO",
            group_cols=GROUP_COLS,
        )


def test_control_mask_for_within_perturbation_needs_group_cols():
    with pytest.raises(ValueError, match="needs aggregate group_cols"):
        bootstrap_control_mask(
            CELLS, PERT_COL, "EGFP", "within_perturbation", reference_group="Vehicle"
        )


def test_construct_mask_tests_only_referenced_non_reference_arms():
    """Reference arms are the null and an unreferenced perturbation has none."""
    construct_table = pd.DataFrame(
        {
            PERT_COL: [
                "TARGET_5=TreatA",
                "TARGET_5=Vehicle",
                "ORPHAN_3=TreatA",
                "EGFP_1=TreatA",
                "EGFP_1=Vehicle",
            ]
        }
    )

    mask = within_perturbation_construct_mask(
        construct_table,
        PERT_COL,
        CELLS[PERT_COL][CELLS["treatment"] == "Vehicle"],
        "Vehicle",
    )

    assert list(mask) == [True, False, False, True, False]
