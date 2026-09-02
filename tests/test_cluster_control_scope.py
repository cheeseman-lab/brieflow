"""Regression tests for which controls the cluster potential is measured against.

`control_scope` picks the control rows `calculate_potential_to_nontargeting` averages
each point's diffusion-potential distance over. "pooled" is the historical scope and
must stay byte-identical. "reference_group" is for a library where the group is a
treatment acting on the perturbation itself: an over-expressed receptor moved by its
ligand has to be scored against the unliganded control state, so the null is pinned to
the vehicle group rather than to a control cloud averaged across every treatment.
Pinning the wrong pool is silent — it returns distances, just to the treated baseline —
so the scopes must be demonstrably different pools, and a mis-set reference group must
raise.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial.distance import pdist, squareform

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`),
# so phate_leiden_clustering.py's own `from lib.aggregate...` imports resolve too.
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.cell_data_utils import GROUP_KEY_SEP  # noqa: E402
from lib.cluster.phate_leiden_clustering import (  # noqa: E402
    calculate_potential_to_nontargeting,
    select_control_indices,
)

PERT_COL = "gene_symbol_0"
CONTROL_KEY = ["EGFP_1", "H2B-EGFP_1"]
GROUP_COLS = ["treatment"]

KEYS = [
    "NR3C1_5=Corticosterone",
    "NR3C1_5=Ethanol",
    "NR3C1_5=Progesterone",
    "EGFP_1=Ethanol",
    "H2B-EGFP_1=Ethanol",
    "EGFP_1=Corticosterone",
    "H2B-EGFP_1=Corticosterone",
    "EGFP_1=Progesterone",
    "H2B-EGFP_1=Progesterone",
]
PERTURBATIONS = pd.Series(KEYS, name=PERT_COL)


def _potential(keys):
    """Potential frame as phate_leiden_pipeline emits it: key column, then potentials."""
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            PERT_COL: keys,
            "potential_0": rng.normal(size=len(keys)),
            "potential_1": rng.normal(size=len(keys)),
        }
    )


def _keys(indices):
    return [KEYS[i] for i in indices]


# --- Pooled scope -----------------------------------------------------------------


def test_pooled_scores_every_point_against_every_control():
    scoped = select_control_indices(PERTURBATIONS, CONTROL_KEY)

    assert set(scoped) == set(PERTURBATIONS.index)
    for indices in scoped.values():
        assert _keys(indices) == KEYS[3:]


def test_pooled_is_the_default_and_reproduces_the_unscoped_distances():
    """The historical null: mean distance to every control row, in index order."""
    potential_df = _potential(KEYS)
    distances = squareform(pdist(potential_df[["potential_0", "potential_1"]].values))
    expected = [np.mean(distances[i, 3:]) for i in range(len(KEYS))]

    result = calculate_potential_to_nontargeting(potential_df, CONTROL_KEY)

    assert result["mean_potential_to_nontargeting"].tolist() == pytest.approx(expected)


# --- Within group scope -----------------------------------------------------------


def test_within_group_scores_each_point_against_its_own_group():
    scoped = select_control_indices(PERTURBATIONS, CONTROL_KEY, "within_group")

    assert _keys(scoped[0]) == ["EGFP_1=Corticosterone", "H2B-EGFP_1=Corticosterone"]
    assert _keys(scoped[1]) == ["EGFP_1=Ethanol", "H2B-EGFP_1=Ethanol"]
    assert _keys(scoped[2]) == ["EGFP_1=Progesterone", "H2B-EGFP_1=Progesterone"]


def test_within_group_is_inert_without_a_composite_key():
    """No group_cols means no group half to split on, so the null stays pooled."""
    perturbations = pd.Series(["NR3C1_5", "EGFP_1", "H2B-EGFP_1"])

    scoped = select_control_indices(perturbations, CONTROL_KEY, "within_group")

    assert all(indices == [1, 2] for indices in scoped.values())


def test_within_group_raises_on_a_group_with_no_controls():
    perturbations = pd.concat(
        [PERTURBATIONS, pd.Series(["NR3C1_5=Estradiol"])], ignore_index=True
    )

    with pytest.raises(
        ValueError, match="No control cells found for group 'Estradiol'"
    ):
        select_control_indices(perturbations, CONTROL_KEY, "within_group")


def test_unknown_scope_raises():
    with pytest.raises(ValueError, match="Unknown control_scope: vehicle"):
        select_control_indices(PERTURBATIONS, CONTROL_KEY, "vehicle")


# --- Reference group scope --------------------------------------------------------


def test_reference_group_pins_every_point_to_one_group():
    scoped = select_control_indices(
        PERTURBATIONS,
        CONTROL_KEY,
        "reference_group",
        reference_group="Ethanol",
        group_cols=GROUP_COLS,
    )

    for indices in scoped.values():
        assert _keys(indices) == ["EGFP_1=Ethanol", "H2B-EGFP_1=Ethanol"]


def test_reference_group_set_is_disjoint_from_the_rest_of_the_pooled_set():
    """The reference null is a strict subset of the pooled null, and the controls it
    drops — every non-vehicle control — share no row with the ones it keeps."""
    pooled = select_control_indices(PERTURBATIONS, CONTROL_KEY)[0]
    reference = select_control_indices(
        PERTURBATIONS,
        CONTROL_KEY,
        "reference_group",
        reference_group="Ethanol",
        group_cols=GROUP_COLS,
    )[0]

    assert set(reference) < set(pooled)
    assert set(reference).isdisjoint(set(pooled) - set(reference))
    assert all(k.endswith(f"{GROUP_KEY_SEP}Ethanol") for k in _keys(reference))


def test_reference_group_pool_differs_from_within_group_pool():
    """The whole point of the scope: same point, a different set of control rows."""
    treated = 0

    reference = select_control_indices(
        PERTURBATIONS,
        CONTROL_KEY,
        "reference_group",
        reference_group="Ethanol",
        group_cols=GROUP_COLS,
    )[treated]
    within = select_control_indices(PERTURBATIONS, CONTROL_KEY, "within_group")[treated]

    assert set(reference).isdisjoint(within)


def test_reference_group_moves_the_measured_distances():
    """A pinned null must reach the emitted column, not just the index selection."""
    potential_df = _potential(KEYS)

    pooled = calculate_potential_to_nontargeting(potential_df, CONTROL_KEY)
    reference = calculate_potential_to_nontargeting(
        potential_df,
        CONTROL_KEY,
        control_scope="reference_group",
        reference_group="Ethanol",
        group_cols=GROUP_COLS,
    )

    distances = squareform(pdist(potential_df[["potential_0", "potential_1"]].values))
    expected = [np.mean(distances[i, [3, 4]]) for i in range(len(KEYS))]
    assert reference["mean_potential_to_nontargeting"].tolist() == pytest.approx(
        expected
    )
    assert not np.allclose(
        reference["mean_potential_to_nontargeting"],
        pooled["mean_potential_to_nontargeting"],
    )


def test_reference_group_matches_within_group_inside_the_reference_group():
    """A point already in the reference group must see the same null either way."""
    vehicle = 1

    reference = select_control_indices(
        PERTURBATIONS,
        CONTROL_KEY,
        "reference_group",
        reference_group="Ethanol",
        group_cols=GROUP_COLS,
    )[vehicle]

    assert (
        reference
        == select_control_indices(PERTURBATIONS, CONTROL_KEY, "within_group")[vehicle]
    )


def test_reference_group_absent_from_the_control_pool_raises():
    with pytest.raises(ValueError, match="'DMSO' is absent from the control pool"):
        select_control_indices(
            PERTURBATIONS,
            CONTROL_KEY,
            "reference_group",
            reference_group="DMSO",
            group_cols=GROUP_COLS,
        )


def test_reference_group_error_lists_the_groups_that_are_present():
    with pytest.raises(ValueError) as excinfo:
        select_control_indices(
            PERTURBATIONS,
            CONTROL_KEY,
            "reference_group",
            reference_group="Etahnol",
            group_cols=GROUP_COLS,
        )

    assert "Corticosterone" in str(excinfo.value)
    assert "Ethanol" in str(excinfo.value)


def test_reference_group_raises_without_group_cols():
    """Ungrouped points carry no group, so silently falling back to pooled would swap
    the null the operator asked for without a word."""
    with pytest.raises(ValueError, match="needs aggregate group_cols"):
        select_control_indices(
            PERTURBATIONS, CONTROL_KEY, "reference_group", reference_group="Ethanol"
        )


def test_reference_group_raises_without_a_reference_group_value():
    with pytest.raises(ValueError, match="requires control_reference_group"):
        select_control_indices(
            PERTURBATIONS, CONTROL_KEY, "reference_group", group_cols=GROUP_COLS
        )


def test_reference_group_raises_on_ungrouped_perturbation_names():
    """group_cols set but an aggregated table written before grouping: the suffix the
    scope splits on is simply absent, so pinning cannot be honoured."""
    perturbations = pd.Series(["NR3C1_5", "EGFP_1", "H2B-EGFP_1"])

    with pytest.raises(ValueError, match="needs grouped perturbation names"):
        select_control_indices(
            perturbations,
            CONTROL_KEY,
            "reference_group",
            reference_group="Ethanol",
            group_cols=GROUP_COLS,
        )


def test_reference_group_spans_several_group_cols():
    """Several group_cols fold into one key joined by GROUP_KEY_SEP, so the reference
    value is the whole joined key, not just the first column's value."""
    perturbations = pd.Series(
        ["NR3C1_5=Corticosterone=6h", "EGFP_1=Ethanol=6h", "EGFP_1=Ethanol=24h"]
    )

    scoped = select_control_indices(
        perturbations,
        CONTROL_KEY,
        "reference_group",
        reference_group=f"Ethanol{GROUP_KEY_SEP}6h",
        group_cols=["treatment", "timepoint"],
    )

    assert scoped[0] == [1]
