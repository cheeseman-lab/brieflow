"""Regression tests for which controls the cluster potential is measured against.

`control_scope` picks the control rows `calculate_potential_to_nontargeting` averages
each point's diffusion-potential distance over. "pooled" is the historical scope and
must stay byte-identical. "reference_group" is for a library where the group is a
treatment acting on the perturbation itself: a perturbation moved by the treatment has
to be scored against the untreated control state, so the null is pinned to
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
    filter_by_perturbation_auc,
    select_control_indices,
)
from lib.shared.rule_utils import get_cluster_control_key  # noqa: E402

PERT_COL = "gene_symbol_0"
CONTROL_KEY = ["EGFP_1", "H2B-EGFP_1"]
GROUP_COLS = ["treatment"]

KEYS = [
    "TARGET_5=TreatA",
    "TARGET_5=Vehicle",
    "TARGET_5=TreatB",
    "EGFP_1=Vehicle",
    "H2B-EGFP_1=Vehicle",
    "EGFP_1=TreatA",
    "H2B-EGFP_1=TreatA",
    "EGFP_1=TreatB",
    "H2B-EGFP_1=TreatB",
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

    assert _keys(scoped[0]) == ["EGFP_1=TreatA", "H2B-EGFP_1=TreatA"]
    assert _keys(scoped[1]) == ["EGFP_1=Vehicle", "H2B-EGFP_1=Vehicle"]
    assert _keys(scoped[2]) == ["EGFP_1=TreatB", "H2B-EGFP_1=TreatB"]


def test_within_group_is_inert_without_a_composite_key():
    """No group_cols means no group half to split on, so the null stays pooled."""
    perturbations = pd.Series(["TARGET_5", "EGFP_1", "H2B-EGFP_1"])

    scoped = select_control_indices(perturbations, CONTROL_KEY, "within_group")

    assert all(indices == [1, 2] for indices in scoped.values())


def test_within_group_raises_on_a_group_with_no_controls():
    perturbations = pd.concat(
        [PERTURBATIONS, pd.Series(["TARGET_5=TreatC"])], ignore_index=True
    )

    with pytest.raises(ValueError, match="No control cells found for group 'TreatC'"):
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
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )

    for indices in scoped.values():
        assert _keys(indices) == ["EGFP_1=Vehicle", "H2B-EGFP_1=Vehicle"]


def test_reference_group_set_is_disjoint_from_the_rest_of_the_pooled_set():
    """The reference null is a strict subset of the pooled null, and the controls it
    drops — every non-vehicle control — share no row with the ones it keeps."""
    pooled = select_control_indices(PERTURBATIONS, CONTROL_KEY)[0]
    reference = select_control_indices(
        PERTURBATIONS,
        CONTROL_KEY,
        "reference_group",
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )[0]

    assert set(reference) < set(pooled)
    assert set(reference).isdisjoint(set(pooled) - set(reference))
    assert all(k.endswith(f"{GROUP_KEY_SEP}Vehicle") for k in _keys(reference))


def test_reference_group_pool_differs_from_within_group_pool():
    """The whole point of the scope: same point, a different set of control rows."""
    treated = 0

    reference = select_control_indices(
        PERTURBATIONS,
        CONTROL_KEY,
        "reference_group",
        reference_group="Vehicle",
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
        reference_group="Vehicle",
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
        reference_group="Vehicle",
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
            reference_group="Vehicel",
            group_cols=GROUP_COLS,
        )

    assert "TreatA" in str(excinfo.value)
    assert "Vehicle" in str(excinfo.value)


def test_reference_group_raises_without_group_cols():
    """Ungrouped points carry no group, so silently falling back to pooled would swap
    the null the operator asked for without a word."""
    with pytest.raises(ValueError, match="needs aggregate group_cols"):
        select_control_indices(
            PERTURBATIONS, CONTROL_KEY, "reference_group", reference_group="Vehicle"
        )


def test_reference_group_raises_without_a_reference_group_value():
    with pytest.raises(ValueError, match="requires control_reference_group"):
        select_control_indices(
            PERTURBATIONS, CONTROL_KEY, "reference_group", group_cols=GROUP_COLS
        )


def test_reference_group_raises_on_ungrouped_perturbation_names():
    """group_cols set but an aggregated table written before grouping: the suffix the
    scope splits on is simply absent, so pinning cannot be honoured."""
    perturbations = pd.Series(["TARGET_5", "EGFP_1", "H2B-EGFP_1"])

    with pytest.raises(ValueError, match="needs grouped perturbation names"):
        select_control_indices(
            perturbations,
            CONTROL_KEY,
            "reference_group",
            reference_group="Vehicle",
            group_cols=GROUP_COLS,
        )


def test_reference_group_spans_several_group_cols():
    """Several group_cols fold into one key joined by GROUP_KEY_SEP, so the reference
    value is the whole joined key, not just the first column's value."""
    perturbations = pd.Series(
        ["TARGET_5=TreatA=6h", "EGFP_1=Vehicle=6h", "EGFP_1=Vehicle=24h"]
    )

    scoped = select_control_indices(
        perturbations,
        CONTROL_KEY,
        "reference_group",
        reference_group=f"Vehicle{GROUP_KEY_SEP}6h",
        group_cols=["treatment", "timepoint"],
    )

    assert scoped[0] == [1]


# within_perturbation: `X=treated` is scored against `X=reference_group`


def test_within_perturbation_scores_each_point_against_its_own_vehicle_arm():
    """Each point's pool is its own perturbation in the reference group, only."""
    scoped = select_control_indices(
        PERTURBATIONS,
        CONTROL_KEY,
        "within_perturbation",
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )
    for idx, key in PERTURBATIONS.items():
        perturbation = key.split(GROUP_KEY_SEP)[0]
        assert [PERTURBATIONS[i] for i in scoped[idx]] == [
            f"{perturbation}{GROUP_KEY_SEP}Vehicle"
        ]


def test_within_perturbation_ignores_control_key():
    """The reference group defines the scope, so the control key cannot change it."""
    pools = [
        select_control_indices(
            PERTURBATIONS,
            key,
            "within_perturbation",
            reference_group="Vehicle",
            group_cols=GROUP_COLS,
        )
        for key in (CONTROL_KEY, ["TARGET_5"], "nontargeting")
    ]
    assert pools[0] == pools[1] == pools[2]


def test_within_perturbation_leaves_an_unreferenced_perturbation_unscored():
    """A perturbation with no reference arm gets an empty pool, not an exception."""
    keys = pd.Series(list(KEYS) + ["ORPHAN_3=TreatD"], name=PERT_COL)
    scoped = select_control_indices(
        keys,
        CONTROL_KEY,
        "within_perturbation",
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )
    assert scoped[keys.index[-1]] == []


def test_within_perturbation_requires_group_cols_and_a_reference_group():
    """Without a group to pin to, the scope cannot be resolved and must raise."""
    for kwargs in (
        {"reference_group": "Vehicle", "group_cols": []},
        {"reference_group": None, "group_cols": GROUP_COLS},
    ):
        with pytest.raises(ValueError):
            select_control_indices(
                PERTURBATIONS, CONTROL_KEY, "within_perturbation", **kwargs
            )


def test_within_perturbation_puts_a_reference_arm_at_zero_distance_from_itself():
    """An arm in the reference group is its own control, so its potential is zero."""
    result = calculate_potential_to_nontargeting(
        _potential(KEYS),
        CONTROL_KEY,
        control_scope="within_perturbation",
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )
    vehicle = result[result[PERT_COL].str.endswith(f"{GROUP_KEY_SEP}Vehicle")]
    assert np.allclose(vehicle["mean_potential_to_nontargeting"], 0.0)


# the AUC filter must not drop the reference arms a within_perturbation null needs

AUC_POINTS = pd.DataFrame(
    {
        PERT_COL: [
            "TARGET_5=TreatA",
            "TARGET_5=Vehicle",
            "EGFP_1=Vehicle",
            "OTHER_2=TreatA",
        ],
        "perturbation_auc": [0.9, 0.5, 0.5, 0.5],
    }
)


def test_auc_filter_keeps_controls_and_high_auc_points():
    kept = filter_by_perturbation_auc(AUC_POINTS, PERT_COL, CONTROL_KEY, 0.6)

    assert list(kept[PERT_COL]) == ["TARGET_5=TreatA", "EGFP_1=Vehicle"]


def test_auc_filter_keeps_every_reference_arm_under_within_perturbation():
    kept = filter_by_perturbation_auc(
        AUC_POINTS,
        PERT_COL,
        CONTROL_KEY,
        0.6,
        control_scope="within_perturbation",
        reference_group="Vehicle",
    )

    assert list(kept[PERT_COL]) == [
        "TARGET_5=TreatA",
        "TARGET_5=Vehicle",
        "EGFP_1=Vehicle",
    ]
    scoped = select_control_indices(
        kept[PERT_COL],
        CONTROL_KEY,
        "within_perturbation",
        reference_group="Vehicle",
        group_cols=GROUP_COLS,
    )
    assert scoped[0] == [1]


def test_auc_filter_is_inert_without_a_threshold():
    pd.testing.assert_frame_equal(
        filter_by_perturbation_auc(AUC_POINTS, PERT_COL, CONTROL_KEY, None), AUC_POINTS
    )


@pytest.mark.parametrize(
    "config,expected",
    [
        ({"aggregate": {"control_key": "nontargeting"}}, "nontargeting"),
        ({"aggregate": {"control_key": "nontargeting"}, "cluster": {}}, "nontargeting"),
        (
            {
                "aggregate": {"control_key": "nontargeting"},
                "cluster": {"control_key": None},
            },
            "nontargeting",
        ),
        (
            {
                "aggregate": {"control_key": "Vehicle"},
                "cluster": {"control_key": ["EGFP_1"]},
            },
            ["EGFP_1"],
        ),
    ],
)
def test_cluster_control_key_falls_back_to_the_aggregate_key(config, expected):
    assert get_cluster_control_key(config) == expected
