"""Tests for the lazy construct-level aggregation in generate_feature_table.py.

The script replaced per-construct Python accumulators (which held every cell's
feature vector in memory across all batches) with a polars scan_parquet ->
group_by -> median over the aligned parquet it already wrote. Two contracts
have to survive that swap:

1. The medians match the np.nanmedian the accumulator computed, including on
   columns that are NaN for a construct. The pool schema is unified across
   wells, so per-well column filtering leaves NaN where a well dropped a
   column -- those must be skipped, not propagated.
2. Pseudo-gene grouping stays scoped to one group label. create_pseudogene_groups
   shuffles every matching construct into a single pool and chunks it, so
   calling it once over a multi-group construct table builds pseudo-genes that
   straddle groups -- which contaminates the bootstrap null with between-group
   variance -- and emits a pseudogene_id with no group suffix, breaking the
   {perturbation}{GROUP_KEY_SEP}{group} namespace every other row uses.

The script is a Snakemake entry point with no importable functions, so these
tests re-run its aggregation and grouping blocks verbatim against a fixture.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.aggregate.bootstrap import create_pseudogene_groups  # noqa: E402
from lib.aggregate.cell_data_utils import GROUP_KEY_SEP  # noqa: E402

FEATURES = ["feat_a", "feat_b"]


def write_aligned(path, rows):
    """Write an aligned-cell parquet the way the script's batch loop does.

    pa.Table.from_pandas is the load-bearing detail: it stores float NaN as a
    parquet null, which is what lets pl.median skip it.
    """
    df = pd.DataFrame(rows)
    for col in FEATURES:
        df[col] = df[col].astype(np.float32)
    table = pa.Table.from_pandas(df, preserve_index=False)
    writer = pq.ParquetWriter(path, table.schema)
    writer.write_table(table)
    writer.close()
    return df


def aggregate(path, group_cols=()):
    """The script's construct-level aggregation block."""
    agg_exprs = [pl.first("gene").alias("gene"), pl.len().alias("cell_count")]
    agg_exprs += [pl.first(c).alias(c) for c in group_cols]
    agg_exprs += [pl.median(c).alias(c) for c in FEATURES]
    return (
        pl.scan_parquet(path)
        .group_by("sgRNA")
        .agg(agg_exprs)
        .collect()
        .to_pandas()
        .sort_values("sgRNA")
        .reset_index(drop=True)
    )


def test_median_matches_nanmedian_accumulator(tmp_path):
    """Lazy polars medians equal the np.nanmedian the accumulator computed."""
    path = tmp_path / "aligned.parquet"
    df = write_aligned(
        path,
        {
            "sgRNA": ["s1", "s1", "s1", "s2", "s2"],
            "gene": ["G1", "G1", "G1", "G2", "G2"],
            # s1 has a NaN in feat_a: the median must be over the other two.
            "feat_a": [1.0, np.nan, 3.0, 10.0, 20.0],
            "feat_b": [4.0, 6.0, 8.0, 30.0, 40.0],
        },
    )

    out = aggregate(path)

    for sgrna in ("s1", "s2"):
        block = df[df["sgRNA"] == sgrna][FEATURES].to_numpy()
        expected = np.nanmedian(block, axis=0)
        actual = out.loc[out["sgRNA"] == sgrna, FEATURES].to_numpy()[0]
        np.testing.assert_allclose(actual, expected, rtol=1e-6)

    assert out["cell_count"].tolist() == [3, 2]


def test_all_nan_feature_column_does_not_become_a_number(tmp_path):
    """A construct with no values for a feature stays NaN, as nanmedian did."""
    path = tmp_path / "aligned.parquet"
    write_aligned(
        path,
        {
            "sgRNA": ["s1", "s1"],
            "gene": ["G1", "G1"],
            "feat_a": [1.0, 3.0],
            "feat_b": [np.nan, np.nan],
        },
    )

    out = aggregate(path)

    assert out.loc[0, "feat_a"] == 2.0
    assert pd.isna(out.loc[0, "feat_b"])


def _pseudogene_patterns():
    return {"nontargeting": {"pattern": "nontargeting", "constructs_per_pseudogene": 2}}


def _multi_group_construct_table():
    """Four nontargeting constructs, two per group, plus one real gene."""
    rows = []
    for group in ("t0", "t1"):
        for i in range(4):
            rows.append(
                {
                    "sgRNA": f"nt{i}{GROUP_KEY_SEP}{group}",
                    "gene": f"nontargeting{GROUP_KEY_SEP}{group}",
                    "group": group,
                    "cell_count": 10,
                    "feat_a": float(i),
                    "feat_b": float(i),
                }
            )
        rows.append(
            {
                "sgRNA": f"g1{GROUP_KEY_SEP}{group}",
                "gene": f"GENE1{GROUP_KEY_SEP}{group}",
                "group": group,
                "cell_count": 10,
                "feat_a": 1.0,
                "feat_b": 1.0,
            }
        )
    return pd.DataFrame(rows)


def group_scoped_pseudogenes(construct_table, construct_group_map):
    """The script's per-group pseudo-gene block."""
    pseudogene_groups = []
    construct_groups = construct_table["sgRNA"].map(construct_group_map)
    for group_label in sorted(construct_groups.dropna().unique()):
        group_pseudogenes, _ = create_pseudogene_groups(
            construct_table[construct_groups == group_label],
            _pseudogene_patterns(),
            "gene",
            seed=42,
        )
        for pseudogene_group in group_pseudogenes:
            pseudogene_group["pseudogene_id"] += f"{GROUP_KEY_SEP}{group_label}"
        pseudogene_groups.extend(group_pseudogenes)
    return pseudogene_groups


def test_group_label_map_survives_the_lazy_aggregation(tmp_path):
    """group_cols carried through the agg rebuild the construct -> group map."""
    path = tmp_path / "aligned.parquet"
    write_aligned(
        path,
        {
            "sgRNA": ["s1", "s1", "s2", "s2"],
            "gene": ["G1", "G1", "G2", "G2"],
            "group": ["t0", "t0", "t1", "t1"],
            "feat_a": [1.0, 3.0, 5.0, 7.0],
            "feat_b": [1.0, 3.0, 5.0, 7.0],
        },
    )

    agg = aggregate(path, group_cols=("group",))
    construct_group_map = dict(zip(agg["sgRNA"], agg["group"].astype(str)))

    assert construct_group_map == {"s1": "t0", "s2": "t1"}


def test_pseudogenes_never_straddle_groups():
    """Every pseudo-gene draws its constructs from exactly one group label."""
    table = _multi_group_construct_table()
    groups = group_scoped_pseudogenes(table, dict(zip(table["sgRNA"], table["group"])))

    assert groups, "expected pseudo-genes to be created"
    for pg in groups:
        labels = {c["group"] for c in pg["constructs"]}
        assert len(labels) == 1, f"{pg['pseudogene_id']} mixes groups {labels}"


def test_pseudogene_id_carries_the_group_suffix():
    """pseudogene_id stays in the {name}{SEP}{group} namespace the table uses."""
    table = _multi_group_construct_table()
    groups = group_scoped_pseudogenes(table, dict(zip(table["sgRNA"], table["group"])))

    ids = sorted(pg["pseudogene_id"] for pg in groups)
    assert ids == [
        f"nontargeting_pseudogene_01{GROUP_KEY_SEP}t0",
        f"nontargeting_pseudogene_01{GROUP_KEY_SEP}t1",
        f"nontargeting_pseudogene_02{GROUP_KEY_SEP}t0",
        f"nontargeting_pseudogene_02{GROUP_KEY_SEP}t1",
    ]


def test_ungrouped_call_would_straddle_groups():
    """Why the per-group loop exists: one flat call mixes groups.

    This pins the regression the group-scoped loop prevents -- it asserts the
    behavior of calling create_pseudogene_groups once over a multi-group table,
    which is what the script must not do.
    """
    table = _multi_group_construct_table()
    flat, _ = create_pseudogene_groups(table, _pseudogene_patterns(), "gene", seed=42)

    straddling = [pg for pg in flat if len({c["group"] for c in pg["constructs"]}) > 1]
    assert straddling, "fixture should produce a group-straddling pseudo-gene"
    assert all(GROUP_KEY_SEP not in pg["pseudogene_id"] for pg in flat)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
