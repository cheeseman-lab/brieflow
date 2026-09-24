"""Regression tests for pooling per-well parquet files and benchmarking small libraries.

1. `pool_dataset` must read per-well files whose schemas disagree: a well that
   filters to no cells writes all-null columns, and a well with missing values
   promotes an integer column to double. Files that agree must read exactly as a
   plain pyarrow dataset does.
2. `calculate_group_enrichment` must count only screened genes in each benchmark
   group, so a group larger than the library cannot make the contingency table
   negative, and a group with no screened genes is not tested at all.
"""

import sys
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

# Import the way the pipeline does at runtime (workflow/ on path -> top-level `lib`).
_WORKFLOW = Path(__file__).resolve().parents[1] / "workflow"
if str(_WORKFLOW) not in sys.path:
    sys.path.insert(0, str(_WORKFLOW))

from lib.cluster.benchmark_clusters import calculate_group_enrichment  # noqa: E402
from lib.shared.parquet_io import pool_dataset  # noqa: E402


def _write(tmp_path, name, table):
    fp = tmp_path / name
    pq.write_table(table, fp)
    return fp


# --- pool_dataset ------------------------------------------------------------------


def test_pool_dataset_promotes_int_to_double(tmp_path):
    """A well with missing values stores a count as double while its neighbour keeps
    int64; the pooled read must hold both as double."""
    a = _write(tmp_path, "a.parquet", pa.table({"x": pa.array([1, 2], pa.int64())}))
    b = _write(tmp_path, "b.parquet", pa.table({"x": pa.array([1.5, None])}))

    table = pool_dataset([a, b]).to_table()

    assert table.schema.field("x").type == pa.float64()
    assert table.column("x").to_pylist() == [1.0, 2.0, 1.5, None]


def test_pool_dataset_resolves_null_to_string(tmp_path):
    """A well that filtered to no cells writes an all-null column."""
    a = _write(tmp_path, "a.parquet", pa.table({"gene": pa.array([None], pa.null())}))
    b = _write(tmp_path, "b.parquet", pa.table({"gene": pa.array(["TP53"])}))

    table = pool_dataset([a, b]).to_table()

    assert table.schema.field("gene").type == pa.string()
    assert table.column("gene").to_pylist() == [None, "TP53"]


def test_pool_dataset_unions_columns(tmp_path):
    """A column missing from one file reads as null for that file's rows, and the
    column order follows the first file."""
    a = _write(tmp_path, "a.parquet", pa.table({"y": [1.0], "x": [2.0]}))
    b = _write(tmp_path, "b.parquet", pa.table({"x": [3.0]}))

    table = pool_dataset([a, b]).to_table()

    assert table.schema.names == ["y", "x"]
    assert table.column("y").to_pylist() == [1.0, None]
    assert table.column("x").to_pylist() == [2.0, 3.0]


def test_pool_dataset_matches_plain_dataset_when_schemas_agree(tmp_path):
    """Files that already agree read exactly as before the change."""
    schema = pa.schema([("x", pa.int64()), ("gene", pa.string())])
    paths = [
        _write(
            tmp_path, f"{i}.parquet", pa.table({"x": [i], "gene": [f"g{i}"]}, schema)
        )
        for i in range(3)
    ]

    pooled = pool_dataset(paths).to_table()
    plain = ds.dataset([str(p) for p in paths], format="parquet").to_table()

    assert pooled.schema.equals(plain.schema, check_metadata=True)
    assert pooled.equals(plain)


# --- calculate_group_enrichment ------------------------------------------------------


def _clustering(genes_by_cluster):
    rows = [
        {"gene_symbol_0": gene, "cluster": cluster}
        for cluster, genes in genes_by_cluster.items()
        for gene in genes
    ]
    return pd.DataFrame(rows)


def _benchmark(genes_by_group):
    rows = [
        {"group": group, "gene_name": gene}
        for group, genes in genes_by_group.items()
        for gene in genes
    ]
    return pd.DataFrame(rows)


def test_group_enrichment_group_larger_than_library():
    """A benchmark group with more genes than the library screens used to make d
    negative and crash fisher_exact."""
    clustering = _clustering({0: ["A", "B", "C"], 1: ["D", "E", "F"]})
    unscreened = [f"U{i}" for i in range(20)]
    benchmark = _benchmark({"big": ["A", "B", "C", *unscreened]})

    table = calculate_group_enrichment(
        benchmark, clustering, return_full_table=True
    ).set_index("cluster")

    assert table.loc[0, "num_enriched_groups"] == 0
    assert table.loc[1, "num_enriched_groups"] == 0


def test_group_enrichment_skips_groups_without_screened_genes():
    """A group with no screened genes cannot be enriched and must not count towards
    the multiple-testing correction, which would otherwise push real hits over FDR."""
    cluster_genes = [f"C{i}" for i in range(10)]
    other_genes = [f"O{i}" for i in range(40)]
    clustering = _clustering({0: cluster_genes, 1: other_genes})
    groups = {"hit": cluster_genes[:3]}
    empty = {f"empty{i}": [f"U{i}"] for i in range(200)}

    alone = calculate_group_enrichment(
        _benchmark(groups), clustering, return_full_table=True
    ).set_index("cluster")
    with_empty = calculate_group_enrichment(
        _benchmark({**groups, **empty}), clustering, return_full_table=True
    ).set_index("cluster")

    assert alone.loc[0, "enriched_groups"] == "hit"
    assert with_empty.loc[0, "enriched_groups"] == "hit"


def test_group_enrichment_no_screened_groups():
    """A benchmark that shares no genes with the library reports zero, not a crash."""
    clustering = _clustering({0: ["A", "B"], 1: ["C", "D"]})
    benchmark = _benchmark({"g": ["X", "Y"]})

    assert calculate_group_enrichment(benchmark, clustering) == 0
