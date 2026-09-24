"""Streaming final merge — attach the full CP phenotype feature table to the
deduplicated merge via a memory-bounded polars left join.

Why not a plain pandas .merge: phenotype_cp is ~3,600 cols x ~1M rows; a pandas
full-load merge (or a wide batched merge) materializes it several times over and
OOM-kills even a 64 GB box. polars scan_parquet -> left join -> sink_parquet
streams it out-of-core and multithreaded, working set bounded.

Lives in lib/ rather than inline in the Snakemake step so the join can be
imported and tested without snakemake; workflow/scripts/merge/final_merge.py is
a thin wrapper over it.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import polars as pl

# merge key: plate/well/tile locate the field, cell_0 the segmented cell.
KEYS = ["plate", "well", "tile", "cell_0"]

# stitch approach reports global (plate-level) coordinates; rename on the dedup
# (left) side before the join, matching the prior pandas behaviour.
_STITCH_RENAME = {
    "i_0": "global_i_0",
    "j_0": "global_j_0",
    "i_1": "global_i_1",
    "j_1": "global_j_1",
}


def final_merge(
    deduplicated_path: Union[str, Path],
    phenotype_cp_path: Union[str, Path],
    output_path: Union[str, Path],
    approach: str = "fast",
    exclude_markers: Optional[Sequence[str]] = None,
) -> None:
    """Left-join the full CP feature table onto the deduplicated merge, streaming.

    Output columns = all deduplicated cols, then CP feature cols (phenotype
    'label' -> 'cell_0', minus the join keys) — the same column set/order the
    prior batched-pandas path produced. Unmatched dedup rows keep null CP cols.
    """
    exclude_markers = list(exclude_markers or [])

    dedup = pl.scan_parquet(deduplicated_path)
    dedup_schema = dedup.collect_schema()
    if approach == "stitch":
        ren = {k: v for k, v in _STITCH_RENAME.items() if k in dedup_schema.names()}
        if ren:
            dedup = dedup.rename(ren)
            dedup_schema = dedup.collect_schema()

    cp = pl.scan_parquet(phenotype_cp_path)
    cp_names = cp.collect_schema().names()
    drop_cols = [c for c in cp_names if any(f"_{m}_" in c for m in exclude_markers)]
    if drop_cols:
        cp = cp.drop(drop_cols)
    cp = cp.rename({"label": "cell_0"})

    # phenotype parquets can store plate/tile as String while dedup has them as
    # Int64 (see phenotype-plate-tile-dtype-spec); reconcile key dtypes or the
    # join silently drops every row.
    cp = cp.with_columns([pl.col(k).cast(dedup_schema[k]) for k in KEYS])

    merged = dedup.join(cp, on=KEYS, how="left")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # streaming hash-join builds on the ~1M-row CP side (~30 GB peak
    # for 3,591 float cols); fine <64 GB. If CP row count grows, join per-tile.
    merged.sink_parquet(str(output_path))
