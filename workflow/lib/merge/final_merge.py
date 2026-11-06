"""Streaming final merge: attach the CP phenotype feature table to the deduplicated merge.

phenotype_cp is ~3,600 columns x ~1M rows. A pandas left merge materializes it several
times over and OOM-kills a 64 GB box; polars scan -> join -> sink keeps the working set
bounded.
"""

from pathlib import Path
from typing import Optional, Sequence, Union

import polars as pl

# merge key: plate/well/tile locate the field, cell_0 the segmented cell.
KEYS = ["plate", "well", "tile", "cell_0"]


def final_merge(
    deduplicated_path: Union[str, Path],
    phenotype_cp_path: Union[str, Path],
    output_path: Union[str, Path],
    exclude_markers: Optional[Sequence[str]] = None,
) -> None:
    """Left-join the CP feature table onto the deduplicated merge.

    Output columns are the deduplicated columns followed by the CP feature columns
    ('label' renamed to 'cell_0'), as the pandas path produced; unmatched rows keep
    null CP columns.

    Args:
        deduplicated_path: Per-well deduplicated merge parquet.
        phenotype_cp_path: Per-well phenotype_cp parquet.
        output_path: Destination parquet.
        exclude_markers: Marker names whose `_<marker>_` feature columns are dropped.
    """
    exclude_markers = list(exclude_markers or [])

    dedup = pl.scan_parquet(deduplicated_path)
    dedup_schema = dedup.collect_schema()

    cp = pl.scan_parquet(phenotype_cp_path)
    cp_names = cp.collect_schema().names()
    drop_cols = [c for c in cp_names if any(f"_{m}_" in c for m in exclude_markers)]
    if drop_cols:
        cp = cp.drop(drop_cols)
    cp = cp.rename({"label": "cell_0"})

    # phenotype parquets can carry plate/tile as String where dedup has Int64;
    # without the cast the join silently matches nothing.
    cp = cp.with_columns([pl.col(k).cast(dedup_schema[k]) for k in KEYS])

    merged = dedup.join(cp, on=KEYS, how="left")

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    # The hash-join builds on the ~1M-row CP side (~30 GB at 3,591 float columns).
    merged.sink_parquet(str(output_path))
