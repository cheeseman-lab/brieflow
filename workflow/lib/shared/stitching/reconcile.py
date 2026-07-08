"""Intra-modality overlap cell reconciliation for stitched cell tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


def reconcile_overlap_cells(
    cells: pd.DataFrame,
    radius_px: float = 5.0,
    gy_col: str = "gy",
    gx_col: str = "gx",
    tile_col: str = "tile",
) -> pd.DataFrame:
    """De-duplicate cells from overlapping tiles by collapsing near-duplicate pairs.

    Cells from adjacent tiles that lie within ``radius_px`` of each other in
    the global frame (columns ``gy_col``, ``gx_col``) are treated as the same
    physical cell detected twice in the tile overlap. Union-find connects all
    such cross-tile near-neighbour pairs into groups; from each group the cell
    whose position is closest to the group centroid (mean of all member
    positions) is retained. Cells from the same tile are never merged regardless
    of proximity.

    Args:
        cells: Global-frame cell table with at least ``gy_col``, ``gx_col``,
            and ``tile_col`` columns.
        radius_px: Maximum centre-to-centre distance (same units as the
            coordinate columns) for two cells from different tiles to count as
            duplicates.
        gy_col: Column name for global y coordinate.
        gx_col: Column name for global x coordinate.
        tile_col: Column name for tile identifier.

    Returns:
        De-duplicated DataFrame in the same column order as ``cells``,
        containing one row per unique physical cell (index reset).
    """
    if len(cells) == 0:
        return cells.reset_index(drop=True)

    coords = cells[[gy_col, gx_col]].to_numpy(dtype=np.float64)
    tile_ids = cells[tile_col].to_numpy()
    n = len(cells)

    parent = np.arange(n, dtype=np.intp)

    def _find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return int(i)

    def _union(i: int, j: int) -> None:
        pi, pj = _find(i), _find(j)
        if pi != pj:
            parent[pi] = pj

    tree = cKDTree(coords)
    for i, j in tree.query_pairs(radius_px):
        if tile_ids[i] != tile_ids[j]:
            _union(i, j)

    roots = np.array([_find(k) for k in range(n)])

    keep_indices: list[int] = []
    for root in np.unique(roots):
        members = np.where(roots == root)[0]
        if len(members) == 1:
            keep_indices.append(int(members[0]))
        else:
            group_coords = coords[members]
            centroid = group_coords.mean(axis=0)
            dists = np.linalg.norm(group_coords - centroid, axis=1)
            keep_indices.append(int(members[np.argmin(dists)]))

    return cells.iloc[keep_indices].reset_index(drop=True)
