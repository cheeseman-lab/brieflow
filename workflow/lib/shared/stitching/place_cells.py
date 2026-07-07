"""Place per-tile cell centroids into the global well frame."""

from __future__ import annotations

import pandas as pd

from workflow.lib.shared.stitching.types import TileOffsets


def place_cells(
    cells: pd.DataFrame,
    offsets: TileOffsets,
    y_col: str = "i",
    x_col: str = "j",
    tile_col: str = "tile",
) -> pd.DataFrame:
    """Add global gy, gx columns to a per-tile cell table using tile offsets.

    Args:
        cells: DataFrame with a tile column and local centroid columns.
        offsets: TileOffsets for this modality's well.
        y_col: local centroid row column.
        x_col: local centroid col column.
        tile_col: tile-id column in ``cells``.

    Returns:
        Copy of ``cells`` with float ``gy``, ``gx`` global-frame columns.
    """
    off = offsets.to_frame().set_index("tile")
    out = cells.copy()
    out["gy"] = out[y_col] + out[tile_col].map(off["y"])
    out["gx"] = out[x_col] + out[tile_col].map(off["x"])
    return out
