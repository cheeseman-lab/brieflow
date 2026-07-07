"""Whole-well stitch orchestration: pairwise registration + global placement."""

from __future__ import annotations

import numpy as np
import shapely

from workflow.lib.shared.stitching.place import solve_global_offsets
from workflow.lib.shared.stitching.register import register_pair
from workflow.lib.shared.stitching.types import TileOffsets


def find_neighbor_pairs(
    prior: dict[int, tuple[float, float]],
    tile_shape: tuple[int, int],
    overlap_fraction: float,
) -> list[tuple[int, int, tuple[float, float]]]:
    """Return (i, j, expected_shift) for tiles whose prior boxes overlap.

    Args:
        prior: {tile: (y, x)} stage-coordinate offsets.
        tile_shape: (height, width) of each tile in pixels.
        overlap_fraction: expected fractional tile overlap (unused in detection,
            kept for API consistency with stitch_well).

    Returns:
        List of (i, j, expected_shift) tuples where i < j and the tiles'
        prior bounding boxes intersect. expected_shift is (dy, dx) = prior[j] - prior[i].
    """
    h, w = tile_shape
    ids = sorted(prior)
    boxes = {
        t: shapely.box(prior[t][1], prior[t][0], prior[t][1] + w, prior[t][0] + h)
        for t in ids
    }
    pairs = []
    for a_idx in range(len(ids)):
        for b_idx in range(a_idx + 1, len(ids)):
            i, j = ids[a_idx], ids[b_idx]
            inter = boxes[i].intersection(boxes[j])
            if inter.is_empty or inter.area <= 0:
                continue
            # Exclude corner-only (diagonal) overlaps: require the intersection to span
            # most of the tile in at least one dimension (strip, not just a corner).
            bounds = inter.bounds  # (minx, miny, maxx, maxy)
            inter_w = bounds[2] - bounds[0]
            inter_h = bounds[3] - bounds[1]
            if not (inter_h > (1 - overlap_fraction) * h or inter_w > (1 - overlap_fraction) * w):
                continue
            expected = (prior[j][0] - prior[i][0], prior[j][1] - prior[i][1])
            pairs.append((i, j, expected))
    return pairs


def stitch_well(
    planes: dict[int, np.ndarray],
    prior: dict[int, tuple[float, float]],
    overlap_fraction: float = 0.1,
    min_confidence: float = 0.2,
) -> TileOffsets:
    """Stitch one modality's well from 2D tile planes and a stage-coord prior.

    Args:
        planes: {tile: 2D registration plane}.
        prior: {tile: (y, x)} stage-coordinate offsets (initial guess).
        overlap_fraction: expected fractional tile overlap.
        min_confidence: minimum ZNCC to trust a pairwise edge.

    Returns:
        TileOffsets in a single global well frame.
    """
    any_plane = next(iter(planes.values()))
    tile_shape = any_plane.shape
    pairs = find_neighbor_pairs(prior, tile_shape, overlap_fraction)
    edges = []
    for i, j, expected in pairs:
        shift_yx, conf = register_pair(
            planes[i],
            planes[j],
            expected_shift=expected,
            overlap_fraction=overlap_fraction,
            max_shift=0.5 * min(tile_shape),
        )
        edges.append((i, j, shift_yx, conf))
    return solve_global_offsets(len(planes), edges, prior, min_confidence)
