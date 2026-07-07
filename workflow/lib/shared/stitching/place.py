"""Globally-consistent tile placement from pairwise shifts."""

from __future__ import annotations

import igraph as ig
import numpy as np
import pandas as pd

from workflow.lib.shared.stitching.types import TileOffsets


def solve_global_offsets(n_tiles, edges, prior, min_confidence=0.2):
    """Solve per-tile global offsets from pairwise shift edges.

    Args:
        n_tiles: number of tiles.
        edges: list of (i, j, shift_yx, confidence); shift_yx is j minus i.
        prior: {tile: (y, x)} stage-coordinate fallback offsets.
        min_confidence: edges below this confidence are ignored.

    Returns:
        TileOffsets in a single well frame (root tile at its prior position).
    """
    good = [(i, j, s, c) for (i, j, s, c) in edges if c >= min_confidence]
    g = ig.Graph()
    g.add_vertices(n_tiles)
    for i, j, s, c in good:
        g.add_edge(i, j, weight=float(c), shift=np.asarray(s, dtype=float))

    offsets = {t: None for t in range(n_tiles)}
    for comp in g.connected_components(mode="weak"):
        if not comp:
            continue
        sub = g.subgraph(comp)
        # maximum-confidence spanning tree = MST on negative weights
        neg = [-w for w in sub.es["weight"]] if sub.ecount() > 0 else []
        mst = sub.spanning_tree(weights=neg if neg else None)
        root_local = 0
        root_global = comp[root_local]
        offsets[root_global] = np.asarray(prior[root_global], dtype=float)
        # propagate offsets along MST edges in BFS order from root
        bfs_vids, _, parent = mst.bfs(root_local)
        for v in bfs_vids:
            if v == root_local:
                continue
            p = parent[v]
            eid = mst.get_eid(p, v)
            shift = mst.es[eid]["shift"]
            # orient shift so it maps parent -> child (shift_yx = child - parent)
            src, tgt = mst.es[eid].tuple
            s = shift if (src, tgt) == (p, v) else -shift
            offsets[comp[v]] = offsets[comp[p]] + s

    for t in range(n_tiles):
        if offsets[t] is None:
            offsets[t] = np.asarray(prior[t], dtype=float)

    frame = pd.DataFrame(
        {
            "tile": list(range(n_tiles)),
            "y": [offsets[t][0] for t in range(n_tiles)],
            "x": [offsets[t][1] for t in range(n_tiles)],
        }
    )
    return TileOffsets.from_frame(frame)
