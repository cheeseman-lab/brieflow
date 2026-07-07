"""Sub-tile hash merge over image-stitched global cell frames."""

from __future__ import annotations

import numpy as np
import pandas as pd

from workflow.lib.merge.hash import find_triangles, evaluate_match
from workflow.lib.merge.fast_merge import merge_triangle_hash
from workflow.lib.merge.deduplicate_merge import deduplicate_cells


def assign_subtiles(
    cells: pd.DataFrame,
    subtile_size: tuple[int, int],
    gy_col: str = "gy",
    gx_col: str = "gx",
) -> pd.DataFrame:
    """Bucket global-frame cells into a regular sub-tile grid.

    Args:
        cells: DataFrame with global centroid columns.
        subtile_size: (height, width) of each sub-tile in px.
        gy_col: global-y column.
        gx_col: global-x column.

    Returns:
        Copy of ``cells`` with an integer ``subtile`` column.
    """
    sh, sw = subtile_size
    gy = cells[gy_col].to_numpy()
    gx = cells[gx_col].to_numpy()
    row = np.floor((gy - gy.min()) / sh).astype(int)
    col = np.floor((gx - gx.min()) / sw).astype(int)
    ncol = col.max() + 1
    out = cells.copy()
    out["subtile"] = row * ncol + col
    return out


def subtile_bounds(
    cells: pd.DataFrame,
    subtile_size: tuple[int, int],
    gy_col: str = "gy",
    gx_col: str = "gx",
) -> dict[int, tuple[int, int, int, int]]:
    """Return {subtile: (y0, x0, y1, x1)} bounds for QC."""
    sh, sw = subtile_size
    gy0, gx0 = cells[gy_col].min(), cells[gx_col].min()
    bounds: dict[int, tuple[int, int, int, int]] = {}
    for st, grp in cells.groupby("subtile"):
        r = int((grp[gy_col].min() - gy0) // sh)
        c = int((grp[gx_col].min() - gx0) // sw)
        bounds[int(st)] = (
            gy0 + r * sh,
            gx0 + c * sw,
            gy0 + (r + 1) * sh,
            gx0 + (c + 1) * sw,
        )
    return bounds


def _align_subtile(
    ph_sub: pd.DataFrame,
    sbs_sub: pd.DataFrame,
    evaluate_kwargs: dict | None,
) -> dict | None:
    """Hash-align phenotype→SBS on subtile-local cells; return alignment dict or None."""
    t_ph = find_triangles(ph_sub[["gy", "gx"]].rename(columns={"gy": "i", "gx": "j"}))
    t_sbs = find_triangles(sbs_sub[["gy", "gx"]].rename(columns={"gy": "i", "gx": "j"}))
    rot, trans, score = evaluate_match(t_ph, t_sbs, **(evaluate_kwargs or {}))
    if rot is None:
        return None
    return {"rotation": rot, "translation": trans, "score": score}


def merge_subtiles(
    ph_cells: pd.DataFrame,
    sbs_cells: pd.DataFrame,
    subtile_size: tuple[int, int],
    threshold: float = 4,
    local_refinement: str | None = None,
    warp_kwargs: dict | None = None,
    evaluate_kwargs: dict | None = None,
) -> pd.DataFrame:
    """Hash-merge phenotype and SBS cells sub-tile by sub-tile (piecewise affine).

    PRECONDITION: ``ph_cells`` and ``sbs_cells`` must already be in a COMMON coarse
    coordinate frame — the caller is responsible for global scale+rotation+translation
    alignment (e.g. pixel-size scaling + a global hash alignment). This function
    performs only LOCAL piecewise refinement + matching within shared sub-tiles.

    Subtiles are assigned on the shared common frame using the same ``subtile_size``
    for both modalities. Only subtiles present in BOTH ph and sbs are processed, and
    each subtile matches only its own local cells — O(n_subtile²) per subtile rather
    than O(n_subtile × n_all). Concatenated results are deduplicated to strict 1:1 via
    ``deduplicate_cells`` (approach="fast", distance ascending priority at both steps).

    Args:
        ph_cells: phenotype global cell table (needs gy, gx, i, j, tile, well, plate, cell).
        sbs_cells: SBS global cell table (same columns).
        subtile_size: (h, w) of each sub-tile in px (applied to the shared coordinate frame).
        threshold: nearest-neighbour match threshold in px.
        local_refinement: None | "polynomial" | "thin_plate_spline".
        warp_kwargs: extra kwargs for the warp model.
        evaluate_kwargs: extra kwargs for evaluate_match (e.g. ransac_kwargs).

    Returns:
        Deduplicated 1:1 matched-cell DataFrame with a ``subtile`` column.
    """
    ph = assign_subtiles(ph_cells, subtile_size)
    sbs = assign_subtiles(sbs_cells, subtile_size)

    shared_subtiles = sorted(set(ph["subtile"].unique()) & set(sbs["subtile"].unique()))

    out = []
    for st in shared_subtiles:
        ph_sub = ph[ph["subtile"] == st].reset_index(drop=True)
        sbs_sub = sbs[sbs["subtile"] == st].reset_index(drop=True)
        if len(ph_sub) < 30 or len(sbs_sub) < 30:
            continue
        alignment = _align_subtile(ph_sub, sbs_sub, evaluate_kwargs)
        if alignment is None:
            continue
        ph_hash = ph_sub.assign(i=ph_sub["gy"], j=ph_sub["gx"])
        sbs_hash = sbs_sub.assign(i=sbs_sub["gy"], j=sbs_sub["gx"])
        m = merge_triangle_hash(
            ph_hash, sbs_hash, alignment, threshold=threshold,
            local_refinement=local_refinement, warp_kwargs=warp_kwargs,
        )
        if len(m):
            m = m.copy()
            m["subtile"] = st
            out.append(m)

    if not out:
        return ph_cells.head(0).assign(subtile=pd.Series(dtype=int))

    combined = pd.concat(out, ignore_index=True)
    if "mapped_single_gene" not in combined.columns:
        combined["mapped_single_gene"] = False

    combined = deduplicate_cells(
        combined,
        approach="fast",
        sbs_dedup_prior={"distance": True},
        pheno_dedup_prior={"distance": True},
    )

    return combined
