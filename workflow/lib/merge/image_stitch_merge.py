"""Sub-tile hash merge over image-stitched global cell frames."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from workflow.lib.merge.hash import find_triangles, evaluate_match
from workflow.lib.merge.fast_merge import merge_triangle_hash
from workflow.lib.merge.deduplicate_merge import deduplicate_cells
from workflow.lib.shared.stitching.types import TileOffsets


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


def merge_reference_tiles(
    sbs_cells: pd.DataFrame,
    ph_cells: pd.DataFrame,
    coarse: dict,
    sbs_offsets: TileOffsets,
    tile_shape: tuple[int, int],
    threshold: float = 4,
    local_refinement: str | None = None,
    warp_kwargs: dict | None = None,
    evaluate_kwargs: dict | None = None,
    margin_px: float = 100.0,
    min_cells: int = 30,
    align_ratio: float = float("inf"),
    global_model: bool = True,
) -> pd.DataFrame:
    """Hash-merge phenotype and SBS cells anchored on SBS tile footprints (tier 3 recut).

    Applies a coarse transform (from coarse_align_dapi) to bring phenotype cells into
    the SBS reference frame, then for each SBS tile performs a local triangle-hash merge
    using only the cells whose coarse-projected positions fall within that tile's
    footprint (plus margin). All per-tile matches are deduplicated to strict 1:1.

    Each local hash is bounded by the SBS tile (~5k cells), avoiding the O(n²) cost
    of hashing the entire well at once.

    When ``global_model=True`` (default), a two-pass strategy is used to rescue tiles
    where per-tile RANSAC fails due to PH/SBS density mismatch:

    Pass 1 — run ``evaluate_match`` on every tile; classify tiles with det(R) ∈ [0.5, 2.0]
    and score > 0.05 as GOOD.  Extract their residual rotation angle.

    Pass 2 — compute ``R_global`` as the rotation at the median good angle (or identity if
    fewer than 2 good tiles), then for each tile estimate a per-tile translation via
    cKDTree nearest-neighbour robust median and call ``merge_triangle_hash`` with
    ``(R_global, t_tile)``.

    When ``global_model=False``, the original per-tile independent RANSAC path is used.
    ``align_ratio`` applies only in this fallback path.

    When the PH footprint contains significantly more cells than the SBS tile (a common
    occurrence when PH is denser), the RANSAC triangle-hash alignment step sees a
    density-flooded point set that can produce spurious transforms. ``align_ratio``
    controls the maximum PH-to-SBS cell ratio used *for alignment only*: if
    ``len(ph_sub) > align_ratio * len(sbs_sub)``, PH is randomly subsampled to
    ``int(align_ratio * len(sbs_sub))`` cells before ``find_triangles`` /
    ``evaluate_match``. The full ``ph_sub`` is always forwarded to
    ``merge_triangle_hash`` so that matching still sees every candidate cell.

    Args:
        sbs_cells: SBS global cell table with columns gy, gx, tile, well, plate, cell.
        ph_cells: Phenotype global cell table with same required columns.
        coarse: Transform dict from coarse_align_dapi with keys rotation (2×2 ndarray)
            and translation (shape (2,) ndarray) mapping PH global px → SBS global px.
        sbs_offsets: Per-tile pixel offsets for SBS (TileOffsets).
        tile_shape: (height, width) of each SBS tile in pixels.
        threshold: Nearest-neighbour match threshold in pixels.
        local_refinement: None | "polynomial" | "thin_plate_spline" for refine_local_warp.
        warp_kwargs: Extra kwargs for refine_local_warp.
        evaluate_kwargs: Extra kwargs for evaluate_match (e.g. ransac_kwargs).
        margin_px: Pixels to expand each SBS tile footprint when gathering PH cells.
        min_cells: Minimum cells on each side required to attempt a hash merge.
        align_ratio: Maximum PH-to-SBS cell ratio for the RANSAC alignment step
            (``global_model=False`` only).  Set to ``float("inf")`` (default) to
            disable subsampling.
        global_model: If True (default), bootstrap a single R_global from confident
            tiles and apply it to every tile with a per-tile cKDTree translation
            estimate.  If False, use independent per-tile RANSAC (original behaviour).

    Returns:
        Deduplicated 1:1 matched-cell DataFrame with a ``subtile`` column (SBS tile id).
    """
    h, w = tile_shape
    R = np.asarray(coarse["rotation"], dtype=np.float64)
    t = np.asarray(coarse["translation"], dtype=np.float64)

    ph_xy = ph_cells[["gy", "gx"]].to_numpy(dtype=np.float64)
    ph_ref_xy = (R @ ph_xy.T).T + t
    ph_aug = ph_cells.copy()
    ph_aug["gy_ref"] = ph_ref_xy[:, 0]
    ph_aug["gx_ref"] = ph_ref_xy[:, 1]

    off_frame = sbs_offsets.to_frame().set_index("tile")
    out: list[pd.DataFrame] = []

    if global_model:
        # Pass 1: per-tile evaluate_match to collect good tiles
        ph_footprints: dict = {}   # tile_id -> (ph_sub, sbs_sub_r)
        tile_fits: dict = {}       # tile_id -> {"rot", "trans", "det", "score", "angle"}

        for tile_id, sbs_sub in sbs_cells.groupby("tile"):
            if tile_id not in off_frame.index:
                continue
            oy = float(off_frame.loc[tile_id, "y"])
            ox = float(off_frame.loc[tile_id, "x"])
            y0, y1 = oy - margin_px, oy + h + margin_px
            x0, x1 = ox - margin_px, ox + w + margin_px
            mask = (
                (ph_aug["gy_ref"] >= y0) & (ph_aug["gy_ref"] < y1)
                & (ph_aug["gx_ref"] >= x0) & (ph_aug["gx_ref"] < x1)
            )
            ph_sub = ph_aug[mask].reset_index(drop=True)
            sbs_sub_r = sbs_sub.reset_index(drop=True)
            if len(ph_sub) < min_cells or len(sbs_sub_r) < min_cells:
                continue
            ph_footprints[tile_id] = (ph_sub, sbs_sub_r)

            t_ph = find_triangles(
                ph_sub[["gy_ref", "gx_ref"]].rename(columns={"gy_ref": "i", "gx_ref": "j"})
            )
            t_sbs = find_triangles(
                sbs_sub_r[["gy", "gx"]].rename(columns={"gy": "i", "gx": "j"})
            )
            rot, trans, score = evaluate_match(t_ph, t_sbs, **(evaluate_kwargs or {}))
            if rot is not None:
                det = float(np.linalg.det(rot))
                angle = float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))
                tile_fits[int(tile_id)] = {
                    "rot": rot, "trans": trans, "det": det, "score": score, "angle": angle,
                }

        # Classify GOOD tiles: det in [0.5, 2.0] AND score > floor
        score_floor = 0.05
        good = [
            v for v in tile_fits.values()
            if 0.5 <= v["det"] <= 2.0 and v["score"] > score_floor
        ]
        good_angles = [v["angle"] for v in good]
        good_trans_list = [v["trans"] for v in good]

        if len(good) >= 2:
            global_angle = float(np.median(good_angles))
            theta_g = np.deg2rad(global_angle)
            R_global = np.array([
                [np.cos(theta_g), -np.sin(theta_g)],
                [np.sin(theta_g),  np.cos(theta_g)],
            ])
            t_global = np.median(good_trans_list, axis=0)
        else:
            R_global = np.eye(2)   # identity residual — use coarse rotation as-is
            t_global = t           # fall back to coarse translation

        # Pass 2: per-tile translation fine-tuning + merge using R_global
        for tile_id, (ph_sub, sbs_sub_r) in ph_footprints.items():
            ph_ij = ph_sub[["gy_ref", "gx_ref"]].to_numpy(dtype=np.float64)
            # Apply R_global + global translation: initial prediction already near SBS
            ph_in_sbs = (R_global @ ph_ij.T).T + t_global

            sbs_ij = sbs_sub_r[["gy", "gx"]].to_numpy(dtype=np.float64)
            tree = cKDTree(sbs_ij)
            dists, nn_idx = tree.query(ph_in_sbs, k=1)
            close = dists < 100.0
            if close.sum() >= 5:
                t_fine = np.median(sbs_ij[nn_idx[close]] - ph_in_sbs[close], axis=0)
                t_tile = t_global + t_fine
            else:
                t_tile = t_global

            ph_hash = ph_sub.assign(i=ph_sub["gy_ref"], j=ph_sub["gx_ref"])
            sbs_hash = sbs_sub_r.assign(i=sbs_sub_r["gy"], j=sbs_sub_r["gx"])
            m = merge_triangle_hash(
                ph_hash,
                sbs_hash,
                {"rotation": R_global, "translation": t_tile},
                threshold=threshold,
                local_refinement=local_refinement,
                warp_kwargs=warp_kwargs,
            )
            if len(m):
                m = m.copy()
                m["subtile"] = tile_id
                out.append(m)

    else:
        # Original per-tile independent RANSAC path
        for tile_id, sbs_sub in sbs_cells.groupby("tile"):
            if tile_id not in off_frame.index:
                continue
            oy = float(off_frame.loc[tile_id, "y"])
            ox = float(off_frame.loc[tile_id, "x"])

            y0, y1 = oy - margin_px, oy + h + margin_px
            x0, x1 = ox - margin_px, ox + w + margin_px
            mask = (
                (ph_aug["gy_ref"] >= y0)
                & (ph_aug["gy_ref"] < y1)
                & (ph_aug["gx_ref"] >= x0)
                & (ph_aug["gx_ref"] < x1)
            )
            ph_sub = ph_aug[mask].reset_index(drop=True)
            sbs_sub = sbs_sub.reset_index(drop=True)

            if len(ph_sub) < min_cells or len(sbs_sub) < min_cells:
                continue

            # Subsample PH for alignment only when it would flood RANSAC.
            if align_ratio < float("inf"):
                n_align = int(align_ratio * len(sbs_sub))
                ph_align = ph_sub.sample(n=n_align, random_state=0) if len(ph_sub) > n_align else ph_sub
            else:
                ph_align = ph_sub

            t_ph = find_triangles(
                ph_align[["gy_ref", "gx_ref"]].rename(columns={"gy_ref": "i", "gx_ref": "j"})
            )
            t_sbs = find_triangles(
                sbs_sub[["gy", "gx"]].rename(columns={"gy": "i", "gx": "j"})
            )
            rot, trans, _score = evaluate_match(t_ph, t_sbs, **(evaluate_kwargs or {}))
            if rot is None:
                continue

            ph_hash = ph_sub.assign(i=ph_sub["gy_ref"], j=ph_sub["gx_ref"])
            sbs_hash = sbs_sub.assign(i=sbs_sub["gy"], j=sbs_sub["gx"])

            m = merge_triangle_hash(
                ph_hash,
                sbs_hash,
                {"rotation": rot, "translation": trans},
                threshold=threshold,
                local_refinement=local_refinement,
                warp_kwargs=warp_kwargs,
            )
            if len(m):
                m = m.copy()
                m["subtile"] = tile_id
                out.append(m)

    if not out:
        return ph_cells.head(0).assign(subtile=pd.Series(dtype=int))

    combined = pd.concat(out, ignore_index=True)
    if "mapped_single_gene" not in combined.columns:
        combined["mapped_single_gene"] = False

    combined = deduplicate_cells(
        combined,
        approach="fast",
        mapped_single_gene=False,
        sbs_dedup_prior={"distance": True},
        pheno_dedup_prior={"distance": True},
    )

    return combined
