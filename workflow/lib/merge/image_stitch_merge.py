"""Sub-tile hash merge over image-stitched global cell frames."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from workflow.lib.merge.hash import find_triangles, evaluate_match
from workflow.lib.merge.fast_merge import merge_triangle_hash
from workflow.lib.merge.deduplicate_merge import deduplicate_cells
from workflow.lib.shared.stitching.types import TileOffsets

logger = logging.getLogger(__name__)


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



def _offsets_index_frame(sbs_offsets: "dict | TileOffsets") -> pd.DataFrame:
    """Normalize SBS offsets into a tile-indexed frame with ``y``/``x`` columns.

    Accepts either a :class:`TileOffsets` (via ``to_frame``) or a plain mapping
    ``{tile_id: (y, x)}`` so callers can pass whichever they already hold.

    Args:
        sbs_offsets: TileOffsets or ``{tile_id: (y, x)}`` mapping in SBS pixels.

    Returns:
        DataFrame indexed by ``tile`` with float ``y`` and ``x`` columns.
    """
    if hasattr(sbs_offsets, "to_frame"):
        return sbs_offsets.to_frame().set_index("tile")
    rows = [
        {"tile": int(t), "y": float(v[0]), "x": float(v[1])}
        for t, v in dict(sbs_offsets).items()
    ]
    return pd.DataFrame(rows).set_index("tile")


def _merge_phtile_pair(
    ph_sub: pd.DataFrame,
    sbs_sub: pd.DataFrame,
    alignment: dict,
    sbs_tile_id: int,
    ph_tile_id: int,
    threshold: float,
    local_refinement: str | None,
    warp_kwargs: dict | None,
) -> pd.DataFrame:
    """Triangle-hash merge one (SBS tile, PH tile) pair under a fixed alignment.

    Args:
        ph_sub: Phenotype cells of the PH tile, carrying coarse-projected
            ``gy_ref``/``gx_ref`` columns plus id columns (plate, well, tile, cell).
        sbs_sub: SBS cells of the SBS tile, carrying ``gy``/``gx`` plus id columns.
        alignment: Dict with ``rotation`` (2x2) and ``translation`` (2,) mapping
            PH reference coords -> SBS coords.
        sbs_tile_id: SBS tile id (stamped into the ``subtile`` column).
        ph_tile_id: PH tile id (stamped into the ``ph_tile`` column).
        threshold: Nearest-neighbour match threshold in SBS pixels.
        local_refinement: None | "polynomial" | "thin_plate_spline" for the warp.
        warp_kwargs: Extra kwargs for the local warp model.

    Returns:
        Matched-pair DataFrame (possibly empty) with ``subtile``/``ph_tile`` stamped.
    """
    ph_hash = ph_sub.assign(i=ph_sub["gy_ref"], j=ph_sub["gx_ref"])
    sbs_hash = sbs_sub.assign(i=sbs_sub["gy"], j=sbs_sub["gx"])
    m = merge_triangle_hash(
        ph_hash,
        sbs_hash,
        alignment,
        threshold=threshold,
        local_refinement=local_refinement,
        warp_kwargs=warp_kwargs,
    )
    if len(m):
        m = m.copy()
        m["subtile"] = int(sbs_tile_id)
        m["ph_tile"] = int(ph_tile_id)
    return m


def merge_reference_tiles_per_phtile(
    sbs_cells: pd.DataFrame,
    ph_cells: pd.DataFrame,
    coarse: dict,
    sbs_offsets: dict,
    tile_shape: tuple[int, int],
    margin_px: int = 200,
    min_overlap_cells: int = 30,
    threshold: float = 4.0,
    local_refinement: bool = True,
    warp_kwargs: dict | None = None,
    neg_det_fallback: bool = True,
) -> pd.DataFrame:
    """Hash-merge SBS and phenotype cells one (SBS tile, PH tile) pair at a time.

    ``merge_reference_tiles`` hashes each SBS tile footprint against every
    coarse-projected phenotype cell inside it at once. When one SBS tile overlaps
    several phenotype tiles (inevitable when the phenotype scope has a finer pixel
    size), that footprint mixes cells from multiple PH tiles that each carry a
    slightly different residual local transform, so a single per-footprint affine
    cannot align them and recall collapses.

    This variant instead hashes each SBS tile against every overlapping PH tile
    **separately**, keeping each point set coherent under one residual transform.
    Phenotype cells are first coarse-aligned into the SBS reference frame
    (``gy_ref``/``gx_ref``) via ``coarse`` exactly as ``merge_reference_tiles``
    does. For each SBS tile ``S`` the overlapping PH tiles are those with at least
    ``min_overlap_cells`` coarse-projected cells inside ``S``'s footprint bbox
    (from ``sbs_offsets`` + ``tile_shape``, expanded by ``margin_px``). For each
    such PH tile ``P``, ``find_triangles`` + ``evaluate_match`` estimate a local
    rotation/translation; pairs whose rotation is ``None`` or a reflection
    (``det < 0``) are skipped, otherwise the full PH tile is matched to ``S`` via
    ``merge_triangle_hash``.

    Neg-det global-model fallback: the confident (``det > 0``) per-pair rotations
    define ``R_global`` (rotation at their median angle) and a seed translation
    (median of their translations). When ``neg_det_fallback`` is True, each pair
    that failed the per-pair estimate is retried once with ``R_global`` plus a
    per-pair robust-median translation refined by ``cKDTree`` nearest-neighbours,
    rather than being dropped. All surviving pairs are unioned and deduplicated to
    a strict 1:1 mapping via ``deduplicate_cells``.

    Args:
        sbs_cells: SBS global cell table with columns ``gy``, ``gx``, ``tile``,
            ``well``, ``plate``, ``cell``.
        ph_cells: Phenotype global cell table with the same required columns.
        coarse: Transform dict from ``coarse_align_dapi`` with keys ``rotation``
            (2x2 ndarray) and ``translation`` (shape (2,) ndarray) mapping PH
            global px -> SBS global px.
        sbs_offsets: Per-tile SBS pixel offsets as a ``TileOffsets`` or a plain
            ``{tile_id: (y, x)}`` mapping.
        tile_shape: (height, width) of each SBS tile in pixels.
        margin_px: Pixels to expand each SBS tile footprint when gathering PH cells.
        min_overlap_cells: Minimum coarse-projected PH cells inside a footprint for
            a PH tile to count as overlapping (also the minimum SBS tile size).
        threshold: Nearest-neighbour match threshold in SBS pixels.
        local_refinement: If True (default), refine each pair with a thin-plate-spline
            local warp; if False, use the plain affine. A model-name string
            ("polynomial" | "thin_plate_spline") is also accepted and forwarded.
        warp_kwargs: Extra kwargs for the local warp model.
        neg_det_fallback: If True (default), retry failed (rotation None or det<0)
            pairs with the bootstrapped ``R_global`` + per-pair translation instead
            of dropping them.

    Returns:
        Deduplicated strict-1:1 matched-cell DataFrame with ``subtile`` (SBS tile
        id) and ``ph_tile`` (PH tile id) columns.
    """
    h, w = tile_shape
    R = np.asarray(coarse["rotation"], dtype=np.float64)
    t = np.asarray(coarse["translation"], dtype=np.float64)

    # Coarse-project every PH cell into the SBS reference frame.
    ph_xy = ph_cells[["gy", "gx"]].to_numpy(dtype=np.float64)
    ph_ref_xy = (R @ ph_xy.T).T + t
    ph_aug = ph_cells.copy()
    ph_aug["gy_ref"] = ph_ref_xy[:, 0]
    ph_aug["gx_ref"] = ph_ref_xy[:, 1]

    # One coherent point set per original PH tile.
    ph_by_tile = {
        int(tid): grp.reset_index(drop=True) for tid, grp in ph_aug.groupby("tile")
    }

    off_frame = _offsets_index_frame(sbs_offsets)
    evaluate_kwargs = {"ransac_kwargs": {"random_state": 0}}

    # Map local_refinement (bool default) -> merge_triangle_hash's model argument.
    if local_refinement is True:
        refine: str | None = "thin_plate_spline"
    elif local_refinement:
        refine = local_refinement  # explicit model-name string
    else:
        refine = None

    out: list[pd.DataFrame] = []
    valid_rotations: list[np.ndarray] = []
    valid_translations: list[np.ndarray] = []
    failed_pairs: list[tuple[int, int, pd.DataFrame, pd.DataFrame]] = []

    for sbs_tile_id, sbs_sub in sbs_cells.groupby("tile"):
        if sbs_tile_id not in off_frame.index:
            continue
        oy = float(off_frame.loc[sbs_tile_id, "y"])
        ox = float(off_frame.loc[sbs_tile_id, "x"])
        y0, y1 = oy - margin_px, oy + h + margin_px
        x0, x1 = ox - margin_px, ox + w + margin_px

        sbs_sub_r = sbs_sub.reset_index(drop=True)
        if len(sbs_sub_r) < min_overlap_cells:
            continue

        for ph_tile_id, ph_tile_cells in ph_by_tile.items():
            gy_r = ph_tile_cells["gy_ref"].to_numpy()
            gx_r = ph_tile_cells["gx_ref"].to_numpy()
            overlap = (gy_r >= y0) & (gy_r < y1) & (gx_r >= x0) & (gx_r < x1)
            if int(overlap.sum()) < min_overlap_cells:
                continue

            ph_sub = ph_tile_cells  # full coherent PH tile
            if len(ph_sub) < min_overlap_cells:
                continue

            t_ph = find_triangles(
                ph_sub[["gy_ref", "gx_ref"]].rename(
                    columns={"gy_ref": "i", "gx_ref": "j"}
                )
            )
            t_sbs = find_triangles(
                sbs_sub_r[["gy", "gx"]].rename(columns={"gy": "i", "gx": "j"})
            )
            rot, trans, _score = evaluate_match(t_ph, t_sbs, **evaluate_kwargs)

            if rot is None:
                logger.debug(
                    "per-phtile skip S=%s P=%s: no rotation", sbs_tile_id, ph_tile_id
                )
                failed_pairs.append(
                    (int(sbs_tile_id), int(ph_tile_id), ph_sub, sbs_sub_r)
                )
                continue

            det = float(np.linalg.det(rot))
            if det < 0:
                logger.debug(
                    "per-phtile skip S=%s P=%s: reflection det=%.4f",
                    sbs_tile_id,
                    ph_tile_id,
                    det,
                )
                failed_pairs.append(
                    (int(sbs_tile_id), int(ph_tile_id), ph_sub, sbs_sub_r)
                )
                continue

            valid_rotations.append(rot)
            valid_translations.append(np.asarray(trans, dtype=np.float64))

            m = _merge_phtile_pair(
                ph_sub,
                sbs_sub_r,
                {"rotation": rot, "translation": trans},
                sbs_tile_id,
                ph_tile_id,
                threshold,
                refine,
                warp_kwargs,
            )
            if len(m):
                out.append(m)

    # Neg-det global-model fallback: rescue failed pairs with a bootstrapped model.
    if neg_det_fallback and failed_pairs and valid_rotations:
        angles = [
            float(np.degrees(np.arctan2(r[1, 0], r[0, 0]))) for r in valid_rotations
        ]
        theta_g = np.deg2rad(float(np.median(angles)))
        R_global = np.array(
            [
                [np.cos(theta_g), -np.sin(theta_g)],
                [np.sin(theta_g), np.cos(theta_g)],
            ]
        )
        t_global = np.median(np.vstack(valid_translations), axis=0)

        for sbs_tile_id, ph_tile_id, ph_sub, sbs_sub_r in failed_pairs:
            ph_ij = ph_sub[["gy_ref", "gx_ref"]].to_numpy(dtype=np.float64)
            ph_in_sbs = (R_global @ ph_ij.T).T + t_global
            sbs_ij = sbs_sub_r[["gy", "gx"]].to_numpy(dtype=np.float64)
            tree = cKDTree(sbs_ij)
            dists, nn_idx = tree.query(ph_in_sbs, k=1)
            close = dists < 100.0
            if int(close.sum()) >= 5:
                t_fine = np.median(sbs_ij[nn_idx[close]] - ph_in_sbs[close], axis=0)
                t_tile = t_global + t_fine
            else:
                t_tile = t_global

            m = _merge_phtile_pair(
                ph_sub,
                sbs_sub_r,
                {"rotation": R_global, "translation": t_tile},
                sbs_tile_id,
                ph_tile_id,
                threshold,
                "thin_plate_spline",
                warp_kwargs,
            )
            if len(m):
                out.append(m)

    if not out:
        return ph_cells.head(0).assign(
            subtile=pd.Series(dtype=int), ph_tile=pd.Series(dtype=int)
        )

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
