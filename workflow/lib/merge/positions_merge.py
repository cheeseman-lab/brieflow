"""Positions merge: match phenotype and SBS cells from tile positions and cell centroids.

The approach reads only per-tile cell tables and tile metadata. Each cell is placed in a
shared micrometre frame from its tile's stage position and its local centroid. A per-modality
camera matrix (scale, rotation, shear) plus one translation is fitted by iterative closest
point on the centroids, then a per-phenotype-tile translation absorbs stage repeatability.
Cells are matched one-to-one by mutual nearest neighbour within the merge threshold. No
stitched image, mask or image registration is built.
"""

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import lsqr
from scipy.spatial import cKDTree

MERGE_COLUMNS = [
    "plate",
    "well",
    "tile",
    "cell_0",
    "i_0",
    "j_0",
    "site",
    "cell_1",
    "i_1",
    "j_1",
    "distance",
]
ICP_RADII_UM = (20.0, 12.0, 8.0, 6.0, 5.0, 4.0, 4.0, 4.0)
TILE_SHIFT_RADII_UM = (8.0, 4.0, 4.0)
TILE_SHIFT_MIN_PAIRS = 5
TILE_SHIFT_MAX_MAD_UM = 2.0
TILE_SHIFT_PRIOR_WEIGHT = 0.1
SEAM_MIN_CELLS = 50
SEAM_MIN_AREA_RATIO = 0.9
DEDUP_RADIUS_UM = 6.0
COARSE_BIN_UM = 10.0
COARSE_MAX_SHIFT_UM = 500.0
COARSE_MIN_PEAK_RATIO = 3.0
SEAM_SEARCH_UM = 25.0
SEAM_TOLERANCE_UM = 3.0
SEAM_WARN_RATIO = 0.8
LOW_MATCH_RATE = 0.2
CAMERA_SCALE_RANGE = (0.8, 1.25)
ORIENTATIONS = [
    (flipud, fliplr, rot90)
    for rot90 in (0, 1)
    for flipud in (False, True)
    for fliplr in (False, True)
]


def positions_merge(
    phenotype_info,
    sbs_info,
    phenotype_metadata,
    sbs_metadata,
    phenotype_dimensions,
    sbs_dimensions,
    threshold=2,
    flipud=False,
    fliplr=False,
    rot90=0,
    phenotype_pixel_size=None,
    sbs_pixel_size=None,
):
    """Match phenotype cells to SBS cells of one well from positions alone.

    Works the same with dense and sparse tiles: the camera model is fitted on all cells of
    the well, and a tile with too few cells or inconsistent residuals borrows its shift from
    neighbouring tiles or keeps the well-level model.

    Args:
        phenotype_info (pandas.DataFrame): Phenotype cells with `plate`, `well`, `tile`,
            `cell`, `i`, `j` (local pixel centroid).
        sbs_info (pandas.DataFrame): SBS cells with the same columns.
        phenotype_metadata (pandas.DataFrame): One row per phenotype tile with `tile`,
            `x_pos`, `y_pos` (stage, um) and optionally `pixel_size_x`.
        sbs_metadata (pandas.DataFrame): One row per SBS tile, same columns.
        phenotype_dimensions (tuple): Phenotype tile (height, width) in pixels.
        sbs_dimensions (tuple): SBS tile (height, width) in pixels.
        threshold (float): Maximum match distance in SBS pixels. Defaults to 2.
        flipud (bool): Tile rows run against stage y. Defaults to False.
        fliplr (bool): Tile columns run against stage x. Defaults to False.
        rot90 (int): Counterclockwise quarter turns between tile and stage axes.
        phenotype_pixel_size (float | None): Fallback um/pixel when metadata has none.
        sbs_pixel_size (float | None): Fallback um/pixel when metadata has none.

    Returns:
        tuple: (merge DataFrame with the `fast_merge` columns, one-row QC DataFrame,
            placement dict for `tile_pixel_to_um`, or None when there are too few cells).
    """
    orientation = (bool(flipud), bool(fliplr), int(rot90) % 4)
    ph = _place_cells(
        phenotype_info,
        phenotype_metadata,
        phenotype_dimensions,
        orientation,
        _pixel_size(phenotype_metadata, phenotype_pixel_size, "phenotype"),
    )
    sbs = _place_cells(
        sbs_info,
        sbs_metadata,
        sbs_dimensions,
        orientation,
        _pixel_size(sbs_metadata, sbs_pixel_size, "sbs"),
    )
    qc = {
        "n_phenotype_cells": len(ph["cells"]),
        "n_sbs_cells": len(sbs["cells"]),
        "phenotype_cells_per_tile_median": _cells_per_tile(ph),
        "sbs_cells_per_tile_median": _cells_per_tile(sbs),
    }
    qc.update(
        _orientation_qc(
            phenotype_info,
            phenotype_metadata,
            phenotype_dimensions,
            ph["pixel_size"],
            orientation,
            "phenotype",
        )
    )
    qc.update(
        _orientation_qc(
            sbs_info,
            sbs_metadata,
            sbs_dimensions,
            sbs["pixel_size"],
            orientation,
            "sbs",
        )
    )
    if len(ph["cells"]) < 4 or len(sbs["cells"]) < 4:
        qc["status"] = "insufficient_cells"
        return pd.DataFrame(columns=MERGE_COLUMNS), pd.DataFrame([qc]), None

    shift, peak_ratio = coarse_translation(_nominal_xy(ph), _nominal_xy(sbs))
    qc.update(
        coarse_shift_x_um=shift[0],
        coarse_shift_y_um=shift[1],
        coarse_peak_ratio=peak_ratio,
    )
    model, icp_qc = fit_camera_model(ph, sbs, shift)
    qc.update(icp_qc)
    ph_xy = _apply_model(ph, model, "phenotype")
    sbs_xy = _apply_model(sbs, model, "sbs")
    shifts = fit_tile_shifts(ph, ph_xy, sbs, sbs_xy)
    ph_xy = ph_xy + _point_shifts(shifts["phenotype"], ph)
    sbs_xy = sbs_xy + _point_shifts(shifts["sbs"], sbs)
    for name, table in shifts.items():
        norm = np.hypot(table["dx"], table["dy"])
        qc[f"{name}_tile_shift_median_um"] = float(norm.median())
        qc[f"{name}_tile_shift_max_um"] = float(norm.max())
        for source in ("own", "neighbour", "global"):
            qc[f"{name}_tiles_shift_{source}"] = int((table["source"] == source).sum())

    keep_ph = deduplicate_overlap(ph_xy, ph["tiles"], ph["centrality"])
    keep_sbs = deduplicate_overlap(sbs_xy, sbs["tiles"], sbs["centrality"])
    qc.update(n_phenotype_kept=int(keep_ph.sum()), n_sbs_kept=int(keep_sbs.sum()))
    ph_cells = ph["cells"][keep_ph].reset_index(drop=True)
    sbs_cells = sbs["cells"][keep_sbs].reset_index(drop=True)
    pairs = mutual_nearest_pairs(
        ph_xy[keep_ph], sbs_xy[keep_sbs], threshold * sbs["pixel_size"]
    )
    merged = _merge_frame(ph_cells, sbs_cells, pairs, sbs["pixel_size"])
    qc.update(_match_qc(merged, ph_cells, sbs_cells))
    qc["status"] = _status(qc)
    placement = {
        name: {
            "pixel_size": placed["pixel_size"],
            "dimensions": tuple(dims),
            "orientation": orientation,
            "matrix": model[name],
            "radial": model["radial"][name],
            "offset": model["t"] if name == "phenotype" else np.zeros(2),
            "tiles": shifts[name],
        }
        for name, placed, dims in (
            ("phenotype", ph, phenotype_dimensions),
            ("sbs", sbs, sbs_dimensions),
        )
    }
    return merged, pd.DataFrame([qc]), placement


def tile_pixel_to_um(placement, tile, pixels):
    """Map raw pixel coordinates (i, j) of one tile to the shared frame in um.

    Args:
        placement (dict): One modality's entry of the placement returned by `positions_merge`.
        tile (int): Tile id.
        pixels (numpy.ndarray): (n, 2) raw (i, j) coordinates.

    Returns:
        numpy.ndarray: (n, 2) x/y positions in um.
    """
    pixels = np.atleast_2d(np.asarray(pixels, dtype=float))
    i, j, (height, width) = orient_local(
        pixels[:, 0], pixels[:, 1], placement["dimensions"], *placement["orientation"]
    )
    half = np.array([(width - 1) / 2, (height - 1) / 2])
    u = np.column_stack([j, i]) - half
    xy = u @ placement["matrix"].T
    xy = xy + _radial_terms(u, half, placement["pixel_size"]) @ placement["radial"]
    row = placement["tiles"].loc[tile]
    return (
        xy
        + np.array([row["x_pos"] + row["dx"], row["y_pos"] + row["dy"]])
        + placement["offset"]
    )


def um_to_tile_pixel(placement, tile, xy, iterations=4):
    """Inverse of `tile_pixel_to_um`: positions in um to raw (i, j) of one tile.

    Args:
        placement (dict): One modality's placement.
        tile (int): Tile id.
        xy (numpy.ndarray): (n, 2) positions in um.
        iterations (int): Fixed-point iterations for the radial term.

    Returns:
        numpy.ndarray: (n, 2) raw (i, j) coordinates.
    """
    xy = np.atleast_2d(np.asarray(xy, dtype=float))
    probe = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    linear = dict(placement, radial=np.zeros(2))
    base = tile_pixel_to_um(linear, tile, probe)
    a = np.column_stack([base[1] - base[0], base[2] - base[0]])
    pixels = np.linalg.solve(a, (xy - base[0]).T).T
    for _ in range(iterations):
        bend = tile_pixel_to_um(placement, tile, pixels) - tile_pixel_to_um(
            linear, tile, pixels
        )
        pixels = np.linalg.solve(a, (xy - bend - base[0]).T).T
    return pixels


def filter_tile_metadata(metadata, cycle=None, channel=None):
    """Reduce combined metadata to one row per tile.

    Args:
        metadata (pandas.DataFrame): Combined metadata, possibly one row per channel or cycle.
        cycle (int | None): Keep only this `cycle` when given.
        channel (str | None): Keep only this `channel` when given.

    Returns:
        pandas.DataFrame: One row per (plate, well, tile).
    """
    if cycle is not None and "cycle" in metadata.columns:
        metadata = metadata[metadata["cycle"] == cycle]
    if channel is not None:
        metadata = metadata[metadata["channel"] == channel]
    return metadata.drop_duplicates(subset=["plate", "well", "tile"]).reset_index(
        drop=True
    )


def orient_local(i, j, dimensions, flipud=False, fliplr=False, rot90=0):
    """Map local pixel coordinates the way `augment_tile` maps the tile image.

    Applies the vertical flip, then the horizontal flip, then `rot90` counterclockwise
    quarter turns (`numpy.rot90`), so positions and stitched previews agree.

    Args:
        i (numpy.ndarray): Row coordinates.
        j (numpy.ndarray): Column coordinates.
        dimensions (tuple): Tile (height, width) before transformation.
        flipud (bool): Flip rows.
        fliplr (bool): Flip columns.
        rot90 (int): Counterclockwise quarter turns.

    Returns:
        tuple: (i, j, (height, width)) after transformation.
    """
    i = np.asarray(i, dtype=float)
    j = np.asarray(j, dtype=float)
    height, width = dimensions
    if flipud:
        i = height - 1 - i
    if fliplr:
        j = width - 1 - j
    for _ in range(int(rot90) % 4):
        i, j = width - 1 - j, i
        height, width = width, height
    return i, j, (height, width)


def coarse_translation(points_0, points_1):
    """Estimate the translation from points_0 to points_1 by density cross-correlation.

    Args:
        points_0 (numpy.ndarray): (n, 2) x/y positions in um.
        points_1 (numpy.ndarray): (m, 2) x/y positions in um.

    Returns:
        tuple: (shift (2,), ratio of the correlation peak over the 99th percentile within
            `COARSE_MAX_SHIFT_UM`). The shift is zero when the peak is not distinct.
    """
    lo = np.minimum(points_0.min(0), points_1.min(0)) - COARSE_MAX_SHIFT_UM
    hi = np.maximum(points_0.max(0), points_1.max(0)) + COARSE_MAX_SHIFT_UM
    bins = [np.arange(lo[k], hi[k] + COARSE_BIN_UM, COARSE_BIN_UM) for k in range(2)]
    h0 = np.histogram2d(points_0[:, 0], points_0[:, 1], bins=bins)[0]
    h1 = np.histogram2d(points_1[:, 0], points_1[:, 1], bins=bins)[0]
    corr = np.fft.irfft2(
        np.fft.rfft2(h1 - h1.mean()) * np.conj(np.fft.rfft2(h0 - h0.mean())), s=h0.shape
    )
    reach = int(COARSE_MAX_SHIFT_UM // COARSE_BIN_UM)
    lags = np.arange(-reach, reach + 1)
    window = corr[np.ix_(lags % h0.shape[0], lags % h0.shape[1])]
    peak = np.unravel_index(np.argmax(window), window.shape)
    background = np.percentile(np.abs(window), 99)
    ratio = float(window[peak] / background) if background > 0 else 0.0
    if ratio < COARSE_MIN_PEAK_RATIO:
        return np.zeros(2), ratio
    return lags[list(peak)] * COARSE_BIN_UM, ratio


def fit_camera_model(ph, sbs, initial_shift):
    """Fit per-modality camera matrices and one translation by iterative closest point.

    Each modality places a cell at `stage + C u + p (k1 r^2 + k2 r^4) u`, with `u` the oriented
    local offset from the tile centre in pixels, `r` its distance from the centre relative to
    the half tile, `p` the pixel size and `k1`, `k2` a radial lens distortion; the phenotype
    frame adds one translation `t`. Phenotype-to-SBS nearest neighbours and cells seen by two
    overlapping tiles of one modality are both fitted, which ties each camera to the stage
    pitch so neighbouring tiles of one modality also line up.

    Args:
        ph (dict): Placed phenotype cells from `_place_cells`.
        sbs (dict): Placed SBS cells from `_place_cells`.
        initial_shift (numpy.ndarray): Starting translation in um.

    Returns:
        tuple: (model dict with `phenotype`, `sbs` matrices and `t`, QC dict).
    """
    model = {
        "phenotype": np.eye(2) * ph["pixel_size"],
        "sbs": np.eye(2) * sbs["pixel_size"],
        "t": np.asarray(initial_shift, dtype=float),
        "radial": {"phenotype": np.zeros(2), "sbs": np.zeros(2)},
    }
    n_pairs, residual, fit_failed = 0, np.nan, False
    seam_pairs = {"phenotype": 0, "sbs": 0}
    for radius in ICP_RADII_UM:
        q = _apply_model(ph, model, "phenotype")
        s = _apply_model(sbs, model, "sbs")
        dist, idx = cKDTree(s).query(q, distance_upper_bound=radius)
        ok = np.isfinite(dist)
        if ok.sum() < 10:
            break
        n_pairs, residual = int(ok.sum()), float(np.median(dist[ok]))
        seams = {
            "phenotype": _seam_pairs(ph, q, radius),
            "sbs": _seam_pairs(sbs, s, radius),
        }
        seam_pairs = {name: len(pair[0]) for name, pair in seams.items()}
        candidate = _solve_camera(ph, np.flatnonzero(ok), sbs, idx[ok], seams)
        if not _plausible(candidate, ph, sbs):
            fit_failed = True
            break
        model = candidate
    qc = {
        "icp_pairs": n_pairs,
        "icp_median_residual_um": residual,
        "phenotype_seam_pairs": seam_pairs["phenotype"],
        "sbs_seam_pairs": seam_pairs["sbs"],
        "translation_x_um": float(model["t"][0]),
        "translation_y_um": float(model["t"][1]),
    }
    for name, placed in (("phenotype", ph), ("sbs", sbs)):
        scale, angle = _scale_rotation(model[name] / placed["pixel_size"])
        qc[f"{name}_camera_scale"] = scale
        qc[f"{name}_camera_rotation_deg"] = angle
        qc[f"{name}_radial_edge_um"] = float(
            np.sum(model["radial"][name]) * placed["pixel_size"] * max(placed["half"])
        )
    return model, qc


def fit_tile_shifts(ph, ph_points, sbs, sbs_points):
    """Estimate one translation per phenotype tile and per SBS tile (stage repeatability).

    One sparse least-squares solve over all tiles of both modalities. Each observation is the
    median offset between two tiles: phenotype-to-SBS nearest neighbours, or the two copies of
    cells in a seam of one modality. A tile pair only counts with at least
    `TILE_SHIFT_MIN_PAIRS` pairs agreeing within `TILE_SHIFT_MAX_MAD_UM`, so sparse tiles
    contribute nothing unreliable; a weak prior keeps tiles without observations at the
    well-level model. Relinearised over `TILE_SHIFT_RADII_UM`.

    Args:
        ph (dict): Placed phenotype cells from `_place_cells`.
        ph_points (numpy.ndarray): (n, 2) phenotype positions in um.
        sbs (dict): Placed SBS cells from `_place_cells`.
        sbs_points (numpy.ndarray): (m, 2) SBS positions in um.

    Returns:
        dict: Per modality, a DataFrame indexed by tile with `x_pos`, `y_pos`, `dx`, `dy`
            (um) and `source`: "own" (tied to the other modality by its own cells),
            "neighbour" (tied only through seams with neighbouring tiles) or "global".
    """
    n_ph = len(ph["tile_table"])
    keys = {"phenotype": ph["tile_table"].index, "sbs": sbs["tile_table"].index}
    shift = np.zeros((n_ph + len(keys["sbs"]), 2))
    for radius in TILE_SHIFT_RADII_UM:
        ph_xy = ph_points + shift[ph["tile_table"].index.get_indexer(ph["tiles"])]
        sbs_xy = (
            sbs_points + shift[n_ph + sbs["tile_table"].index.get_indexer(sbs["tiles"])]
        )
        obs = []
        dist, idx = cKDTree(sbs_xy).query(ph_xy, distance_upper_bound=radius)
        ok = np.flatnonzero(np.isfinite(dist))
        obs.append(
            _pair_medians(
                ph["tile_table"].index.get_indexer(ph["tiles"][ok]),
                n_ph + sbs["tile_table"].index.get_indexer(sbs["tiles"][idx[ok]]),
                sbs_xy[idx[ok]] - ph_xy[ok],
                "cross",
            )
        )
        for placed, points, base in ((ph, ph_xy, 0), (sbs, sbs_xy, n_ph)):
            a, b = _seam_pairs(placed, points, radius)
            index = placed["tile_table"].index
            obs.append(
                _pair_medians(
                    base + index.get_indexer(placed["tiles"][a]),
                    base + index.get_indexer(placed["tiles"][b]),
                    points[b] - points[a],
                    "seam",
                )
            )
        obs = pd.concat(obs, ignore_index=True)
        shift += _solve_tile_shifts(obs, len(shift))
    cross_tiles = set(obs.loc[obs["kind"] == "cross", ["a", "b"]].to_numpy().ravel())
    seam_tiles = set(obs.loc[obs["kind"] == "seam", ["a", "b"]].to_numpy().ravel())
    tables = {}
    for name, placed, base in (("phenotype", ph, 0), ("sbs", sbs, n_ph)):
        rows = base + np.arange(len(keys[name]))
        source = np.where(
            [r in cross_tiles for r in rows],
            "own",
            np.where([r in seam_tiles for r in rows], "neighbour", "global"),
        )
        tables[name] = placed["tile_table"].assign(
            dx=shift[rows, 0], dy=shift[rows, 1], source=source
        )
    return tables


def mutual_nearest_pairs(points_0, points_1, max_distance):
    """Pair points that are each other's nearest neighbour closer than max_distance.

    Args:
        points_0 (numpy.ndarray): (n, 2) positions.
        points_1 (numpy.ndarray): (m, 2) positions.
        max_distance (float): Strict upper bound on the pair distance.

    Returns:
        numpy.ndarray: (k, 3) rows of (index into points_0, index into points_1, distance).
    """
    dist, idx = cKDTree(points_1).query(points_0, distance_upper_bound=max_distance)
    keep = np.isfinite(dist) & (dist < max_distance)
    i0 = np.flatnonzero(keep)
    i1 = idx[keep]
    _, back = cKDTree(points_0).query(points_1[i1])
    mutual = back == i0
    return np.column_stack([i0[mutual], i1[mutual], dist[keep][mutual]])


def deduplicate_overlap(points, tiles, centrality, radius=DEDUP_RADIUS_UM):
    """Keep one record per cell imaged by several overlapping tiles.

    Records from different tiles closer than `radius` are linked; each connected group
    keeps the record lying most centrally in its own tile.

    Args:
        points (numpy.ndarray): (n, 2) positions in um, after the camera model.
        tiles (numpy.ndarray): Tile id of each record.
        centrality (numpy.ndarray): Normalised distance of each record from its tile centre.
        radius (float): Linking distance in um.

    Returns:
        numpy.ndarray: Boolean mask of records to keep.
    """
    keep = np.ones(len(points), dtype=bool)
    pairs = cKDTree(points).query_pairs(radius, output_type="ndarray")
    pairs = pairs[tiles[pairs[:, 0]] != tiles[pairs[:, 1]]]
    if len(pairs) == 0:
        return keep
    graph = coo_matrix(
        (np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(points),) * 2
    )
    labels = connected_components(graph, directed=False)[1]
    members = np.unique(pairs)
    groups = pd.DataFrame(
        {"index": members, "group": labels[members], "centrality": centrality[members]}
    )
    winners = groups.sort_values("centrality").drop_duplicates("group")["index"].values
    keep[members] = False
    keep[winners] = True
    return keep


def seam_agreement(info, metadata, dimensions, pixel_size, orientation):
    """Score how well cells seen by two overlapping tiles coincide under an orientation.

    For every cell owned by a neighbouring tile, the nearest cell of that tile is found.
    Under the right orientation the displacements of one tile pair agree; the score is the
    fraction within `SEAM_TOLERANCE_UM` of their tile pair's median displacement.

    Args:
        info (pandas.DataFrame): Cells with `tile`, `i`, `j`.
        metadata (pandas.DataFrame): One row per tile with `tile`, `x_pos`, `y_pos`.
        dimensions (tuple): Tile (height, width) in pixels.
        pixel_size (float): um per pixel.
        orientation (tuple): (flipud, fliplr, rot90).

    Returns:
        float: Agreement in [0, 1], or NaN when no tiles overlap.
    """
    placed = _place_cells(info, metadata, dimensions, orientation, pixel_size)
    xy = _nominal_xy(placed)
    tiles = placed["tiles"]
    seam = np.flatnonzero(~placed["owned"])
    if len(seam) < SEAM_MIN_CELLS:
        return float("nan")
    rows = []
    for other, members in pd.Series(seam).groupby(placed["owner"][seam]):
        dist, idx = cKDTree(xy[tiles == other]).query(
            xy[members.values], distance_upper_bound=SEAM_SEARCH_UM
        )
        ok = np.isfinite(dist)
        disp = xy[tiles == other][idx[ok]] - xy[members.values[ok]]
        rows.append(
            pd.DataFrame(
                {
                    "src": tiles[members.values[ok]],
                    "dst": other,
                    "dx": disp[:, 0],
                    "dy": disp[:, 1],
                }
            )
        )
    disp = pd.concat(rows, ignore_index=True)
    med = disp.groupby(["src", "dst"])[["dx", "dy"]].transform("median")
    close = (
        np.hypot(disp["dx"] - med["dx"], disp["dy"] - med["dy"]) <= SEAM_TOLERANCE_UM
    )
    return float(close.sum() / len(seam))


def _pair_medians(a, b, offsets, kind):
    """Median offset per tile pair, kept only with enough agreeing pairs."""
    frame = pd.DataFrame({"a": a, "b": b, "dx": offsets[:, 0], "dy": offsets[:, 1]})
    frame = frame[frame["a"] != frame["b"]]
    grouped = frame.groupby(["a", "b"])
    med = grouped[["dx", "dy"]].transform("median")
    frame["dev"] = np.hypot(frame["dx"] - med["dx"], frame["dy"] - med["dy"])
    out = (
        frame.groupby(["a", "b"])
        .agg(
            dx=("dx", "median"),
            dy=("dy", "median"),
            n=("dx", "size"),
            mad=("dev", "median"),
        )
        .reset_index()
    )
    out = out[
        (out["n"] >= TILE_SHIFT_MIN_PAIRS) & (out["mad"] <= TILE_SHIFT_MAX_MAD_UM)
    ]
    return out.assign(kind=kind)


def _solve_tile_shifts(obs, n_tiles):
    """Sparse least squares for shift_a - shift_b = offset (b minus a), weak prior toward zero."""
    m = len(obs)
    weight = np.sqrt(np.minimum(obs["n"].to_numpy(float), 100.0))
    rows = np.concatenate([np.arange(m), np.arange(m), m + np.arange(n_tiles)])
    cols = np.concatenate(
        [obs["a"].to_numpy(), obs["b"].to_numpy(), np.arange(n_tiles)]
    )
    vals = np.concatenate([weight, -weight, np.full(n_tiles, TILE_SHIFT_PRIOR_WEIGHT)])
    design = coo_matrix((vals, (rows, cols)), shape=(m + n_tiles, n_tiles)).tocsr()
    out = np.zeros((n_tiles, 2))
    for dim, column in enumerate(("dx", "dy")):
        target = np.concatenate([obs[column].to_numpy() * weight, np.zeros(n_tiles)])
        out[:, dim] = lsqr(design, target, atol=1e-8, btol=1e-8)[0]
    return out


def _point_shifts(table, placed):
    """Expand a per-tile shift table to one (dx, dy) row per cell."""
    return table[["dx", "dy"]].to_numpy(float)[
        placed["tile_table"].index.get_indexer(placed["tiles"])
    ]


def _cells_per_tile(placed):
    """Median number of cells per tile, counting tiles without cells as zero."""
    counts = (
        pd.Series(placed["tiles"])
        .value_counts()
        .reindex(placed["tile_table"].index, fill_value=0)
    )
    return float(counts.median())


def _place_cells(info, metadata, dimensions, orientation, pixel_size):
    """Attach stage position, oriented centred offset and overlap owner to each cell."""
    meta = metadata.drop_duplicates("tile").copy()
    meta["tile"] = meta["tile"].astype("int64")
    meta = meta.set_index("tile").sort_index()
    cells = info.assign(tile=info["tile"].astype("int64"))
    cells = cells[cells["tile"].isin(meta.index)].reset_index(drop=True)
    tiles = cells["tile"].to_numpy(dtype=np.int64)
    flipud, fliplr, rot90 = orientation
    i, j, (height, width) = orient_local(
        cells["i"].to_numpy(float),
        cells["j"].to_numpy(float),
        dimensions,
        flipud,
        fliplr,
        rot90,
    )
    half = np.array([(width - 1) / 2, (height - 1) / 2])
    u = np.column_stack([j, i]) - half
    tile_xy = meta[["x_pos", "y_pos"]].to_numpy(dtype=float)
    stage = tile_xy[meta.index.get_indexer(tiles)]
    owner = _overlap_owner(
        stage + u * pixel_size, tile_xy, meta.index.to_numpy(), half * pixel_size, tiles
    )
    return {
        "cells": cells,
        "tiles": tiles,
        "stage": stage,
        "u": u,
        "pixel_size": pixel_size,
        "half": half,
        "owner": owner,
        "owned": owner == tiles,
        "centrality": np.max(np.abs(u) / half, axis=1),
        "area": cells["area"].to_numpy(float) if "area" in cells.columns else None,
        "tile_table": meta[["x_pos", "y_pos"]].astype(float),
        "tile_width_um": float(max(width, height) * pixel_size),
    }


def _nominal_xy(placed):
    """Positions in um from stage and nominal pixel size."""
    return placed["stage"] + placed["u"] * placed["pixel_size"]


def _overlap_owner(xy, tile_xy, tile_ids, half_extent, own_tile):
    """Return, per point, the tile whose centre is nearest among tiles containing it."""
    k = min(9, len(tile_xy))
    dist, idx = cKDTree(tile_xy).query(xy, k=k)
    dist, idx = dist.reshape(len(xy), k), idx.reshape(len(xy), k)
    inside = np.all(np.abs(xy[:, None, :] - tile_xy[idx]) <= half_extent, axis=2)
    inside |= tile_ids[idx] == own_tile[:, None]
    dist = np.where(inside, dist, np.inf)
    return tile_ids[idx[np.arange(len(xy)), np.argmin(dist, axis=1)]]


def _apply_model(placed, model, name):
    """Place cells in the shared frame with the fitted camera model."""
    u = placed["u"]
    xy = placed["stage"] + u @ model[name].T
    xy = (
        xy
        + _radial_terms(u, placed["half"], placed["pixel_size"]) @ model["radial"][name]
    )
    return xy + model["t"] if name == "phenotype" else xy


def _plausible(model, ph, sbs):
    """Whether a fitted camera keeps each modality's scale near its pixel size, unflipped."""
    for name, placed in (("phenotype", ph), ("sbs", sbs)):
        matrix = model[name] / placed["pixel_size"]
        det = np.linalg.det(matrix)
        if not (CAMERA_SCALE_RANGE[0] ** 2 <= det <= CAMERA_SCALE_RANGE[1] ** 2):
            return False
    return True


def _radial_terms(u, half, pixel_size):
    """(n, 2, 2) radial basis: per cell and axis, p u r^2 and p u r^4."""
    r2 = np.sum((u / max(half)) ** 2, axis=1)
    return pixel_size * np.stack([u * r2[:, None], u * (r2**2)[:, None]], axis=2)


def _solve_camera(ph, ph_index, sbs, sbs_index, seams):
    """Least-squares solve of both cameras, radial terms and translation.

    Unknowns: C_ph (4), C_sbs (4), t (2), k_ph (2), k_sbs (2). Cross-modality rows equate the
    two placements of a nearest-neighbour pair; seam rows equate two copies of one cell.
    """
    blocks, targets = [], []

    def rows(placed, index, sign, c_col, k_col):
        u = placed["u"][index]
        radial = _radial_terms(u, placed["half"], placed["pixel_size"])
        out = []
        for dim in range(2):
            block = np.zeros((len(index), 14))
            block[:, c_col + 2 * dim : c_col + 2 * dim + 2] = sign * u
            block[:, k_col : k_col + 2] = sign * radial[:, dim, :]
            out.append(block)
        return out

    cross_ph = rows(ph, ph_index, 1.0, 0, 10)
    cross_sbs = rows(sbs, sbs_index, -1.0, 4, 12)
    delta = sbs["stage"][sbs_index] - ph["stage"][ph_index]
    for dim in range(2):
        block = cross_ph[dim] + cross_sbs[dim]
        block[:, 8 + dim] = 1.0
        blocks.append(block)
        targets.append(delta[:, dim])
    for (name, (a, b)), c_col, k_col in zip(seams.items(), (0, 4), (10, 12)):
        placed = ph if name == "phenotype" else sbs
        rows_a = rows(placed, a, 1.0, c_col, k_col)
        rows_b = rows(placed, b, -1.0, c_col, k_col)
        delta = placed["stage"][b] - placed["stage"][a]
        for dim in range(2):
            blocks.append(rows_a[dim] + rows_b[dim])
            targets.append(delta[:, dim])
    sol = np.linalg.lstsq(np.vstack(blocks), np.concatenate(targets), rcond=None)[0]
    return {
        "phenotype": sol[:4].reshape(2, 2),
        "sbs": sol[4:8].reshape(2, 2),
        "t": sol[8:10],
        "radial": {"phenotype": sol[10:12], "sbs": sol[12:14]},
    }


def _seam_pairs(placed, points, radius):
    """Pairs (a, b) of records of one cell seen by two tiles: a not owned, b in the owner tile."""
    seam = np.flatnonzero(~placed["owned"])
    if len(seam) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
    dist, idx = cKDTree(points).query(points[seam], k=4, distance_upper_bound=radius)
    found = np.isfinite(dist) & (
        placed["tiles"][np.minimum(idx, len(points) - 1)]
        == placed["owner"][seam][:, None]
    )
    has = found.any(axis=1)
    first = np.argmax(found, axis=1)
    a, b = seam[has], idx[np.flatnonzero(has), first[has]]
    if placed["area"] is not None:
        # a copy cut by its tile border has a smaller area and a centroid pulled inward
        area_a, area_b = placed["area"][a], placed["area"][b]
        whole = np.minimum(area_a, area_b) >= SEAM_MIN_AREA_RATIO * np.maximum(
            area_a, area_b
        )
        a, b = a[whole], b[whole]
    return a, b


def _scale_rotation(matrix):
    """Return the mean scale and rotation angle (degrees) of a 2x2 matrix."""
    u, s, vt = np.linalg.svd(matrix)
    rot = u @ vt
    return float(s.mean()), float(np.degrees(np.arctan2(rot[1, 0], rot[0, 0])))


def _pixel_size(metadata, fallback, name):
    """Return the median um/pixel of the tiles from metadata, else the configured fallback."""
    if "pixel_size_x" in metadata.columns and metadata["pixel_size_x"].notna().any():
        return float(metadata["pixel_size_x"].dropna().median())
    if fallback is not None:
        return float(fallback)
    raise ValueError(
        f"No {name} pixel size in metadata; set merge.{name}_pixel_size in the config"
    )


def _orientation_qc(info, metadata, dimensions, pixel_size, orientation, name):
    """Seam agreement of the configured orientation against the best of all eight."""
    scores = {
        o: seam_agreement(info, metadata, dimensions, pixel_size, o)
        for o in ORIENTATIONS
    }
    configured = _canonical(orientation)
    best = max(scores, key=lambda o: -1 if np.isnan(scores[o]) else scores[o])
    conf_score = scores.get(configured, np.nan)
    warn = bool(
        np.isfinite(conf_score)
        and np.isfinite(scores[best])
        and conf_score < SEAM_WARN_RATIO * scores[best]
    )
    return {
        f"{name}_seam_agreement": conf_score,
        f"{name}_seam_agreement_best": scores[best],
        f"{name}_best_orientation": "flipud={},fliplr={},rot90={}".format(*best),
        f"{name}_orientation_warning": warn,
    }


def _canonical(orientation):
    """Express any flip/rotation combination as one of the eight listed orientations."""
    probe_i, probe_j = np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0])
    target = orient_local(probe_i, probe_j, (3, 3), *orientation)[:2]
    for candidate in ORIENTATIONS:
        if all(
            np.allclose(a, b)
            for a, b in zip(
                orient_local(probe_i, probe_j, (3, 3), *candidate)[:2], target
            )
        ):
            return candidate
    return orientation


def _merge_frame(ph_cells, sbs_cells, pairs, sbs_pixel_size):
    """Build the merge table with the `fast_merge` columns."""
    i0 = pairs[:, 0].astype(int)
    i1 = pairs[:, 1].astype(int)
    ph = ph_cells.iloc[i0].reset_index(drop=True)
    sbs = sbs_cells.iloc[i1].reset_index(drop=True)
    merged = pd.DataFrame(
        {
            "plate": ph["plate"].values,
            "well": ph["well"].values,
            "tile": ph["tile"].values,
            "cell_0": ph["cell"].values,
            "i_0": ph["i"].values,
            "j_0": ph["j"].values,
            "site": sbs["tile"].values,
            "cell_1": sbs["cell"].values,
            "i_1": sbs["i"].values,
            "j_1": sbs["j"].values,
            "distance": pairs[:, 2] / sbs_pixel_size,
        }
    )
    return merged[MERGE_COLUMNS]


def _match_qc(merged, ph_cells, sbs_cells):
    """Match counts and rates for the QC table."""
    return {
        "n_matches": len(merged),
        "phenotype_match_rate": len(merged) / max(len(ph_cells), 1),
        "sbs_match_rate": len(merged) / max(len(sbs_cells), 1),
        "median_distance_px": float(merged["distance"].median())
        if len(merged)
        else np.nan,
    }


def _status(qc):
    """Summarise the QC row into one status string."""
    if qc.get("camera_fit_failed"):
        return "camera_fit_failed"
    if qc["phenotype_match_rate"] < LOW_MATCH_RATE:
        return "low_match_rate"
    if qc["phenotype_orientation_warning"] or qc["sbs_orientation_warning"]:
        return "orientation_warning"
    return "ok"
