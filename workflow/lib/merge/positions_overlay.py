"""Image readouts that show whether a positions merge placed the tiles correctly.

The merge itself reads no images. These readouts read a bounded sample afterwards: tile-overlap
strips of neighbouring tiles of one modality, phenotype DAPI mapped into SBS tiles, and a
downsampled nuclei mosaic of the whole well. Every image is placed with the fitted
`tile_pixel_to_um`, so aligned structures come out white in the magenta/green overlays
and a placement error shows every nucleus twice.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.spatial import cKDTree
from skimage.registration import phase_cross_correlation

from lib.merge.merge_utils import align_metadata
from lib.merge.positions_merge import (
    positions_merge,
    tile_pixel_to_um,
    um_to_tile_pixel,
)
from lib.shared.file_utils import get_image_output_path, split_well
from lib.shared.alignment_overlay import (
    colored_fraction,
    magenta_green_overlay,
    plot_overlay_grid,
)

N_TILE_OVERLAPS = 20
N_PHENOTYPE_IN_SBS = 4
PHENOTYPE_IN_SBS_INNER_FRACTION = 0.7
MOSAIC_MAX_PX = 3000
OVERLAP_WARN_UM = 1.5
MIN_CONTRAST = 1.0
READOUT_COLUMNS = [
    "kind",
    "tile_a",
    "tile_b",
    "residual_dy_px",
    "residual_dx_px",
    "residual_px",
    "residual_um",
    "colored_fraction",
]


def positions_merge_well(
    phenotype_info,
    sbs_info,
    phenotype_metadata,
    sbs_metadata,
    plate,
    well,
    phenotype_dimensions,
    sbs_dimensions,
    threshold=2,
    flipud=False,
    fliplr=False,
    rot90=0,
    phenotype_pixel_size=None,
    sbs_pixel_size=None,
    alignment=None,
    templates=None,
    dapi_index=None,
):
    """Run the positions merge for one well with its QC row and image readouts.

    This is what the `positions_merge` rule runs and what the merge notebook previews, so
    both use the same metadata alignment, pixel-size fallbacks, readouts and status.

    Args:
        phenotype_info (pandas.DataFrame): Phenotype cells (`tile`, `cell`, `i`, `j`).
        sbs_info (pandas.DataFrame): SBS cells, same columns.
        phenotype_metadata (pandas.DataFrame): One row per phenotype tile.
        sbs_metadata (pandas.DataFrame): One row per SBS tile.
        plate (str): Plate.
        well (str): Well.
        phenotype_dimensions (tuple): Phenotype tile (height, width) in pixels.
        sbs_dimensions (tuple): SBS tile (height, width) in pixels.
        threshold (float): Maximum match distance in SBS pixels.
        flipud (bool): Tile rows run against stage y.
        fliplr (bool): Tile columns run against stage x.
        rot90 (int): Counterclockwise quarter turns between tile and stage axes.
        phenotype_pixel_size (float | None): Fallback um/pixel when metadata has none.
        sbs_pixel_size (float | None): Fallback um/pixel when metadata has none.
        alignment (dict, optional): `metadata_align`, `flip_x`, `flip_y`, `rotate_90`; when any
            is set the stage frames are aligned with `align_metadata`, as `fast_alignment` does.
        templates (dict, optional): `{"labels": {modality: template}, "images": {...}}` per-tile
            path templates (see `image_path_templates`); None skips the image readouts.
        dapi_index (dict, optional): Per modality, DAPI channel index. Defaults to 0.

    Returns:
        tuple: (merge DataFrame, one-row QC DataFrame, placement (see `positions_merge`;
            None when there are too few cells), image readout DataFrame, dict of figures
            "tile_overlaps", "phenotype_in_sbs", "mosaic").
    """
    alignment = alignment or {}
    flips = {key: alignment.get(key) for key in ("flip_x", "flip_y", "rotate_90")}
    if alignment.get("metadata_align") or any(flips.values()):
        phenotype_metadata, sbs_metadata, _ = align_metadata(
            phenotype_metadata, sbs_metadata, x_col="x_pos", y_col="y_pos", **flips
        )
    merged, qc, placement = positions_merge(
        phenotype_info,
        sbs_info,
        phenotype_metadata,
        sbs_metadata,
        phenotype_dimensions=phenotype_dimensions,
        sbs_dimensions=sbs_dimensions,
        threshold=threshold,
        flipud=flipud,
        fliplr=fliplr,
        rot90=rot90,
        phenotype_pixel_size=phenotype_pixel_size,
        sbs_pixel_size=sbs_pixel_size,
    )
    records = pd.DataFrame(columns=READOUT_COLUMNS)
    figures = {}
    if placement is not None and templates:
        start = time.time()
        label_paths, image_paths = overlay_image_paths(
            templates,
            {name: placement[name]["tiles"].index for name in ("phenotype", "sbs")},
            plate,
            well,
        )
        records, figures = positions_image_qc(
            placement,
            label_paths,
            image_paths,
            {name: (dapi_index or {}).get(name) or 0 for name in ("phenotype", "sbs")},
            {
                "phenotype": phenotype_info["tile"].value_counts(),
                "sbs": sbs_info["tile"].value_counts(),
            },
        )
        for key, value in summarize_image_qc(records).items():
            qc[key] = value
        qc["image_qc_seconds"] = round(time.time() - start, 1)
        if qc["image_qc_warning"].iloc[0] and qc["status"].iloc[0] == "ok":
            qc["status"] = "image_qc_warning"
    qc.insert(0, "well", well)
    qc.insert(0, "plate", plate)
    return merged, qc, placement, records, figures


def positions_image_qc(
    placement, label_paths, image_paths, dapi_index, cell_counts=None
):
    """Tile-overlap, phenotype-in-SBS and mosaic readouts for one well.

    Args:
        placement (dict): Placement returned by `positions_merge`.
        label_paths (dict): Per modality, {tile: path of the nuclei label image}.
        image_paths (dict): Per modality, {tile: path of the aligned image}.
        dapi_index (dict): Per modality, channel index of DAPI in the aligned image.
        cell_counts (dict, optional): Per modality, pandas.Series of cells per tile, used
            to include the sparsest tiles in the tile-overlap sample.

    Returns:
        tuple: (DataFrame with one row per readout, dict of figures "tile_overlaps",
            "phenotype_in_sbs" and "mosaic", each None when nothing could be drawn).
    """
    candidates = overlay_candidates(placement, cell_counts)
    overlap_records, overlap_fig = plot_tile_overlaps(
        placement, candidates["tile_overlaps"], label_paths, image_paths, dapi_index
    )
    in_sbs_records, in_sbs_fig = plot_phenotype_in_sbs(
        placement, candidates["phenotype_in_sbs"], label_paths, image_paths, dapi_index
    )
    records = pd.concat([overlap_records, in_sbs_records], ignore_index=True)
    return records, {
        "tile_overlaps": overlap_fig,
        "phenotype_in_sbs": in_sbs_fig,
        "mosaic": plot_mosaics(placement, label_paths),
    }


def overlay_candidates(placement, cell_counts=None):
    """Tile pairs to check by eye: tile overlaps of each modality and phenotype-in-SBS pairs.

    Args:
        placement (dict): Placement returned by `positions_merge`.
        cell_counts (dict, optional): Per modality, pandas.Series of cells per tile, used
            to include the sparsest tiles in the tile-overlap sample.

    Returns:
        dict: `"tile_overlaps"`: list of (modality, tile_a, tile_b), spread over the well;
            `"phenotype_in_sbs"`: list of (phenotype tile, SBS tile).
    """
    counts = cell_counts or {}
    overlaps = [
        (name, tile_a, tile_b)
        for name in ("phenotype", "sbs")
        for tile_a, tile_b in sample_tile_overlaps(
            placement[name], N_TILE_OVERLAPS, counts.get(name)
        )
    ]
    in_sbs = sample_phenotype_in_sbs_pairs(
        placement, N_PHENOTYPE_IN_SBS, counts.get("phenotype")
    )
    return {"tile_overlaps": overlaps, "phenotype_in_sbs": in_sbs}


def summarize_image_qc(records):
    """Median residual and colored fraction per readout kind, for the merge QC row.

    Warns when a median residual exceeds `OVERLAP_WARN_UM` micrometres, so the threshold means
    the same at every magnification.

    Args:
        records (pandas.DataFrame): Rows from `positions_image_qc`.

    Returns:
        dict: Summary values keyed by readout kind.
    """
    out = {}
    for kind, group in records.groupby("kind"):
        out[f"{kind}_n"] = int(group["residual_px"].notna().sum())
        out[f"{kind}_residual_median_px"] = float(group["residual_px"].median())
        out[f"{kind}_residual_max_px"] = float(group["residual_px"].max())
        out[f"{kind}_residual_median_um"] = float(group["residual_um"].median())
        out[f"{kind}_colored_median"] = float(group["colored_fraction"].median())
    out["image_qc_warning"] = bool(
        any(
            out.get(f"{kind}_residual_median_um", 0) > OVERLAP_WARN_UM
            for kind in (
                "tile_overlap_phenotype",
                "tile_overlap_sbs",
                "phenotype_in_sbs",
            )
        )
    )
    return out


def sample_tile_overlaps(placement, n, cell_counts=None):
    """Pick neighbouring tile pairs spread over the well, including corners and sparse tiles.

    Args:
        placement (dict): One modality's placement.
        n (int): Number of pairs.
        cell_counts (pandas.Series, optional): Cells per tile.

    Returns:
        list[tuple]: (tile_a, tile_b) pairs.
    """
    tiles = placement["tiles"]
    xy = tiles[["x_pos", "y_pos"]].to_numpy(float)
    width = max(placement["dimensions"]) * placement["pixel_size"]
    pairs = cKDTree(xy).query_pairs(1.2 * width, output_type="ndarray")
    if len(pairs) == 0:
        return []
    mid = xy[pairs].mean(axis=1)
    chosen = [int(np.argmin(mid.sum(axis=1)))]
    if cell_counts is not None:
        counts = cell_counts.reindex(tiles.index, fill_value=0).to_numpy()
        sparse = counts[pairs].min(axis=1) + counts[pairs].max(axis=1) * 1e-3
        chosen += [int(i) for i in np.argsort(sparse)[:2]]
    dist = np.full(len(pairs), np.inf)
    for index in chosen:
        dist = np.minimum(dist, np.hypot(*(mid - mid[index]).T))
    while len(set(chosen)) < min(n, len(pairs)):
        index = int(np.argmax(dist))
        chosen.append(index)
        dist = np.minimum(dist, np.hypot(*(mid - mid[index]).T))
    ids = tiles.index.to_numpy()
    return [
        (int(ids[pairs[i, 0]]), int(ids[pairs[i, 1]])) for i in dict.fromkeys(chosen)
    ]


def sample_phenotype_in_sbs_pairs(placement, n, cell_counts=None):
    """Pick phenotype tiles spread over the well and the SBS tile nearest each one.

    Keeps to the inner part of the well, away from the well edge, and prefers phenotype tiles
    with at least the median cell count, so the overlay has nuclei.

    Args:
        placement (dict): Placement returned by `positions_merge`.
        n (int): Number of pairs.
        cell_counts (pandas.Series, optional): Phenotype cells per tile.

    Returns:
        list[tuple]: (phenotype tile, SBS tile) pairs.
    """
    ph = placement["phenotype"]["tiles"]
    sbs = placement["sbs"]["tiles"]
    if cell_counts is not None:
        counts = cell_counts.reindex(ph.index, fill_value=0)
        ph = ph[counts >= counts.median()]
    xy = ph[["x_pos", "y_pos"]].to_numpy(float)
    center = xy.mean(axis=0)
    radius = np.hypot(*(xy - center).T)
    inner = radius <= PHENOTYPE_IN_SBS_INNER_FRACTION * radius.max()
    if inner.sum() >= n:
        ph, xy = ph[inner], xy[inner]
    if len(xy) == 0:
        return []
    chosen = [int(np.argmin(np.hypot(*(xy - center).T)))]
    dist = np.hypot(*(xy - xy[chosen[0]]).T)
    while len(chosen) < min(n, len(xy)):
        index = int(np.argmax(dist))
        chosen.append(index)
        dist = np.minimum(dist, np.hypot(*(xy - xy[index]).T))
    sbs_tree = cKDTree(sbs[["x_pos", "y_pos"]].to_numpy(float))
    _, nearest = sbs_tree.query(xy[chosen])
    return [(int(ph.index[i]), int(sbs.index[j])) for i, j in zip(chosen, nearest)]


def tile_overlap_overlay(
    placement, image_paths, label_paths, dapi_index, tile_a, tile_b
):
    """Overlay the overlap of two tiles of one modality, tile B resampled into tile A.

    Uses the DAPI channel of the aligned images, or nuclei labels when images are not on disk.

    Args:
        placement (dict): One modality's placement.
        image_paths (dict): {tile: aligned image path}.
        label_paths (dict): {tile: nuclei label path}.
        dapi_index (int): DAPI channel index.
        tile_a (int): Reference tile (magenta).
        tile_b (int): Moving tile (green).

    Returns:
        tuple: (record dict with residual and colored fraction, overlay panel or None).
    """
    record = {
        "tile_a": tile_a,
        "tile_b": tile_b,
        "residual_dy_px": np.nan,
        "residual_dx_px": np.nan,
        "residual_px": np.nan,
        "colored_fraction": np.nan,
    }
    a = _read_dapi_or_labels(image_paths, label_paths, dapi_index, tile_a)
    b = _read_dapi_or_labels(image_paths, label_paths, dapi_index, tile_b)
    if a is None or b is None:
        return record, None
    resampled = _resample(b, placement, tile_b, placement, tile_a, a.shape, order=1)
    if resampled is None:
        return record, None
    mov, mask, window = resampled
    ref = a[window]
    record.update(_residual(ref, mov, mask))
    record["colored_fraction"] = colored_fraction(magenta_green_overlay(ref, mov), mask)
    title = f"tiles {tile_a}|{tile_b}: {record['residual_px']:.1f} px"
    return record, (ref, mov, title, mask)


def phenotype_in_sbs_overlay(
    placement, image_paths, label_paths, dapi_index, ph_tile, sbs_tile
):
    """Overlay phenotype DAPI mapped into SBS pixel space on SBS DAPI.

    Uses nuclei labels for both modalities when either aligned image is not on disk, so the
    two panels always show the same kind of image.

    Args:
        placement (dict): Placement returned by `positions_merge`.
        image_paths (dict): Per modality, {tile: aligned image path}.
        label_paths (dict): Per modality, {tile: nuclei label path}.
        dapi_index (dict): Per modality, DAPI channel index.
        ph_tile (int): Phenotype tile.
        sbs_tile (int): SBS tile.

    Returns:
        tuple: (record dict with residual, overlay panel or None).
    """
    record = {
        "tile_a": sbs_tile,
        "tile_b": ph_tile,
        "residual_dy_px": np.nan,
        "residual_dx_px": np.nan,
        "residual_px": np.nan,
        "colored_fraction": np.nan,
    }
    tiles = {"phenotype": ph_tile, "sbs": sbs_tile}
    if not all(tile in image_paths.get(name, {}) for name, tile in tiles.items()):
        image_paths = {}
    planes = {}
    for name, tile in tiles.items():
        planes[name] = _read_dapi_or_labels(
            image_paths.get(name, {}),
            label_paths.get(name, {}),
            dapi_index.get(name, 0),
            tile,
        )
        if planes[name] is None:
            return record, None
    ph_plane = planes["phenotype"]
    ratio = placement["sbs"]["pixel_size"] / placement["phenotype"]["pixel_size"]
    if ratio > 1.5:
        ph_plane = ndimage.gaussian_filter(ph_plane, sigma=ratio / 2)
    resampled = _resample(
        ph_plane,
        placement["phenotype"],
        ph_tile,
        placement["sbs"],
        sbs_tile,
        planes["sbs"].shape,
        order=1,
    )
    if resampled is None:
        return record, None
    mov, mask, window = resampled
    ref = planes["sbs"][window]
    record.update(_residual(ref, mov, mask))
    title = f"PH {ph_tile} in SBS {sbs_tile}: {record['residual_px']:.1f} px"
    return record, (ref, mov, title, mask)


def plot_tile_overlaps(placement, pairs, label_paths, image_paths, dapi_index):
    """Overlay the overlap strips of neighbouring tiles of one modality.

    Args:
        placement (dict): Placement returned by `positions_merge`.
        pairs (list[tuple]): (modality, tile_a, tile_b) from `overlay_candidates`.
        label_paths (dict): Per modality, {tile: nuclei label path}.
        image_paths (dict): Per modality, {tile: aligned image path}.
        dapi_index (dict): Per modality, DAPI channel index.

    Returns:
        tuple: (DataFrame with one row per pair, figure or None).
    """
    records, panels = [], []
    for name, tile_a, tile_b in pairs:
        record, panel = tile_overlap_overlay(
            placement[name],
            image_paths.get(name, {}),
            label_paths.get(name, {}),
            dapi_index.get(name, 0),
            tile_a,
            tile_b,
        )
        record["residual_um"] = record["residual_px"] * placement[name]["pixel_size"]
        records.append({"kind": f"tile_overlap_{name}", **record})
        if panel is not None:
            reference, moving, title, mask = panel
            panels.append((reference, moving, f"{name} {title}", mask))
    figure = plot_overlay_grid(
        panels,
        ncols=6,
        panel_size=2.6,
        colored=True,
        suptitle="Tile overlaps: tile A magenta, tile B green (white = aligned)",
    )
    return pd.DataFrame(records, columns=READOUT_COLUMNS), figure


def plot_phenotype_in_sbs(placement, pairs, label_paths, image_paths, dapi_index):
    """Overlay phenotype DAPI mapped into SBS tiles on SBS DAPI.

    Args:
        placement (dict): Placement returned by `positions_merge`.
        pairs (list[tuple]): (phenotype tile, SBS tile) from `overlay_candidates`.
        label_paths (dict): Per modality, {tile: nuclei label path}.
        image_paths (dict): Per modality, {tile: aligned image path}.
        dapi_index (dict): Per modality, DAPI channel index.

    Returns:
        tuple: (DataFrame with one row per pair, figure or None).
    """
    records, panels = [], []
    for ph_tile, sbs_tile in pairs:
        record, panel = phenotype_in_sbs_overlay(
            placement, image_paths, label_paths, dapi_index, ph_tile, sbs_tile
        )
        record["residual_um"] = record["residual_px"] * placement["sbs"]["pixel_size"]
        records.append({"kind": "phenotype_in_sbs", **record})
        if panel is not None:
            panels.append(panel)
    figure = plot_overlay_grid(
        panels,
        ncols=4,
        panel_size=4,
        colored=False,
        suptitle="Phenotype DAPI mapped into SBS tiles: SBS magenta, phenotype green "
        "(white = aligned)",
    )
    return pd.DataFrame(records, columns=READOUT_COLUMNS), figure


def plot_mosaics(placement, label_paths):
    """Downsampled nuclei mosaics of the well, one panel per modality.

    Tiles alternate magenta and green in a checkerboard, so tile overlaps show white where the two
    tiles agree and doubled nuclei where they do not.

    Args:
        placement (dict): Placement returned by `positions_merge`.
        label_paths (dict): Per modality, {tile: nuclei label path}.

    Returns:
        matplotlib.figure.Figure: The figure, or None if no labels were readable.
    """
    import matplotlib.pyplot as plt

    mosaics = {
        name: well_mosaic(placement[name], label_paths[name])
        for name in ("phenotype", "sbs")
    }
    mosaics = {name: m for name, m in mosaics.items() if m is not None}
    if not mosaics:
        return None
    fig, axes = plt.subplots(
        1,
        len(mosaics),
        figsize=(9 * len(mosaics), 9),
        squeeze=False,
        layout="constrained",
    )
    for ax, (name, (image, um_per_px)) in zip(axes.ravel(), mosaics.items()):
        ax.imshow(image, interpolation="nearest")
        ax.set_title(
            f"{name} nuclei, {um_per_px:.1f} um per pixel (checkerboard tiles)"
        )
        ax.axis("off")
    return fig


def well_mosaic(placement, paths):
    """Render a downsampled checkerboard nuclei mosaic of one modality.

    Args:
        placement (dict): One modality's placement.
        paths (dict): {tile: nuclei label path}.

    Returns:
        tuple: (RGB image, um per mosaic pixel), or None when no tile is readable.
    """
    tiles = placement["tiles"]
    readable = [t for t in tiles.index if t in paths and Path(paths[t]).exists()]
    if not readable:
        return None
    width_um = max(placement["dimensions"]) * placement["pixel_size"]
    xy = tiles[["x_pos", "y_pos"]].to_numpy(float)
    lo = xy.min(axis=0) - width_um
    hi = xy.max(axis=0) + width_um
    um_per_px = max(float((hi - lo).max()) / MOSAIC_MAX_PX, placement["pixel_size"])
    step = max(1, int(um_per_px // placement["pixel_size"]))
    shape = (int((hi - lo)[1] / um_per_px) + 1, int((hi - lo)[0] / um_per_px) + 1)
    canvas = np.zeros(shape + (2,), dtype=np.float32)
    parity = _checkerboard(xy, width_um)
    for tile in readable:
        mask = read_plane(paths[tile], step=step) > 0
        ii, jj = np.nonzero(mask)
        xy_um = tile_pixel_to_um(
            placement, tile, np.column_stack([ii * step, jj * step])
        )
        col = ((xy_um[:, 0] - lo[0]) / um_per_px).astype(int)
        row = ((xy_um[:, 1] - lo[1]) / um_per_px).astype(int)
        keep = (row >= 0) & (row < shape[0]) & (col >= 0) & (col < shape[1])
        channel = parity[tiles.index.get_loc(tile)]
        np.maximum.at(canvas[..., channel], (row[keep], col[keep]), 1.0)
    rgb = np.stack([canvas[..., 0], canvas[..., 1], canvas[..., 0]], axis=-1)
    return rgb, um_per_px


def image_path_templates(root_fp, image_format):
    """Per-tile nuclei-label and aligned-image path templates of a screen.

    Args:
        root_fp (str | Path): The screen's `all.root_fp`.
        image_format (str): `"tiff"` or `"zarr"`.

    Returns:
        dict: `{"labels": {modality: template}, "images": {modality: template}}`.
    """
    tile = {"plate": "{plate}", "well": "{well}", "tile": "{tile}"}
    return {
        kind: {
            name: str(
                Path(root_fp)
                / name
                / get_image_output_path(tile, info, image_format, subdirectory=sub)
            )
            for name in ("phenotype", "sbs")
        }
        for kind, info, sub in (
            ("labels", "nuclei", "labels"),
            ("images", "aligned", None),
        )
    }


def overlay_image_paths(templates, tiles, plate, well):
    """Nuclei-label and aligned-image paths on disk for the given tiles of each modality.

    Args:
        templates (dict): `{"labels": {modality: template}, "images": {...}}`, see
            `image_path_templates`.
        tiles (dict): Per modality, the tile ids to look up.
        plate (str): Plate.
        well (str): Well.

    Returns:
        tuple: (label paths, image paths), each `{modality: {tile: path}}`.
    """
    found = {
        kind: {
            name: tile_image_paths(
                templates[kind][name], tiles.get(name, []), plate, well
            )
            for name in ("phenotype", "sbs")
        }
        for kind in ("labels", "images")
    }
    return found["labels"], found["images"]


def tile_image_paths(template, tiles, plate, well):
    """Format a per-tile output path template for every tile that exists on disk.

    Args:
        template (str): Path template with `{plate}`, `{well}` or `{row}`/`{col}`, and `{tile}`.
        tiles (Iterable[int]): Tile ids.
        plate (str): Plate.
        well (str): Well.

    Returns:
        dict: {tile: path}.
    """
    row, col = split_well(str(well))
    paths = {}
    for tile in tiles:
        path = str(template).format(plate=plate, well=well, row=row, col=col, tile=tile)
        if Path(path).exists():
            paths[tile] = path
    return paths


def save_figures(figures, paths):
    """Save the readout figures, writing a placeholder for any that could not be drawn.

    Args:
        figures (dict): Figures keyed "tile_overlaps", "phenotype_in_sbs", "mosaic" (missing or None allowed).
        paths (Sequence[str]): Output PNG paths, in that order.
    """
    import matplotlib.pyplot as plt

    for key, path in zip(("tile_overlaps", "phenotype_in_sbs", "mosaic"), paths):
        fig = figures.get(key)
        if fig is None:
            fig = plt.figure(figsize=(4, 1))
            fig.text(0.5, 0.5, "image readout not available", ha="center", va="center")
        fig.savefig(path, dpi=100)
        plt.close(fig)


def read_plane(path, channel=0, cycle=0, step=1):
    """Read one 2D plane of a tile lazily from OME-Zarr or TIFF.

    Args:
        path (str | Path): Zarr image or label group, or a TIFF file.
        channel (int): Channel index.
        cycle (int): Cycle index for SBS images that hold several cycles.
        step (int): Keep every `step`-th row and column.

    Returns:
        numpy.ndarray: 2D array.
    """
    import zarr

    p = Path(path)
    if p.name == "zarr.json":
        p = p.parent
    if p.suffix.lower() in {".tif", ".tiff"}:
        import tifffile

        array = zarr.open(tifffile.imread(str(p), aszarr=True), mode="r")
    else:
        group = zarr.open_group(str(p), mode="r")
        array = group["0"]
    index = _plane_index(array.shape, channel, cycle)
    return np.asarray(array[index + (slice(None, None, step), slice(None, None, step))])


def _plane_index(shape, channel, cycle):
    """Leading-axis index that selects one 2D plane, mirroring `image_io.read_image`."""
    lead = list(shape[:-2])
    index = []
    if len(lead) == 3:
        index.append(0)
        lead = lead[1:]
    if len(lead) == 2:
        if lead[1] == 1:
            return tuple(index + [min(channel, lead[0] - 1), 0])
        if lead[0] == 1:
            return tuple(index + [0, min(channel, lead[1] - 1)])
        return tuple(index + [min(cycle, lead[0] - 1), min(channel, lead[1] - 1)])
    if len(lead) == 1:
        return tuple(index + [min(channel, lead[0] - 1)])
    return tuple(index)


def _resample(image, place_from, tile_from, place_to, tile_to, shape, order):
    """Resample one tile's image into another tile's pixel frame over their overlap.

    Returns:
        tuple: (resampled crop, valid mask, window slices into the target frame), or None
            when the overlap is too small.
    """
    height, width = image.shape
    corners = np.array(
        [[0, 0], [0, width - 1], [height - 1, 0], [height - 1, width - 1]], float
    )
    target = um_to_tile_pixel(
        place_to, tile_to, tile_pixel_to_um(place_from, tile_from, corners)
    )
    lo = np.clip(np.floor(target.min(axis=0)).astype(int), 0, np.array(shape))
    hi = np.clip(np.ceil(target.max(axis=0)).astype(int) + 1, 0, np.array(shape))
    if np.any(hi - lo < 16):
        return None
    ii, jj = np.mgrid[lo[0] : hi[0], lo[1] : hi[1]]
    grid = np.column_stack([ii.ravel(), jj.ravel()]).astype(float)
    source = um_to_tile_pixel(
        place_from, tile_from, tile_pixel_to_um(place_to, tile_to, grid)
    )
    valid = (
        (source[:, 0] >= 0)
        & (source[:, 0] <= height - 1)
        & (source[:, 1] >= 0)
        & (source[:, 1] <= width - 1)
    ).reshape(ii.shape)
    moved = ndimage.map_coordinates(
        image.astype(np.float32), source.T, order=order, mode="constant", cval=0.0
    ).reshape(ii.shape)
    window = _bounding_window(valid)
    if window is None:
        return None
    full = (
        slice(lo[0] + window[0].start, lo[0] + window[0].stop),
        slice(lo[1] + window[1].start, lo[1] + window[1].stop),
    )
    return moved[window], valid[window], full


def _bounding_window(valid):
    """Slices of the bounding box of the valid region, or None if it is too small."""
    rows = np.flatnonzero(valid.any(axis=1))
    cols = np.flatnonzero(valid.any(axis=0))
    if len(rows) < 16 or len(cols) < 16:
        return None
    return slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1)


def _residual(reference, moving, mask):
    """Masked normalised cross-correlation shift of moving relative to reference, in pixels."""
    if min(_contrast(reference[mask]), _contrast(moving[mask])) < MIN_CONTRAST:
        return {}
    reference, moving = _normalize(reference), _normalize(moving)
    shift = phase_cross_correlation(
        reference, moving, reference_mask=mask, moving_mask=mask
    )[0]
    return {
        "residual_dy_px": float(shift[0]),
        "residual_dx_px": float(shift[1]),
        "residual_px": float(np.hypot(*shift)),
    }


def _read_dapi_or_labels(image_paths, label_paths, dapi_index, tile):
    """DAPI plane of the aligned image, else the binary nuclei labels, else None."""
    if tile in image_paths:
        return read_plane(image_paths[tile], channel=dapi_index).astype(np.float32)
    if tile in label_paths:
        return (read_plane(label_paths[tile]) > 0).astype(np.float32)
    return None


def _contrast(values):
    """(99.5th percentile - median) / median: about 0 for an empty, background-only crop."""
    median = float(np.median(values))
    return (float(np.percentile(values, 99.5)) - median) / max(median, 1e-6)


def _normalize(image):
    """Scale an image to [0, 1] between its 1st and 99.5th percentiles."""
    low, high = np.percentile(image, (1, 99.5))
    return np.clip((image - low) / max(high - low, 1e-6), 0, 1)


def _checkerboard(xy, width_um):
    """0/1 per tile from its grid row and column, so neighbouring tiles alternate."""
    grid = np.round((xy - xy.min(axis=0)) / (0.9 * width_um)).astype(int)
    return (grid.sum(axis=1) % 2).astype(int)
