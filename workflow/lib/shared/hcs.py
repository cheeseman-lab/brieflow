"""HCS (High Content Screening) OME-NGFF metadata-only fusion utilities.

After Snakemake jobs write zarr stores directly into the HCS plate hierarchy
(e.g., aligned_{plate}.zarr/{row}/{col}/{tile}/zarr.json), these functions
discover what was written and layer the OME-NGFF metadata on top.

No symlinks or data copies — only zarr.json metadata files.
"""

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from iohub.ngff import open_ome_zarr
from iohub.ngff.display import channel_display_settings
from iohub.ngff.models import OMEROMeta, RDefsMeta, TransformationMeta

from lib.shared.image_io import DEFAULT_CHANNEL_COLORS


# ---------------------------------------------------------------------------
# High-level API (used by Snakemake scripts)
# ---------------------------------------------------------------------------


def write_hcs_metadata(plate_zarr_path, channels_metadata=None):
    """Write OME-NGFF HCS metadata for an existing plate zarr directory.

    Args:
        plate_zarr_path: Path to the plate zarr directory (e.g., sbs/1.zarr).
        channels_metadata: Optional list[dict] to embed at plate root under
        attributes["channels_metadata"].
    """
    plate_path = Path(plate_zarr_path)
    if not plate_path.exists():
        raise FileNotFoundError(f"Plate zarr directory not found: {plate_path}")

    structure = discover_plate_structure(plate_path)
    if not structure:
        print(f"  No fields found in {plate_path}. Skipping metadata.")
        return

    wells_by_row_col = {}
    for row, col, _tile in structure:
        wells_by_row_col[(row, col)] = True  # deduplicate

    # Compute field_count = max number of tiles in any well
    fields_by_well = {}
    for row, col, tile in structure:
        fields_by_well.setdefault((row, col), []).append(tile)
    field_count = max(len(tiles) for tiles in fields_by_well.values())

    _write_plate_metadata(
        plate_path,
        wells_by_row_col,
        channels_metadata=channels_metadata,
        field_count=field_count,
    )

    # Write row-level group metadata
    for row in sorted(set(rc[0] for rc in wells_by_row_col)):
        _write_zarr_v3_group_metadata(plate_path / row)

    # Write well-level and field-level metadata
    for (row, col), tiles in sorted(fields_by_well.items()):
        well_dir = plate_path / row / col
        field_indices = sorted(tiles)
        _write_well_metadata(well_dir, field_indices)

        # Write labels group metadata for fields that have label stores
        for tile in field_indices:
            field_dir = well_dir / str(tile)
            _maybe_write_labels_metadata(field_dir)


def discover_plate_structure(plate_zarr_path):
    """Discover (row, col, tile) by locating tile-level zarr.json files.

    Pass 1 (Option D): plate.zarr/{row}/{col}/{tile}/zarr.json
    Pass 2 (fallback): any deeper zarr.json (e.g. preprocess cycle level)
                       plate.zarr/{row}/{col}/{tile}/.../zarr.json
    """
    plate_path = Path(plate_zarr_path)
    results = []
    seen = set()

    # ---- Pass 1: strict Option D tile marker ----
    for zjson in sorted(plate_path.rglob("zarr.json")):
        rel = zjson.relative_to(plate_path)
        parts = rel.parts
        if len(parts) != 4 or parts[-1] != "zarr.json":
            continue

        row, col, tile = parts[0], parts[1], parts[2]
        if not re.match(r"^[A-Za-z]+$", str(row)):
            continue
        if not str(col).isdigit() or not str(tile).isdigit():
            continue

        key = (row, col, tile)
        if key not in seen:
            seen.add(key)
            results.append(key)

    if results:
        return results

    # ---- Pass 2: fallback for preprocess-style extra nesting ----
    for zjson in sorted(plate_path.rglob("zarr.json")):
        rel = zjson.relative_to(plate_path)
        parts = rel.parts
        if len(parts) < 4 or parts[-1] != "zarr.json":
            continue

        row, col, tile = parts[0], parts[1], parts[2]
        if not re.match(r"^[A-Za-z]+$", str(row)):
            continue
        if not str(col).isdigit() or not str(tile).isdigit():
            continue

        key = (row, col, tile)
        if key not in seen:
            seen.add(key)
            results.append(key)

    return results


def patch_store_metadata_with_iohub(
    store_path: Path,
    preprocess_root: Path,
    config_channel_names: list[str] | None = None,
    modality_config: dict | None = None,
    channels_metadata: list[dict] | None = None,
):
    """Open a plate zarr store in r+ mode and enrich tile-level metadata.

    Patches:
      - Pixel scale (x/y) from combined_metadata.parquet
      - Spatial axis units → micrometer
      - Channel names (rename from c0/c1/… to real names)
      - OMERO rendering defaults (colors, rdefs, contrast limits)
      - image-label version on nested label stores
      - Label coordinate scales (same pixel size as parent image)
      - segmentation_metadata on label stores
    """
    store_type = _parse_store_type(store_path)
    plate = _parse_plate_from_store_name(store_path)
    modality = _infer_modality_from_store_path(store_path)
    pixel_map = _load_pixel_size_map(preprocess_root, modality, plate)

    print(f"[patch] store={store_path}  type={store_type}")
    print(f"[patch] pixel_map entries={len(pixel_map)}")

    ds = open_ome_zarr(str(store_path), layout="hcs", mode="r+", version="0.5")
    pos_list = list(ds.positions())
    print(f"[patch] positions={len(pos_list)}")

    for pos_path, pos in pos_list:
        parts = pos_path.split("/")
        if len(parts) != 3:
            print(f"[patch] skipping unexpected pos_path={pos_path}")
            continue

        row, col, tile = parts
        key = (str(row), str(col), str(tile))
        fallback = (str(row), str(col), "*")

        # --- pixel scale (per-dataset, with downsampling factors) ---
        px_x = px_y = None
        if key in pixel_map:
            px_x, px_y = pixel_map[key]
        elif fallback in pixel_map:
            px_x, px_y = pixel_map[fallback]

        if px_x is not None:
            _set_per_dataset_scales(pos, px_x, px_y)

        # --- channel names ---
        resolved = _resolve_channel_names_for_store(
            pos, config_channel_names, store_type
        )
        _rename_channels(pos, resolved)

        # --- OMERO rendering defaults (colors, rdefs, contrast limits) ---
        _build_and_set_omero(pos, resolved, channels_metadata=channels_metadata)

        # --- axis units ---
        _ensure_axes_units_micrometer(pos)

        pos.dump_meta()

    ds.dump_meta()
    ds.close()

    # --- Re-inject downsamplingMethod (iohub dump_meta strips it) ---
    for zj in sorted(store_path.rglob("*/zarr.json")):
        # Only patch image-group level (has multiscales but not inside labels/)
        if "labels" in zj.parts:
            continue
        try:
            meta = json.loads(zj.read_text())
        except Exception:
            continue
        attrs = meta.get("attributes", {})
        ome = attrs.get("ome", {})
        ms_list = ome.get("multiscales", [])
        if ms_list and "downsamplingMethod" not in ms_list[0]:
            ms_list[0]["downsamplingMethod"] = "gaussian"
            zj.write_text(json.dumps(meta, indent=2))

    # --- Direct JSON patching for label stores (iohub doesn't expose these) ---
    _patch_label_versions(store_path)
    _patch_label_axis_units(store_path)
    _patch_label_scales(store_path, pixel_map)

    if modality_config:
        _patch_segmentation_metadata(store_path, modality_config, channels_metadata)


def compute_and_inject_omero_windows(
    plate_paths: list[Path],
    low_pct: float = 1.0,
    high_pct: float = 99.0,
) -> int:
    """Compute screen-wide per-channel display windows + statistics and inject into every tile zarr.json.

    Accumulates uint16 histograms across every image tile in every plate
    store, derives ``window.start`` / ``window.end`` at the given
    percentile cutpoints, computes per-channel mean / std / median for ML
    dataloader normalization, and writes both into every image-level
    zarr.json's ``omero.channels[i]``.

    Args:
        plate_paths: List of plate.zarr roots to histogram and patch.
        low_pct: Lower percentile for window.start (default 1.0).
        high_pct: Upper percentile for window.end (default 99.0).

    Returns:
        Total number of zarr.json files updated across all plates. Zero if
        no image tiles were found.
    """
    if not plate_paths:
        return 0

    print(
        f"\nComputing screen-wide OMERO windows "
        f"[{low_pct}, {high_pct}]th pct over {len(plate_paths)} store(s)..."
    )
    histograms: dict[int, np.ndarray] = {}
    for plate_path in plate_paths:
        _accumulate_channel_histograms(plate_path, histograms)
    if not histograms:
        print("No image tiles found for window computation; skipping.")
        return 0

    windows = _windows_from_histograms(histograms, low_pct, high_pct)
    stats = _stats_from_histograms(histograms)
    for ch in sorted(windows.keys()):
        start, end = windows[ch]
        s = stats[ch]
        print(
            f"  channel {ch}: window=({start:.1f}, {end:.1f})  "
            f"mean={s['mean']:.1f} std={s['std']:.1f} median={s['median']:.1f}"
        )

    patched_total = 0
    for plate_path in plate_paths:
        n = _inject_omero_windows(plate_path, windows, stats=stats)
        patched_total += n
        print(f"  {plate_path.name}: patched {n} zarr.json")
    print(
        f"OMERO windows + statistics injected into {patched_total} tile zarr.json files."
    )
    return patched_total


# ---------------------------------------------------------------------------
# Helpers — well parsing
# ---------------------------------------------------------------------------


def _normalize_channels_metadata(channels_metadata):
    """Normalize channels_metadata for root zarr.json."""
    if not channels_metadata:
        return []

    out = []
    for ch in channels_metadata:
        if not isinstance(ch, dict):
            continue

        entry = dict(ch)

        # Ensure required fields exist
        name = (entry.get("name") or "").strip()
        if not name:
            raise ValueError("channels_metadata entry is missing a non-empty 'name'")
        entry["name"] = name

        entry.setdefault("description", "")
        entry.setdefault("channel_type", "fluorescence")

        # Keep biological_annotation only if it has real (non-empty) values, and keep ONLY the keys the user actually filled in
        bio = entry.get("biological_annotation", None)

        if isinstance(bio, dict):
            cleaned = {}
            for k in ("biological_target", "marker", "marker_type", "full_label"):
                v = bio.get(k, None)
                if v is None:
                    continue
                v = str(v).strip()
                if v:  # keep only non empty values
                    cleaned[k] = v

            if cleaned:
                entry["biological_annotation"] = cleaned
            else:
                entry.pop("biological_annotation", None)
        else:
            entry.pop("biological_annotation", None)

        out.append(entry)

    for i, entry in enumerate(out):
        entry.setdefault("index", i)

    return out


def _split_well(well_str):
    """Split a well identifier like 'A1' into (row, col) -> ('A', '1')."""
    match = re.match(r"^([A-Za-z]+)(\d+)$", str(well_str))
    if not match:
        raise ValueError(f"Cannot parse well identifier: '{well_str}'")
    return match.group(1), match.group(2)


# ---------------------------------------------------------------------------
# Helpers — label detection
# ---------------------------------------------------------------------------


def _is_label_store(zarr_path):
    """Check if a zarr store is a label image by reading its zarr.json."""
    zarr_json = Path(zarr_path) / "zarr.json"
    if not zarr_json.exists():
        return False
    with open(zarr_json) as f:
        meta = json.load(f)
    attrs = meta.get("attributes", {})
    # image-label may be under ome namespace (v3) or top-level
    return "image-label" in attrs or "image-label" in attrs.get("ome", {})


def _maybe_write_labels_metadata(field_dir):
    """If a field has label stores, write the labels/ group metadata."""
    labels_dir = field_dir / "labels"
    if not labels_dir.is_dir():
        return

    label_stores = []
    for child in sorted(labels_dir.iterdir()):
        if child.is_dir() and child.suffix == ".zarr":
            label_stores.append(child.stem)
        elif child.is_dir() and (child / "zarr.json").exists():
            # Label stored as a named group (not .zarr suffix)
            if _is_label_store(child):
                label_stores.append(child.name)

    if label_stores:
        _write_labels_group_metadata(labels_dir, label_stores)


# ---------------------------------------------------------------------------
# Helpers — metadata writers
# ---------------------------------------------------------------------------


def _write_zarr_v3_group_metadata(path):
    """Write a minimal zarr v3 group zarr.json file."""
    metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
    }
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    with open(path / "zarr.json", "w") as f:
        json.dump(metadata, f, indent=2)


def _write_plate_metadata(
    plate_zarr_path, wells_by_row_col, channels_metadata=None, field_count=1
):
    """Write HCS plate-level zarr.json with OME-NGFF plate metadata."""
    plate_path = Path(plate_zarr_path)
    plate_path.mkdir(parents=True, exist_ok=True)

    rows = sorted(set(rc[0] for rc in wells_by_row_col.keys()))
    cols = sorted(set(rc[1] for rc in wells_by_row_col.keys()), key=lambda x: int(x))

    plate_name = plate_path.stem  # e.g. "aligned_1"

    plate_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "plate": {
                    "version": "0.5",
                    "name": plate_name,
                    "field_count": field_count,
                    "acquisitions": [{"id": 0}],
                    "columns": [{"name": c} for c in cols],
                    "rows": [{"name": r} for r in rows],
                    "wells": [
                        {
                            "path": f"{rc[0]}/{rc[1]}",
                            "rowIndex": rows.index(rc[0]),
                            "columnIndex": cols.index(rc[1]),
                        }
                        for rc in sorted(wells_by_row_col.keys())
                    ],
                },
            }
        },
    }

    norm = _normalize_channels_metadata(channels_metadata)
    if norm:  # embed into zarr.json if metadata is not empty
        plate_metadata["attributes"]["channels_metadata"] = norm

    with open(plate_path / "zarr.json", "w") as f:
        json.dump(plate_metadata, f, indent=2)


def _write_well_metadata(well_path, field_indices):
    """Write HCS well-level zarr.json listing fields (tiles)."""
    well_dir = Path(well_path)
    well_dir.mkdir(parents=True, exist_ok=True)

    well_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "well": {
                    "version": "0.5",
                    "images": [
                        {"path": str(idx), "acquisition": 0} for idx in field_indices
                    ],
                },
            }
        },
    }

    with open(well_dir / "zarr.json", "w") as f:
        json.dump(well_metadata, f, indent=2)


def _write_labels_group_metadata(labels_dir, label_names):
    """Write labels group zarr.json listing available labels."""
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "labels": label_names,
            }
        },
    }
    with open(labels_dir / "zarr.json", "w") as f:
        json.dump(metadata, f, indent=2)


# ---------------------------------------------------------------------------
# Helpers — store name parsing
# ---------------------------------------------------------------------------


def _parse_plate_from_store_name(store_path: Path) -> str:
    """aligned_1.zarr -> "1", illumination_corrected_12.zarr -> "12"."""
    m = re.search(r"_(\d+)\.zarr$", store_path.name)
    if not m:
        raise ValueError(f"Could not parse plate from store name: {store_path.name}")
    return m.group(1)


def _parse_store_type(store_path: Path) -> str:
    """aligned_1.zarr -> "aligned", peaks_1.zarr -> "peaks"."""
    name = store_path.name  # e.g. "illumination_corrected_1.zarr"
    m = re.match(r"^(.+?)_\d+\.zarr$", name)
    if not m:
        return name.replace(".zarr", "")
    return m.group(1)


def _infer_modality_from_store_path(store_path: Path) -> str:
    """Return 'sbs' or 'phenotype' based on which appears in the path parts."""
    parts = store_path.parts
    if "sbs" in parts:
        return "sbs"
    if "phenotype" in parts:
        return "phenotype"
    raise ValueError(f"Could not infer modality from path: {store_path}")


# ---------------------------------------------------------------------------
# Helpers — pixel size loading
# ---------------------------------------------------------------------------


def _load_pixel_size_map(
    preprocess_root: Path, modality: str, plate: str
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Return (row, col, tile) -> (px_x, px_y) in micrometers.

    Reads preprocess/metadata/{modality}/{plate}/{row}/{col}/combined_metadata.parquet.
    """
    meta_root = preprocess_root / "metadata" / modality / plate
    pixel_map: dict[tuple[str, str, str], tuple[float, float]] = {}

    if not meta_root.exists():
        print(f"[iohub patch] metadata root missing: {meta_root}")
        return pixel_map

    for fp in meta_root.rglob("combined_metadata.parquet"):
        rel = fp.relative_to(meta_root)
        if len(rel.parts) < 3:
            continue
        row, col = str(rel.parts[0]), str(rel.parts[1])
        df = pd.read_parquet(fp)

        if "tile" not in df.columns:
            if (
                "pixel_size_x" in df.columns
                and "pixel_size_y" in df.columns
                and len(df) > 0
            ):
                pixel_map[(row, col, "*")] = (
                    float(df["pixel_size_x"].iloc[0]),
                    float(df["pixel_size_y"].iloc[0]),
                )
            continue

        for _, r in df.iterrows():
            tile = str(r["tile"])
            if pd.isna(r.get("pixel_size_x")) or pd.isna(r.get("pixel_size_y")):
                continue
            pixel_map[(row, col, tile)] = (
                float(r["pixel_size_x"]),
                float(r["pixel_size_y"]),
            )

    return pixel_map


# ---------------------------------------------------------------------------
# Helpers — per-store channel name resolution
# ---------------------------------------------------------------------------

# Single-channel stores whose channel name is the store type itself.
_SINGLE_CHANNEL_STORES = {"peaks", "standard_deviation"}


def _resolve_channel_names_for_store(
    pos, config_channel_names: list[str] | None, store_type: str
) -> list[str]:
    """Determine the real channel names for a position in a given store.

    Rules:
      - Single-channel stores (peaks, standard_deviation) → [store_type]
      - Multi-channel stores: if config_channel_names count matches the
        position's channel count, use config names; otherwise keep current.
    """
    try:
        n_channels = len(list(pos.channel_names))
    except Exception:
        return []

    if store_type in _SINGLE_CHANNEL_STORES:
        return [store_type]

    if config_channel_names and len(config_channel_names) == n_channels:
        return list(config_channel_names)

    # Fallback: keep whatever names the store already has.
    # iohub may return ints when OMERO metadata is missing — always stringify.
    try:
        return [str(n) for n in pos.channel_names]
    except Exception:
        return [f"c{i}" for i in range(n_channels)]


# ---------------------------------------------------------------------------
# Helpers — iohub-based metadata patching
# ---------------------------------------------------------------------------


def _get_axis_index_ci(pos, name: str) -> int:
    """Case-insensitive axis index lookup.

    Handles stores written with either uppercase (TCZYX) or lowercase (tczyx)
    axis names.
    """
    try:
        return pos.get_axis_index(name.upper())
    except (ValueError, KeyError):
        return pos.get_axis_index(name.lower())


def _ensure_axes_units_micrometer(pos) -> None:
    """Set unit='micrometer' on spatial axes via the iohub metadata model.

    Must modify pos.metadata (not pos.zattrs) so changes survive dump_meta().
    """
    if not pos.metadata.multiscales:
        return
    for ax in pos.metadata.multiscales[0].axes:
        if ax.name.lower() in ("x", "y", "z"):
            ax.unit = "micrometer"


def _set_per_dataset_scales(pos, px_x: float, px_y: float) -> None:
    """Set absolute physical pixel scale at each pyramid level.

    For each dataset (pyramid level), the scale is the base pixel size
    multiplied by the downsampling factor inferred from the array shapes.
    Clears any FOV-level coordinateTransformations so that the per-dataset
    transforms are the single source of truth.
    """
    ms = pos.metadata.multiscales[0]
    n_axes = len(ms.axes)
    y_idx = _get_axis_index_ci(pos, "y")
    x_idx = _get_axis_index_ci(pos, "x")

    base_shape = pos["0"].shape

    for ds_meta in ms.datasets:
        level_shape = pos[ds_meta.path].shape
        factor_y = (
            base_shape[y_idx] / level_shape[y_idx] if level_shape[y_idx] > 0 else 1.0
        )
        factor_x = (
            base_shape[x_idx] / level_shape[x_idx] if level_shape[x_idx] > 0 else 1.0
        )

        scale = [1.0] * n_axes
        scale[y_idx] = px_y * factor_y
        scale[x_idx] = px_x * factor_x

        ds_meta.coordinate_transformations = [
            TransformationMeta(type="scale", scale=scale)
        ]

    # Clear FOV-level transform — pixel size now lives per-dataset.
    ms.coordinate_transformations = None


def _rename_channels(pos, target_names: list[str]) -> None:
    """Rename channels to *target_names* if they differ from current names."""
    try:
        current = list(pos.channel_names)
    except Exception:
        return

    n = min(len(current), len(target_names))
    for i in range(n):
        old, new = current[i], target_names[i]
        if old != new:
            try:
                pos.rename_channel(old, new)
            except Exception:
                pass


def _build_and_set_omero(
    pos,
    resolved_names: list[str],
    channels_metadata: list[dict] | None = None,
) -> None:
    """Create OMERO rendering metadata (channels + rdefs) on the position.

    Uses iohub's ``channel_display_settings`` for color assignment, with
    overrides from ``channels_metadata[].color`` config and a default
    palette fallback for unrecognized channel names. All channels are
    set to active.
    """
    if not resolved_names:
        return

    # Build color lookup from config if available
    config_colors = {}
    if channels_metadata:
        for ch in channels_metadata:
            if isinstance(ch, dict) and "color" in ch:
                config_colors[ch.get("name", "")] = ch["color"]

    channels = []
    for i, name in enumerate(resolved_names):
        ch_meta = channel_display_settings(name, clim=None, first_chan=True)

        # Override color: config → iohub (if not white) → default palette
        if name in config_colors:
            ch_meta.color = config_colors[name]
        elif ch_meta.color == "FFFFFF":
            ch_meta.color = DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)]

        # All channels active
        ch_meta.active = True

        channels.append(ch_meta)

    pos.metadata.omero = OMEROMeta(
        version="0.5",
        channels=channels,
        rdefs=RDefsMeta(default_t=0, default_z=0),
    )


# ---------------------------------------------------------------------------
# Helpers — label metadata patching
# ---------------------------------------------------------------------------


def _patch_label_axis_units(store_path: Path) -> None:
    """Set units on all axes in label store multiscales metadata.

    Always runs regardless of pixel size availability — axis units and
    pixel scales are independent concerns.
    """
    _AXIS_UNITS = {
        "X": "micrometer",
        "Y": "micrometer",
        "Z": "micrometer",
        "T": "second",
    }
    for zj in sorted(store_path.rglob("labels/*/zarr.json")):
        try:
            meta = json.loads(zj.read_text())
        except Exception:
            continue

        attrs = meta.get("attributes", meta)
        ome = attrs.get("ome", {})
        ms_list = ome.get("multiscales", attrs.get("multiscales", []))
        if not ms_list:
            continue

        changed = False
        for ax in ms_list[0].get("axes", []):
            name = ax.get("name", "").upper()
            if name in _AXIS_UNITS and ax.get("unit") != _AXIS_UNITS[name]:
                ax["unit"] = _AXIS_UNITS[name]
                changed = True

        if changed:
            zj.write_text(json.dumps(meta, indent=2))
            print(f"[patch] label axis units set: {zj.parent.name}")


def _patch_label_scales(
    store_path: Path,
    pixel_map: dict[tuple[str, str, str], tuple[float, float]],
) -> None:
    """Apply coordinate scales to label stores.

    Labels share the same physical pixel size as their parent image.
    Uses direct JSON patching (iohub doesn't iterate labels).
    """
    for zj in sorted(store_path.rglob("labels/*/zarr.json")):
        label_dir = zj.parent
        field_dir = label_dir.parent.parent  # …/labels/{name} → field dir

        # Derive row/col/tile from field path
        rel = field_dir.relative_to(store_path)
        parts = rel.parts
        if len(parts) != 3:
            continue
        row, col, tile = parts

        key = (str(row), str(col), str(tile))
        fallback = (str(row), str(col), "*")
        px_x = px_y = None
        if key in pixel_map:
            px_x, px_y = pixel_map[key]
        elif fallback in pixel_map:
            px_x, px_y = pixel_map[fallback]
        if px_x is None:
            continue

        try:
            meta = json.loads(zj.read_text())
        except Exception:
            continue

        # Navigate to multiscales — may be under ome namespace (v3) or top-level
        attrs = meta.get("attributes", meta)
        ome = attrs.get("ome", {})
        ms_list = ome.get("multiscales", attrs.get("multiscales", []))
        if not ms_list:
            continue
        ms = ms_list[0]

        # Find Y and X axis indices (case-insensitive)
        axes = ms.get("axes", [])
        y_idx = x_idx = None
        for i, ax in enumerate(axes):
            name = ax.get("name", "").upper()
            if name == "Y":
                y_idx = i
            elif name == "X":
                x_idx = i
        if y_idx is None or x_idx is None:
            continue

        datasets = ms.get("datasets", [])
        if not datasets:
            continue

        # Read base (level 0) array shape
        base_arr_zj = label_dir / datasets[0].get("path", "0") / "zarr.json"
        base_shape = None
        if base_arr_zj.exists():
            try:
                base_shape = json.loads(base_arr_zj.read_text()).get("shape")
            except Exception:
                pass
        if not base_shape:
            continue

        # Set per-dataset coordinate transformations
        for ds in datasets:
            level_arr_zj = label_dir / ds.get("path", "0") / "zarr.json"
            level_shape = base_shape
            if level_arr_zj.exists():
                try:
                    level_shape = json.loads(level_arr_zj.read_text()).get(
                        "shape", base_shape
                    )
                except Exception:
                    pass

            scale = [1.0] * len(axes)
            fy = (
                base_shape[y_idx] / level_shape[y_idx]
                if level_shape[y_idx] > 0
                else 1.0
            )
            fx = (
                base_shape[x_idx] / level_shape[x_idx]
                if level_shape[x_idx] > 0
                else 1.0
            )
            scale[y_idx] = px_y * fy
            scale[x_idx] = px_x * fx
            ds["coordinateTransformations"] = [{"type": "scale", "scale": scale}]

        zj.write_text(json.dumps(meta, indent=2))
        print(f"[patch] label scales set: {label_dir.name} in {'/'.join(parts)}")


def _patch_label_versions(store_path: Path) -> None:
    """Walk label stores inside a plate zarr and set image-label.version.

    Label stores live at <store>.zarr/{row}/{col}/{tile}/labels/{name}.zarr/.
    Their zarr.json should have ``"image-label": {"version": "0.5"}``.
    This uses direct JSON patching (iohub doesn't iterate labels).
    """
    for zj in sorted(store_path.rglob("labels/*/zarr.json")):
        try:
            meta = json.loads(zj.read_text())
        except Exception:
            continue
        # image-label may be under ome namespace (zarr v3) or top-level attrs
        attrs = meta.get("attributes", meta)
        ome = attrs.get("ome", {})
        il = ome.get("image-label", attrs.get("image-label"))
        if il is None:
            continue
        if il.get("version") != "0.5":
            il["version"] = "0.5"
            zj.write_text(json.dumps(meta, indent=2))
            print(f"[patch] label version set: {zj.parent.name}")


# ---------------------------------------------------------------------------
# Helpers — segmentation metadata
# ---------------------------------------------------------------------------

# Maps label directory stem to (annotation_type, config key for diameter,
# config key for source channel index).
_LABEL_ANNOTATION_MAP = {
    "nuclei": {
        "annotation_type": "nucleus",
        "diameter_key": "nuclei_diameter",
        "source_channel_key": "dapi_index",
        "flow_threshold_key": "nuclei_flow_threshold",
        "cellprob_threshold_key": "nuclei_cellprob_threshold",
    },
    "cells": {
        "annotation_type": "cell",
        "diameter_key": "cell_diameter",
        "source_channel_key": "cyto_index",
        "flow_threshold_key": "cell_flow_threshold",
        "cellprob_threshold_key": "cell_cellprob_threshold",
    },
    "identified_cytoplasms": {
        "annotation_type": "cytoplasm",
        "diameter_key": None,
        "source_channel_key": "cyto_index",
        "flow_threshold_key": None,
        "cellprob_threshold_key": None,
    },
}


def _build_segmentation_meta_for_label(
    label_stem: str,
    modality_config: dict,
    channels_metadata: list[dict] | None,
) -> dict | None:
    """Build a ``segmentation_metadata`` dict for one label store.

    Returns *None* when the label name is unrecognised or there is
    insufficient config to build the block.
    """
    info = _LABEL_ANNOTATION_MAP.get(label_stem)
    if info is None:
        return None

    # Build method string as "method.model" (e.g. "cellpose.cyto3")
    seg_method_base = modality_config.get("segmentation_method", "cellpose")
    seg_model = modality_config.get("cellpose_model") or modality_config.get(
        "stardist_model"
    )
    seg_method = f"{seg_method_base}.{seg_model}" if seg_model else seg_method_base
    source_idx = modality_config.get(info["source_channel_key"])
    if source_idx is None:
        source_idx = 0

    # Biological annotation from channels_metadata
    bio = {}
    if channels_metadata:
        for ch in channels_metadata:
            if isinstance(ch, dict) and ch.get("index") == source_idx:
                ch_bio = ch.get("biological_annotation", {})
                if isinstance(ch_bio, dict):
                    for k in (
                        "biological_target",
                        "marker",
                        "marker_type",
                        "full_label",
                    ):
                        v = ch_bio.get(k)
                        if v:
                            bio[k] = v
                break

    # Segmentation parameters (only non-None values)
    params = {}
    has_flow = has_cellprob = False
    for pkey in ("diameter_key", "flow_threshold_key", "cellprob_threshold_key"):
        cfg_key = info.get(pkey)
        if cfg_key and modality_config.get(cfg_key) is not None:
            params[cfg_key] = modality_config[cfg_key]
            if "flow" in pkey:
                has_flow = True
            if "cellprob" in pkey:
                has_cellprob = True

    # For phenotype modality, fall back to shared flow_threshold / cellprob_threshold
    if not has_flow and "flow_threshold" in modality_config:
        ft = modality_config["flow_threshold"]
        if ft is not None:
            params["flow_threshold"] = ft
    if not has_cellprob and "cellprob_threshold" in modality_config:
        ct = modality_config["cellprob_threshold"]
        if ct is not None:
            params["cellprob_threshold"] = ct

    return {
        "label_name": label_stem,
        "annotation_type": info["annotation_type"],
        "is_ome_label": True,
        "source_channel": {"index": source_idx},
        "biological_annotation": bio,
        "segmentation": {
            "method": seg_method,
            "stitching": "none",
            "parameters": params,
        },
        "description": f"{info['annotation_type']} segmentation via {seg_method}",
    }


def _patch_segmentation_metadata(
    store_path: Path,
    modality_config: dict,
    channels_metadata: list[dict] | None,
) -> None:
    """Inject ``segmentation_metadata`` into each label store's zarr.json.

    Metadata is placed at ``attributes.segmentation_metadata`` alongside the
    existing ``attributes.ome`` block.
    """
    for zj in sorted(store_path.rglob("labels/*/zarr.json")):
        label_dir = zj.parent
        # label_dir.name is e.g. "nuclei.zarr" → stem is "nuclei"
        label_stem = label_dir.name.replace(".zarr", "")

        seg_meta = _build_segmentation_meta_for_label(
            label_stem, modality_config, channels_metadata
        )
        if seg_meta is None:
            continue

        try:
            meta = json.loads(zj.read_text())
        except Exception:
            continue

        # Count labeled objects from the full-resolution array
        n_cells = None
        arr_zj = label_dir / "0" / "zarr.json"
        if arr_zj.exists():
            try:
                import zarr

                arr = zarr.open(str(label_dir / "0"), mode="r")
                n_cells = int(len(np.unique(arr[:])) - 1)  # exclude background (0)
            except Exception:
                pass
        if n_cells is not None:
            seg_meta["statistics"] = {"n_cells": n_cells}

        attrs = meta.setdefault("attributes", {})
        attrs["segmentation_metadata"] = seg_meta

        zj.write_text(json.dumps(meta, indent=2))
        print(f"[patch] segmentation_metadata set: {label_stem} in {label_dir}")


# ---------------------------------------------------------------------------
# Helpers — OMERO display window
# ---------------------------------------------------------------------------


def _accumulate_channel_histograms(
    store_path: Path,
    histograms: dict[int, np.ndarray],
    n_bins: int = 65536,
) -> None:
    """Add per-channel uint16 histograms from every tile array in ``store_path``.

    Mutates ``histograms`` in place: channel_idx -> int64 array of length n_bins.
    Skips label arrays. Loads multiscale level 0 only.
    """
    import zarr

    for zj in sorted(store_path.rglob("zarr.json")):
        rel = zj.relative_to(store_path)
        if "labels" in rel.parts:
            continue
        # Tile-level zarr.json sits at plate.zarr/{row}/{col}/{tile}/zarr.json
        if len(rel.parts) != 4:
            continue
        try:
            meta = json.loads(zj.read_text())
        except Exception:
            continue
        if (
            meta.get("attributes", {}).get("ome", {}).get("omero", {}).get("channels")
            is None
        ):
            continue
        try:
            grp = zarr.open(str(zj.parent), mode="r")
            arr = grp["0"][:]
        except Exception as exc:
            print(f"[window] could not read {zj.parent}: {exc}")
            continue

        # Locate channel axis; image arrays are (T,C,Z,Y,X), (C,Z,Y,X), or (C,Y,X)
        if arr.ndim == 5:
            channel_axis = 1
        elif arr.ndim in (3, 4):
            channel_axis = 0
        else:
            continue

        if arr.dtype != np.uint16:
            arr = np.clip(arr, 0, n_bins - 1).astype(np.uint16)

        n_channels = arr.shape[channel_axis]
        for ch in range(n_channels):
            sl = [slice(None)] * arr.ndim
            sl[channel_axis] = ch
            counts = np.bincount(arr[tuple(sl)].ravel(), minlength=n_bins)
            if ch in histograms:
                histograms[ch] += counts
            else:
                histograms[ch] = counts


def _windows_from_histograms(
    histograms: dict[int, np.ndarray],
    low_pct: float,
    high_pct: float,
) -> dict[int, tuple[float, float]]:
    """Convert per-channel histograms into (start, end) at the given percentiles."""
    windows: dict[int, tuple[float, float]] = {}
    for ch, hist in histograms.items():
        cdf = np.cumsum(hist).astype(np.float64)
        total = cdf[-1]
        if total == 0:
            windows[ch] = (0.0, float(len(hist) - 1))
            continue
        cdf /= total
        start = float(np.searchsorted(cdf, low_pct / 100.0))
        end = float(np.searchsorted(cdf, high_pct / 100.0))
        if end <= start:
            end = start + 1.0
        windows[ch] = (start, end)
    return windows


def _stats_from_histograms(
    histograms: dict[int, np.ndarray],
) -> dict[int, dict[str, float]]:
    """Per-channel mean / std / median computed from the merged histograms.

    Provided alongside ``window`` so dataloaders have dataset-wide
    normalization constants without re-scanning the data. Standard ML
    portability pattern: store stats once, apply at __getitem__ time.
    """
    stats: dict[int, dict[str, float]] = {}
    for ch, hist in histograms.items():
        total = float(hist.sum())
        if total == 0:
            stats[ch] = {"mean": 0.0, "std": 0.0, "median": 0.0}
            continue
        bins = np.arange(len(hist), dtype=np.float64)
        mean = float((bins * hist).sum() / total)
        var = float((hist * (bins - mean) ** 2).sum() / total)
        std = float(np.sqrt(var))

        cdf = np.cumsum(hist).astype(np.float64) / total
        median = float(np.searchsorted(cdf, 0.5))

        stats[ch] = {"mean": mean, "std": std, "median": median}
    return stats


def _inject_omero_windows(
    store_path: Path,
    windows: dict[int, tuple[float, float]],
    stats: dict[int, dict[str, float]] | None = None,
) -> int:
    """Inject ``window`` (and optional ``statistics``) into every image-level zarr.json.

    Returns the number of zarr.json files updated.
    """
    patched = 0
    for zj in sorted(store_path.rglob("zarr.json")):
        if "labels" in zj.relative_to(store_path).parts:
            continue
        try:
            meta = json.loads(zj.read_text())
        except Exception:
            continue
        channels = (
            meta.get("attributes", {}).get("ome", {}).get("omero", {}).get("channels")
        )
        if not channels:
            continue
        changed = False
        for idx, ch in enumerate(channels):
            if idx not in windows:
                continue
            start, end = windows[idx]
            ch["window"] = {
                "start": start,
                "end": end,
                "min": 0.0,
                "max": 65535.0,
            }
            if stats is not None and idx in stats:
                ch["statistics"] = stats[idx]
            changed = True
        if changed:
            zj.write_text(json.dumps(meta, indent=2))
            patched += 1
    return patched
