"""Write HCS plate-level OME-NGFF metadata for plate zarr directories.

This script discovers the structure of plate zarr directories that were
populated directly by Snakemake jobs and writes the necessary zarr.json
metadata files at plate, row, well, and labels levels.  It then uses
iohub to enrich tile-level metadata (pixel sizes, axis units, OMERO
rendering defaults, channel names/colors).
"""

import json
import re
from pathlib import Path

import pandas as pd
from iohub.ngff import open_ome_zarr
from iohub.ngff.display import channel_display_settings
from iohub.ngff.models import OMEROMeta, RDefsMeta, TransformationMeta

from lib.shared.hcs import write_hcs_metadata


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
    """
    Returns (row, col, tile) -> (px_x, px_y) in micrometers.
    Reads preprocess/metadata/{modality}/{plate}/{row}/{col}/combined_metadata.parquet
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
            if "pixel_size_x" in df.columns and "pixel_size_y" in df.columns and len(df) > 0:
                pixel_map[(row, col, "*")] = (
                    float(df["pixel_size_x"].iloc[0]),
                    float(df["pixel_size_y"].iloc[0]),
                )
            continue

        for _, r in df.iterrows():
            tile = str(r["tile"])
            if pd.isna(r.get("pixel_size_x")) or pd.isna(r.get("pixel_size_y")):
                continue
            pixel_map[(row, col, tile)] = (float(r["pixel_size_x"]), float(r["pixel_size_y"]))

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
# Metadata patching functions
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
        factor_y = base_shape[y_idx] / level_shape[y_idx] if level_shape[y_idx] > 0 else 1.0
        factor_x = base_shape[x_idx] / level_shape[x_idx] if level_shape[x_idx] > 0 else 1.0

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


def _build_and_set_omero(pos, resolved_names: list[str]) -> None:
    """Create OMERO rendering metadata (channels + rdefs) on the position.

    Uses iohub's ``channel_display_settings`` to auto-assign colors based
    on channel name keywords (DAPI→blue, GFP→lime, Cy3→yellow, Cy5→orange,
    etc.) and default contrast limits.
    """
    if not resolved_names:
        return

    channels = []
    for i, name in enumerate(resolved_names):
        ch_meta = channel_display_settings(name, clim=None, first_chan=(i == 0))
        channels.append(ch_meta)

    pos.metadata.omero = OMEROMeta(
        version="0.5",
        channels=channels,
        rdefs=RDefsMeta(default_t=0, default_z=0),
    )


def _patch_label_scales(
    store_path: Path,
    pixel_map: dict[tuple[str, str, str], tuple[float, float]],
) -> None:
    """Apply coordinate scales and axis units to label stores.

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
            # Set micrometer unit on spatial axes
            if name in ("X", "Y", "Z"):
                ax["unit"] = "micrometer"
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
            fy = base_shape[y_idx] / level_shape[y_idx] if level_shape[y_idx] > 0 else 1.0
            fx = base_shape[x_idx] / level_shape[x_idx] if level_shape[x_idx] > 0 else 1.0
            scale[y_idx] = px_y * fy
            scale[x_idx] = px_x * fx
            ds["coordinateTransformations"] = [{"type": "scale", "scale": scale}]

        zj.write_text(json.dumps(meta, indent=2))
        print(f"[patch] label scales set: {label_dir.name} in {'/'.join(parts)}")


# ---------------------------------------------------------------------------
# Label metadata — segmentation_metadata + image-label version
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

    seg_method = modality_config.get("segmentation_method", "cellpose")
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
            "version": "",
            "stitching": False,
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

        attrs = meta.setdefault("attributes", {})
        attrs["segmentation_metadata"] = seg_meta

        zj.write_text(json.dumps(meta, indent=2))
        print(f"[patch] segmentation_metadata set: {label_stem} in {label_dir}")


def _patch_label_versions(store_path: Path) -> None:
    """Walk label stores inside a plate zarr and set image-label.version.

    Label stores live at  <store>.zarr/{row}/{col}/{tile}/labels/{name}.zarr/.
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
# Main patching entry point
# ---------------------------------------------------------------------------


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
        resolved = _resolve_channel_names_for_store(pos, config_channel_names, store_type)
        _rename_channels(pos, resolved)

        # --- OMERO rendering defaults (colors, rdefs, contrast limits) ---
        _build_and_set_omero(pos, resolved)

        # --- axis units ---
        _ensure_axes_units_micrometer(pos)

        pos.dump_meta()

    ds.dump_meta()
    ds.close()

    # --- Direct JSON patching for label stores (iohub doesn't expose these) ---
    _patch_label_versions(store_path)
    _patch_label_scales(store_path, pixel_map)

    if modality_config:
        _patch_segmentation_metadata(store_path, modality_config, channels_metadata)


# ===================================================================
# Script entry point (called by Snakemake)
# ===================================================================

plate_zarr_dirs = snakemake.params.plate_zarr_dirs
channels_metadata = getattr(snakemake.params, "channels_metadata", None)
config_channel_names = getattr(snakemake.params, "channel_names", None)

# Preprocess root for pixel-size lookup
root_fp = Path(snakemake.config["all"]["root_fp"])
preprocess_root = root_fp / "preprocess"

# Modality config for segmentation_metadata on label stores.
# The snakemake rule should set params.modality to "sbs" or "phenotype".
modality_name = getattr(snakemake.params, "modality", None)
modality_config = None
if modality_name and modality_name in snakemake.config:
    modality_config = dict(snakemake.config[modality_name])

total = 0
for plate_zarr in plate_zarr_dirs:
    plate_path = Path(plate_zarr)
    if plate_path.exists():
        print(f"Writing HCS metadata for: {plate_path}")
        write_hcs_metadata(plate_path, channels_metadata=channels_metadata)
        total += 1
        # Skip preprocess stores for iohub patching (extra cycle nesting)
        if "preprocess" not in plate_path.parts:
            patch_store_metadata_with_iohub(
                plate_path,
                preprocess_root,
                config_channel_names=config_channel_names,
                modality_config=modality_config,
                channels_metadata=channels_metadata,
            )
    else:
        print(f"Plate zarr not found, skipping: {plate_path}")

if total > 0:
    print(f"\nHCS metadata written for {total} plate zarr(s).")
else:
    print("No plate zarr directories found. Skipping HCS metadata.")
