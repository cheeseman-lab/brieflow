"""OME-NGFF high-content-screening (HCS) metadata for brieflow plate stores.

Pipeline rules write zarr arrays straight into the plate hierarchy
(``aligned_{plate}.zarr/{row}/{col}/{field}``). The functions here read back what
was written and layer the OME-NGFF plate, well, field and label metadata on top:
no arrays are moved, copied or symlinked, only ``zarr.json`` files are written.

The three public entry points run in order, once per plate store:
``write_hcs_metadata`` (plate/row/well/labels groups), ``write_field_image_metadata``
(per-field pixel scale, axis units, channel names, OMERO rendering, label
annotations) and ``write_channel_intensity_statistics`` (screen-wide per-channel
intensity percentiles and summary statistics).
"""

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from iohub.ngff import open_ome_zarr
from iohub.ngff.display import channel_display_settings
from iohub.ngff.models import OMEROMeta, RDefsMeta, TransformationMeta

from lib.shared.file_utils import WELL_ROWCOL_PATTERNS
from lib.shared.image_io import DEFAULT_CHANNEL_COLORS

# Store indices, keyed by resolved store path, so the three passes over one
# plate store share a single directory walk.
_STORE_INDEX_CACHE: dict[str, "_StoreIndex"] = {}

# Stores whose single channel is the store type itself.
_SINGLE_CHANNEL_STORES = {"peaks", "standard_deviation"}

# Label directory stem -> annotation type and the config keys describing it.
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

_AXIS_UNITS = {"X": "micrometer", "Y": "micrometer", "Z": "micrometer", "T": "second"}

# Cap on fields histogrammed per store; above this an evenly spaced subsample
# is used for the intensity percentiles and summary statistics.
_MAX_HISTOGRAM_FIELDS = 256


def write_hcs_metadata(plate_zarr_path, channels_metadata=None):
    """Write the plate, row, well and labels group metadata of a plate store.

    Args:
        plate_zarr_path: Path to the plate zarr directory (e.g. sbs/aligned_1.zarr).
        channels_metadata: Optional list of channel description dicts, embedded at
            the plate root under ``attributes["channels_metadata"]``.

    Returns:
        None.
    """
    plate_path = Path(plate_zarr_path)
    if not plate_path.exists():
        raise FileNotFoundError(f"Plate zarr directory not found: {plate_path}")

    index = _build_store_index(plate_path)
    fields = _find_fields(index)
    if not fields:
        print(f"  No fields found in {plate_path}. Skipping metadata.")
        return

    fields_by_well = {}
    for row, col, field in fields:
        fields_by_well.setdefault((row, col), []).append(field)
    field_count = max(len(f) for f in fields_by_well.values())

    index.set(
        "",
        _plate_metadata(
            plate_path,
            sorted(fields_by_well),
            channels_metadata=channels_metadata,
            field_count=field_count,
        ),
    )

    for row in sorted({row for row, _col in fields_by_well}):
        index.set(row, _group_metadata())

    for (row, col), field_ids in sorted(fields_by_well.items()):
        index.set(f"{row}/{col}", _well_metadata(sorted(field_ids)))
        for field in sorted(field_ids):
            _write_labels_group(index, f"{row}/{col}/{field}")

    index.flush()


def write_field_image_metadata(
    store_path: Path,
    preprocess_root: Path,
    config_channel_names: list[str] | None = None,
    modality_config: dict | None = None,
    channels_metadata: list[dict] | None = None,
    threads: int = 1,
):
    """Write the per-field image metadata of a plate store.

    Covers the physical pixel scale of each pyramid level (from the preprocess
    ``combined_metadata.parquet``), micrometer axis units, real channel names,
    OMERO rendering defaults, the ``image-label`` version and scale of nested
    label stores, and the segmentation provenance of each label store.

    Args:
        store_path: Path to the plate zarr directory.
        preprocess_root: Preprocess output root holding the acquisition metadata.
        config_channel_names: Channel names from the modality config, used when
            their count matches the store's channel count.
        modality_config: The ``sbs`` or ``phenotype`` config block, used to
            describe how each label store was segmented.
        channels_metadata: Optional list of channel description dicts.
        threads: Worker threads for the per-label object counts.

    Returns:
        None.
    """
    store_type = _parse_store_type(store_path)
    plate = _parse_plate_from_store_name(store_path)
    modality = _infer_modality_from_store_path(store_path)
    pixel_sizes = _load_pixel_size_map(preprocess_root, modality, plate)

    index = _get_store_index(store_path)

    dataset = open_ome_zarr(str(store_path), layout="hcs", mode="r+", version="0.5")
    for field_path, field in dataset.positions():
        parts = field_path.split("/")
        if len(parts) != 3:
            continue
        row, col, field_id = parts

        pixel_size = pixel_sizes.get((row, col, field_id)) or pixel_sizes.get(
            (row, col, "*")
        )
        if pixel_size is not None:
            _set_per_dataset_scales(field, *pixel_size)

        resolved = _resolve_channel_names(field, config_channel_names, store_type)
        _rename_channels(field, resolved)
        _set_omero_rendering(field, resolved, channels_metadata=channels_metadata)
        _set_axes_units_micrometer(field)
        field.dump_meta()

    dataset.dump_meta()
    dataset.close()

    # iohub rewrote the plate, well and field zarr.json behind the index.
    index.reload([rel for rel, _ in index.groups() if rel.count("/") < 3])

    # iohub's dump_meta drops downsamplingMethod from the multiscales block.
    for rel, meta in index.groups():
        if not rel or "labels" in rel.split("/"):
            continue
        multiscales = meta.get("attributes", {}).get("ome", {}).get("multiscales", [])
        if multiscales and "downsamplingMethod" not in multiscales[0]:
            multiscales[0]["downsamplingMethod"] = "gaussian"
            index.mark_dirty(rel)

    # iohub does not iterate label stores, so these are written as plain JSON.
    _set_label_versions(index)
    _set_label_axis_units(index)
    _set_label_scales(index, pixel_sizes)
    if modality_config:
        _write_segmentation_metadata(
            index, modality_config, channels_metadata, threads=threads
        )

    print(f"  {store_path.name}: wrote {index.flush()} zarr.json")


def write_channel_intensity_statistics(
    plate_paths: list[Path],
    low_pct: float = 1.0,
    high_pct: float = 99.0,
    threads: int = 1,
) -> int:
    """Write screen-wide per-channel intensity statistics into OMERO metadata.

    Accumulates a full-precision uint16 intensity histogram per channel across
    every image field of every given plate store, then writes the ``low_pct`` /
    ``high_pct`` percentiles as the OMERO display window and the mean, standard
    deviation and median as ``statistics`` on each field's ``omero.channels[i]``.
    The statistics are screen-wide normalization constants, so a dataloader can
    reuse them without rescanning the images.

    Above ``_MAX_HISTOGRAM_FIELDS`` fields per store an evenly spaced,
    deterministic subsample of that many fields is histogrammed instead: an
    unbiased field subsample is accurate to well under a percent for both the
    display window and the statistics. A coarser pyramid level is deliberately
    not used, since gaussian downsampling biases the standard deviation.

    Args:
        plate_paths: Plate zarr roots to histogram and annotate.
        low_pct: Percentile mapped to the OMERO ``window.start``.
        high_pct: Percentile mapped to the OMERO ``window.end``.
        threads: Worker threads for the per-field pixel reads.

    Returns:
        Number of zarr.json files updated across all plates, zero if no image
        fields were found.
    """
    if not plate_paths:
        return 0

    print(
        f"\nComputing screen-wide channel intensity statistics "
        f"[{low_pct}, {high_pct}]th pct over {len(plate_paths)} store(s)..."
    )
    indices = [_get_store_index(path) for path in plate_paths]
    histograms: dict[int, np.ndarray] = {}
    for index in indices:
        _accumulate_channel_histograms(index, histograms, threads=threads)
    if not histograms:
        print("No image fields found for intensity statistics; skipping.")
        return 0

    windows = _windows_from_histograms(histograms, low_pct, high_pct)
    statistics = _statistics_from_histograms(histograms)
    for channel in sorted(windows):
        start, end = windows[channel]
        stat = statistics[channel]
        print(
            f"  channel {channel}: window=({start:.1f}, {end:.1f})  "
            f"mean={stat['mean']:.1f} std={stat['std']:.1f} median={stat['median']:.1f}"
        )

    total = 0
    for index in indices:
        written = _write_intensity_metadata(index, windows, statistics=statistics)
        total += written
        print(f"  {index.root.name}: annotated {written} zarr.json")
    print(f"Channel intensity statistics written to {total} zarr.json files.")
    return total


def _read_zarr_json(path: Path):
    """Parse a zarr.json, returning None if it is missing or unreadable."""
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


class _StoreIndex:
    """Every zarr.json in a plate store, keyed by path relative to the store root.

    Built with one pruned ``os.scandir`` walk. Array directories are recorded but
    never descended into, so the chunk directories under each pyramid level are
    never listed — on a network filesystem that is the difference between
    thousands of stats and millions. Passes mutate the parsed metadata in memory
    and ``flush`` writes each changed node back exactly once.
    """

    def __init__(self, root: Path):
        self.root = Path(root)
        self.nodes: dict[str, dict] = {}
        self.dirty: set[str] = set()
        self._scan(self.root, "")

    def _scan(self, directory: Path, rel: str) -> None:
        meta = _read_zarr_json(directory / "zarr.json")
        if meta is not None:
            self.nodes[rel] = meta
            # An array's children are chunk directories; never walk them.
            if meta.get("node_type") == "array":
                return
        try:
            entries = sorted(os.scandir(directory), key=lambda e: e.name)
        except OSError:
            return
        for entry in entries:
            if entry.is_dir(follow_symlinks=False):
                self._scan(
                    Path(entry.path), f"{rel}/{entry.name}" if rel else entry.name
                )

    def get(self, rel: str):
        """Parsed zarr.json for *rel*, or None if the store has no such node."""
        return self.nodes.get(rel)

    def set(self, rel: str, meta: dict) -> None:
        """Replace the node at *rel* and mark it for write-back."""
        self.nodes[rel] = meta
        self.dirty.add(rel)

    def mark_dirty(self, rel: str) -> None:
        """Mark an already-mutated node for write-back."""
        self.dirty.add(rel)

    def reload(self, rels) -> None:
        """Re-read the named nodes from disk, after an external writer."""
        for rel in rels:
            meta = _read_zarr_json(self.path(rel) / "zarr.json")
            if meta is not None:
                self.nodes[rel] = meta
            self.dirty.discard(rel)

    def path(self, rel: str) -> Path:
        """Absolute directory path of the node at *rel*."""
        return self.root / rel if rel else self.root

    def groups(self):
        """(rel, meta) for every group node, in walk order."""
        return [(r, m) for r, m in self.nodes.items() if m.get("node_type") != "array"]

    def fields(self) -> list[str]:
        """Rel paths of the field-level image groups: ``{row}/{col}/{field}``."""
        out = []
        for rel, meta in self.nodes.items():
            parts = rel.split("/")
            if len(parts) != 3 or meta.get("node_type") == "array":
                continue
            if not parts[2].isdigit():
                continue
            if not any(p.match(f"{parts[0]}{parts[1]}") for p in WELL_ROWCOL_PATTERNS):
                continue
            out.append(rel)
        return sorted(out)

    def label_groups(self) -> list[str]:
        """Rel paths of label groups: ``{row}/{col}/{field}/labels/{name}``."""
        return sorted(r for r in self.nodes if r.split("/")[-2:-1] == ["labels"])

    def array_shape(self, rel: str):
        """``shape`` of the array node at *rel*, or None."""
        meta = self.nodes.get(rel)
        return meta.get("shape") if meta else None

    def flush(self) -> int:
        """Write every dirty node back to disk and return how many were written."""
        written = 0
        for rel in sorted(self.dirty):
            directory = self.path(rel)
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "zarr.json").write_text(json.dumps(self.nodes[rel], indent=2))
            written += 1
        self.dirty.clear()
        return written


def _build_store_index(store_path: Path) -> _StoreIndex:
    """Walk *store_path* afresh and cache the resulting index."""
    index = _StoreIndex(Path(store_path))
    _STORE_INDEX_CACHE[str(Path(store_path).resolve())] = index
    return index


def _get_store_index(store_path: Path) -> _StoreIndex:
    """Return the cached index for *store_path*, walking the store if needed."""
    key = str(Path(store_path).resolve())
    if key not in _STORE_INDEX_CACHE:
        _STORE_INDEX_CACHE[key] = _StoreIndex(Path(store_path))
    return _STORE_INDEX_CACHE[key]


def _find_fields(index) -> list[tuple[str, str, str]]:
    """Locate the (row, col, field) acquisitions recorded in a store index.

    Prefers fields written at ``{row}/{col}/{field}``, and falls back to any
    deeper node under such a prefix, which is how preprocess stores nest cycles.
    """
    strict, loose = [], []
    seen_strict, seen_loose = set(), set()

    for rel in sorted(index.nodes):
        parts = rel.split("/") if rel else []
        if len(parts) < 3:
            continue

        row, col, field = parts[0], parts[1], parts[2]
        if not str(field).isdigit():
            continue
        if not any(p.match(f"{row}{col}") for p in WELL_ROWCOL_PATTERNS):
            continue

        key = (row, col, field)
        if len(parts) == 3 and key not in seen_strict:
            seen_strict.add(key)
            strict.append(key)
        if key not in seen_loose:
            seen_loose.add(key)
            loose.append(key)

    return strict if strict else loose


def _column_order(label):
    """Sort key for an HCS column label: its digits, so `c10` follows `c9`."""
    digits = "".join(ch for ch in str(label) if ch.isdigit())
    return (0, int(digits)) if digits else (1, str(label))


def _plate_metadata(plate_zarr_path, wells, channels_metadata=None, field_count=1):
    """Build the plate-level zarr.json body with OME-NGFF plate metadata."""
    rows = sorted({row for row, _col in wells})
    cols = sorted({col for _row, col in wells}, key=_column_order)

    plate_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "plate": {
                    "version": "0.5",
                    "name": Path(plate_zarr_path).stem,
                    "field_count": field_count,
                    "acquisitions": [{"id": 0}],
                    "columns": [{"name": c} for c in cols],
                    "rows": [{"name": r} for r in rows],
                    "wells": [
                        {
                            "path": f"{row}/{col}",
                            "rowIndex": rows.index(row),
                            "columnIndex": cols.index(col),
                        }
                        for row, col in wells
                    ],
                },
            }
        },
    }

    normalized = _normalize_channels_metadata(channels_metadata)
    if normalized:
        plate_metadata["attributes"]["channels_metadata"] = normalized

    return plate_metadata


def _normalize_channels_metadata(channels_metadata):
    """Normalize the configured channel descriptions for the plate zarr.json."""
    if not channels_metadata:
        return []

    out = []
    for channel in channels_metadata:
        if not isinstance(channel, dict):
            continue

        entry = dict(channel)
        name = (entry.get("name") or "").strip()
        if not name:
            raise ValueError("channels_metadata entry is missing a non-empty 'name'")
        entry["name"] = name
        entry.setdefault("description", "")
        entry.setdefault("channel_type", "fluorescence")

        # Keep biological_annotation only where the operator filled something in.
        annotation = entry.get("biological_annotation")
        cleaned = {}
        if isinstance(annotation, dict):
            for key in ("biological_target", "marker", "marker_type", "full_label"):
                value = annotation.get(key)
                if value is None:
                    continue
                value = str(value).strip()
                if value:
                    cleaned[key] = value
        if cleaned:
            entry["biological_annotation"] = cleaned
        else:
            entry.pop("biological_annotation", None)

        out.append(entry)

    for i, entry in enumerate(out):
        entry.setdefault("index", i)

    return out


def _group_metadata():
    """Build a minimal zarr v3 group zarr.json body."""
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {},
    }


def _well_metadata(field_ids):
    """Build the well-level zarr.json body listing the well's fields."""
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "well": {
                    "version": "0.5",
                    "images": [
                        {"path": str(field), "acquisition": 0} for field in field_ids
                    ],
                },
            }
        },
    }


def _write_labels_group(index, field_rel: str) -> None:
    """Stage the ``labels`` group metadata of a field that has label stores."""
    labels_rel = f"{field_rel}/labels"
    if labels_rel not in index.nodes and not index.path(labels_rel).is_dir():
        return

    label_names = []
    prefix = f"{labels_rel}/"
    for rel in sorted(index.nodes):
        if not rel.startswith(prefix) or "/" in rel[len(prefix) :]:
            continue
        name = rel.rsplit("/", 1)[1]
        if name.endswith(".zarr"):
            label_names.append(name[: -len(".zarr")])
        elif _is_label_group(index.get(rel)):
            label_names.append(name)

    if label_names:
        index.set(labels_rel, _labels_group_metadata(label_names))


def _is_label_group(meta) -> bool:
    """Check whether a parsed zarr.json describes a label image."""
    if not meta:
        return False
    attrs = meta.get("attributes", {})
    # image-label lives under the ome namespace (v3) or at the top level.
    return "image-label" in attrs or "image-label" in attrs.get("ome", {})


def _labels_group_metadata(label_names):
    """Build the labels group zarr.json body listing the available labels."""
    return {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "labels": label_names,
            }
        },
    }


def _parse_plate_from_store_name(store_path: Path) -> str:
    """aligned_1.zarr -> "1", illumination_corrected_12.zarr -> "12"."""
    match = re.search(r"_(\d+)\.zarr$", store_path.name)
    if not match:
        raise ValueError(f"Could not parse plate from store name: {store_path.name}")
    return match.group(1)


def _parse_store_type(store_path: Path) -> str:
    """aligned_1.zarr -> "aligned", peaks_1.zarr -> "peaks"."""
    match = re.match(r"^(.+?)_\d+\.zarr$", store_path.name)
    if not match:
        return store_path.name.replace(".zarr", "")
    return match.group(1)


def _infer_modality_from_store_path(store_path: Path) -> str:
    """Return 'sbs' or 'phenotype' based on which appears in the path parts."""
    parts = store_path.parts
    if "sbs" in parts:
        return "sbs"
    if "phenotype" in parts:
        return "phenotype"
    raise ValueError(f"Could not infer modality from path: {store_path}")


def _load_pixel_size_map(
    preprocess_root: Path, modality: str, plate: str
) -> dict[tuple[str, str, str], tuple[float, float]]:
    """Return (row, col, field) -> (px_x, px_y) in micrometers.

    Read from preprocess/metadata/{modality}/{plate}/{row}/{col}/combined_metadata.parquet.
    Acquisitions that record one pixel size per well rather than per field are
    stored under the field key ``"*"``.
    """
    meta_root = preprocess_root / "metadata" / modality / plate
    pixel_sizes: dict[tuple[str, str, str], tuple[float, float]] = {}

    if not meta_root.exists():
        print(f"  No acquisition metadata under {meta_root}; pixel sizes unavailable")
        return pixel_sizes

    for parquet_fp in meta_root.rglob("combined_metadata.parquet"):
        rel = parquet_fp.relative_to(meta_root)
        if len(rel.parts) < 3:
            continue
        row, col = str(rel.parts[0]), str(rel.parts[1])
        metadata = pd.read_parquet(parquet_fp)

        if "tile" not in metadata.columns:
            if (
                "pixel_size_x" in metadata.columns
                and "pixel_size_y" in metadata.columns
                and len(metadata) > 0
            ):
                pixel_sizes[(row, col, "*")] = (
                    float(metadata["pixel_size_x"].iloc[0]),
                    float(metadata["pixel_size_y"].iloc[0]),
                )
            continue

        for _, record in metadata.iterrows():
            if pd.isna(record.get("pixel_size_x")) or pd.isna(
                record.get("pixel_size_y")
            ):
                continue
            pixel_sizes[(row, col, str(record["tile"]))] = (
                float(record["pixel_size_x"]),
                float(record["pixel_size_y"]),
            )

    return pixel_sizes


def _resolve_channel_names(
    field, config_channel_names: list[str] | None, store_type: str
) -> list[str]:
    """Determine the real channel names for one field of a given store.

    Single-channel stores are named after the store type. Otherwise the
    configured channel names are used when their count matches the store, and
    the names already on the store are kept when it does not.
    """
    try:
        n_channels = len(list(field.channel_names))
    except Exception:
        return []

    if store_type in _SINGLE_CHANNEL_STORES:
        return [store_type]

    if config_channel_names and len(config_channel_names) == n_channels:
        return list(config_channel_names)

    # iohub may return ints when OMERO metadata is missing — always stringify.
    try:
        return [str(name) for name in field.channel_names]
    except Exception:
        return [f"c{i}" for i in range(n_channels)]


def _get_axis_index_ci(field, name: str) -> int:
    """Look up an axis index, tolerating TCZYX or tczyx axis names."""
    try:
        return field.get_axis_index(name.upper())
    except (ValueError, KeyError):
        return field.get_axis_index(name.lower())


def _set_per_dataset_scales(field, px_x: float, px_y: float) -> None:
    """Set the absolute physical pixel scale at each pyramid level.

    Each level's scale is the base pixel size times the downsampling factor
    inferred from the array shapes. The field-level transform is cleared so the
    per-level transforms are the single source of truth.
    """
    multiscales = field.metadata.multiscales[0]
    n_axes = len(multiscales.axes)
    y_idx = _get_axis_index_ci(field, "y")
    x_idx = _get_axis_index_ci(field, "x")
    base_shape = field["0"].shape

    for dataset in multiscales.datasets:
        level_shape = field[dataset.path].shape
        factor_y = (
            base_shape[y_idx] / level_shape[y_idx] if level_shape[y_idx] > 0 else 1.0
        )
        factor_x = (
            base_shape[x_idx] / level_shape[x_idx] if level_shape[x_idx] > 0 else 1.0
        )

        scale = [1.0] * n_axes
        scale[y_idx] = px_y * factor_y
        scale[x_idx] = px_x * factor_x
        dataset.coordinate_transformations = [
            TransformationMeta(type="scale", scale=scale)
        ]

    multiscales.coordinate_transformations = None


def _rename_channels(field, target_names: list[str]) -> None:
    """Rename a field's channels to *target_names* where they differ."""
    try:
        current = list(field.channel_names)
    except Exception:
        return

    for old, new in zip(current, target_names):
        if old == new:
            continue
        try:
            field.rename_channel(old, new)
        except Exception:
            pass


def _set_omero_rendering(
    field,
    channel_names: list[str],
    channels_metadata: list[dict] | None = None,
) -> None:
    """Set the OMERO rendering metadata (channel colors and display defaults).

    Colors come from ``channels_metadata[].color`` where configured, otherwise
    from iohub's per-stain defaults, falling back to the brieflow palette for
    channel names iohub does not recognize.
    """
    if not channel_names:
        return

    config_colors = {}
    if channels_metadata:
        for channel in channels_metadata:
            if isinstance(channel, dict) and "color" in channel:
                config_colors[channel.get("name", "")] = channel["color"]

    channels = []
    for i, name in enumerate(channel_names):
        settings = channel_display_settings(name, clim=None, first_chan=True)
        if name in config_colors:
            settings.color = config_colors[name]
        elif settings.color == "FFFFFF":
            settings.color = DEFAULT_CHANNEL_COLORS[i % len(DEFAULT_CHANNEL_COLORS)]
        settings.active = True
        channels.append(settings)

    field.metadata.omero = OMEROMeta(
        version="0.5",
        channels=channels,
        rdefs=RDefsMeta(default_t=0, default_z=0),
    )


def _set_axes_units_micrometer(field) -> None:
    """Set unit='micrometer' on a field's spatial axes.

    Modifies ``field.metadata`` rather than ``field.zattrs`` so the change
    survives ``dump_meta``.
    """
    if not field.metadata.multiscales:
        return
    for axis in field.metadata.multiscales[0].axes:
        if axis.name.lower() in ("x", "y", "z"):
            axis.unit = "micrometer"


def _set_label_versions(index) -> None:
    """Set ``image-label.version`` on every label store of a plate."""
    for rel in index.label_groups():
        meta = index.get(rel)
        attrs = meta.get("attributes", meta)
        image_label = attrs.get("ome", {}).get("image-label", attrs.get("image-label"))
        if image_label is None:
            continue
        if image_label.get("version") != "0.5":
            image_label["version"] = "0.5"
            index.mark_dirty(rel)


def _set_label_axis_units(index) -> None:
    """Set the axis units on every label store of a plate.

    Runs regardless of pixel size availability — axis units and pixel scales are
    independent concerns.
    """
    for rel in index.label_groups():
        meta = index.get(rel)
        attrs = meta.get("attributes", meta)
        multiscales = attrs.get("ome", {}).get(
            "multiscales", attrs.get("multiscales", [])
        )
        if not multiscales:
            continue

        changed = False
        for axis in multiscales[0].get("axes", []):
            name = axis.get("name", "").upper()
            if name in _AXIS_UNITS and axis.get("unit") != _AXIS_UNITS[name]:
                axis["unit"] = _AXIS_UNITS[name]
                changed = True
        if changed:
            index.mark_dirty(rel)


def _set_label_scales(
    index,
    pixel_sizes: dict[tuple[str, str, str], tuple[float, float]],
) -> None:
    """Set the physical pixel scale of every label store of a plate.

    Labels share the physical pixel size of the parent image field.
    """
    for rel in index.label_groups():
        parts = rel.split("/")
        if len(parts) != 5:
            continue
        row, col, field = parts[0], parts[1], parts[2]

        pixel_size = pixel_sizes.get((row, col, field)) or pixel_sizes.get(
            (row, col, "*")
        )
        if pixel_size is None:
            continue
        px_x, px_y = pixel_size

        meta = index.get(rel)
        attrs = meta.get("attributes", meta)
        multiscales = attrs.get("ome", {}).get(
            "multiscales", attrs.get("multiscales", [])
        )
        if not multiscales:
            continue

        axes = multiscales[0].get("axes", [])
        y_idx = x_idx = None
        for i, axis in enumerate(axes):
            name = axis.get("name", "").upper()
            if name == "Y":
                y_idx = i
            elif name == "X":
                x_idx = i
        if y_idx is None or x_idx is None:
            continue

        datasets = multiscales[0].get("datasets", [])
        if not datasets:
            continue
        base_shape = index.array_shape(f"{rel}/{datasets[0].get('path', '0')}")
        if not base_shape:
            continue

        for dataset in datasets:
            level_shape = (
                index.array_shape(f"{rel}/{dataset.get('path', '0')}") or base_shape
            )
            factor_y = (
                base_shape[y_idx] / level_shape[y_idx]
                if level_shape[y_idx] > 0
                else 1.0
            )
            factor_x = (
                base_shape[x_idx] / level_shape[x_idx]
                if level_shape[x_idx] > 0
                else 1.0
            )

            scale = [1.0] * len(axes)
            scale[y_idx] = px_y * factor_y
            scale[x_idx] = px_x * factor_x
            dataset["coordinateTransformations"] = [{"type": "scale", "scale": scale}]

        index.mark_dirty(rel)


def _build_segmentation_metadata(
    label_stem: str,
    modality_config: dict,
    channels_metadata: list[dict] | None,
) -> dict | None:
    """Describe how one label store was segmented, or None if unrecognized."""
    info = _LABEL_ANNOTATION_MAP.get(label_stem)
    if info is None:
        return None

    method_base = modality_config.get("segmentation_method", "cellpose")
    model = modality_config.get("cellpose_model") or modality_config.get(
        "stardist_model"
    )
    method = f"{method_base}.{model}" if model else method_base
    source_index = modality_config.get(info["source_channel_key"])
    if source_index is None:
        source_index = 0

    annotation = {}
    if channels_metadata:
        for channel in channels_metadata:
            if not isinstance(channel, dict) or channel.get("index") != source_index:
                continue
            channel_annotation = channel.get("biological_annotation", {})
            if isinstance(channel_annotation, dict):
                for key in (
                    "biological_target",
                    "marker",
                    "marker_type",
                    "full_label",
                ):
                    value = channel_annotation.get(key)
                    if value:
                        annotation[key] = value
            break

    parameters = {}
    has_flow = has_cellprob = False
    for param in ("diameter_key", "flow_threshold_key", "cellprob_threshold_key"):
        config_key = info.get(param)
        if config_key and modality_config.get(config_key) is not None:
            parameters[config_key] = modality_config[config_key]
            has_flow = has_flow or "flow" in param
            has_cellprob = has_cellprob or "cellprob" in param

    # Phenotype shares one flow / cellprob threshold across label types.
    if not has_flow and modality_config.get("flow_threshold") is not None:
        parameters["flow_threshold"] = modality_config["flow_threshold"]
    if not has_cellprob and modality_config.get("cellprob_threshold") is not None:
        parameters["cellprob_threshold"] = modality_config["cellprob_threshold"]

    return {
        "label_name": label_stem,
        "annotation_type": info["annotation_type"],
        "is_ome_label": True,
        "source_channel": {"index": source_index},
        "biological_annotation": annotation,
        "segmentation": {
            "method": method,
            "stitching": "none",
            "parameters": parameters,
        },
        "description": f"{info['annotation_type']} segmentation via {method}",
    }


def _count_label_objects(label_dir: str):
    """Number of distinct non-background labels in a label store's level-0 array."""
    import zarr

    try:
        array = zarr.open(label_dir, mode="r")
        return int(len(np.unique(array[:])) - 1)
    except Exception:
        return None


def _write_segmentation_metadata(
    index,
    modality_config: dict,
    channels_metadata: list[dict] | None,
    threads: int = 1,
) -> None:
    """Stage ``segmentation_metadata`` on every recognized label store.

    Written at ``attributes.segmentation_metadata`` alongside ``attributes.ome``.
    The object counts read one label array each, so they are gathered in parallel
    over *threads* worker threads.
    """
    targets = []
    for rel in index.label_groups():
        # The label group is named e.g. "nuclei.zarr" or "nuclei".
        label_stem = rel.rsplit("/", 1)[1].replace(".zarr", "")
        segmentation = _build_segmentation_metadata(
            label_stem, modality_config, channels_metadata
        )
        if segmentation is None:
            continue
        targets.append((rel, segmentation))

    array_dirs = [
        str(index.path(rel) / "0") if index.get(f"{rel}/0") else None
        for rel, _ in targets
    ]
    countable = [d for d in array_dirs if d is not None]
    counts = dict(
        zip(countable, _parallel_map(_count_label_objects, countable, threads))
    )

    for (rel, segmentation), array_dir in zip(targets, array_dirs):
        n_objects = counts.get(array_dir) if array_dir else None
        if n_objects is not None:
            segmentation["statistics"] = {"n_cells": n_objects}
        index.get(rel).setdefault("attributes", {})["segmentation_metadata"] = (
            segmentation
        )
        index.mark_dirty(rel)


def _parallel_map(fn, items, threads: int):
    """Map *fn* over *items*, in a thread pool when *threads* allows it.

    Threads rather than processes: the per-field work is zarr decompression,
    which releases the GIL, and forking a process that already holds zarr's async
    machinery deadlocks.
    """
    if threads <= 1 or len(items) <= 1:
        return [fn(item) for item in items]
    with ThreadPoolExecutor(max_workers=min(threads, len(items))) as pool:
        return list(pool.map(fn, items))


def _accumulate_channel_histograms(
    index,
    histograms: dict[int, np.ndarray],
    threads: int = 1,
) -> None:
    """Add the per-channel intensity histograms of a store's image fields.

    Mutates *histograms* in place, channel index -> int64 counts. Only fields
    that already carry OMERO channels are read, so label arrays are skipped, and
    above ``_MAX_HISTOGRAM_FIELDS`` an evenly spaced subsample is taken so the
    pass stays bounded on large screens.
    """
    fields = [
        rel
        for rel in index.fields()
        if index.get(rel)
        .get("attributes", {})
        .get("ome", {})
        .get("omero", {})
        .get("channels")
        is not None
    ]
    if not fields:
        return

    if len(fields) > _MAX_HISTOGRAM_FIELDS:
        picks = np.unique(
            np.linspace(0, len(fields) - 1, _MAX_HISTOGRAM_FIELDS).round().astype(int)
        )
        print(
            f"  {index.root.name}: histogramming {len(picks)} of "
            f"{len(fields)} fields (evenly spaced subsample)"
        )
        fields = [fields[i] for i in picks]

    dirs = [str(index.path(rel)) for rel in fields]
    n_batches = max(1, min(threads, len(dirs)))
    batches = [dirs[i::n_batches] for i in range(n_batches)]
    for result in _parallel_map(_batch_channel_histograms, batches, threads):
        _merge_histograms(histograms, result)


def _batch_channel_histograms(field_dirs, n_bins: int = 65536):
    """Merged per-channel histograms for a batch of fields.

    Merging inside the worker keeps one accumulator per worker alive instead of
    one full-resolution histogram set per field.
    """
    merged: dict[int, np.ndarray] = {}
    for field_dir in field_dirs:
        _merge_histograms(merged, _field_channel_histograms(field_dir, n_bins=n_bins))
    return merged


def _field_channel_histograms(field_dir: str, n_bins: int = 65536):
    """Per-channel uint16 intensity histograms for one field's level-0 array.

    Reads one chunk row at a time so a whole field is never held in memory.
    """
    import zarr

    try:
        array = zarr.open(field_dir, mode="r")["0"]
    except Exception as exc:
        print(f"  could not read {field_dir}: {exc}")
        return {}

    # Image arrays are (T,C,Z,Y,X), (C,Z,Y,X), or (C,Y,X).
    ndim = len(array.shape)
    if ndim == 5:
        channel_axis = 1
    elif ndim in (3, 4):
        channel_axis = 0
    else:
        return {}

    y_axis = ndim - 2
    n_rows = array.shape[y_axis]
    step = array.chunks[y_axis] if array.chunks else n_rows
    step = max(int(step or n_rows), 1)

    out: dict[int, np.ndarray] = {}
    for y0 in range(0, n_rows, step):
        rows = [slice(None)] * ndim
        rows[y_axis] = slice(y0, min(y0 + step, n_rows))
        slab = array[tuple(rows)]
        if slab.dtype != np.uint16:
            slab = np.clip(slab, 0, n_bins - 1).astype(np.uint16)
        for channel in range(slab.shape[channel_axis]):
            selector = [slice(None)] * ndim
            selector[channel_axis] = channel
            counts = np.bincount(np.ravel(slab[tuple(selector)]), minlength=n_bins)
            if channel in out:
                out[channel] += counts
            else:
                out[channel] = counts
    return out


def _merge_histograms(
    into: dict[int, np.ndarray], other: dict[int, np.ndarray]
) -> None:
    """Add *other*'s per-channel counts into *into*."""
    for channel, counts in other.items():
        if channel in into:
            into[channel] += counts
        else:
            into[channel] = counts


def _windows_from_histograms(
    histograms: dict[int, np.ndarray],
    low_pct: float,
    high_pct: float,
) -> dict[int, tuple[float, float]]:
    """Convert per-channel histograms into (start, end) at the given percentiles."""
    windows: dict[int, tuple[float, float]] = {}
    for channel, hist in histograms.items():
        cdf = np.cumsum(hist).astype(np.float64)
        total = cdf[-1]
        if total == 0:
            windows[channel] = (0.0, float(len(hist) - 1))
            continue
        cdf /= total
        start = float(np.searchsorted(cdf, low_pct / 100.0))
        end = float(np.searchsorted(cdf, high_pct / 100.0))
        if end <= start:
            end = start + 1.0
        windows[channel] = (start, end)
    return windows


def _statistics_from_histograms(
    histograms: dict[int, np.ndarray],
) -> dict[int, dict[str, float]]:
    """Per-channel mean, standard deviation and median from merged histograms."""
    statistics: dict[int, dict[str, float]] = {}
    for channel, hist in histograms.items():
        total = float(hist.sum())
        if total == 0:
            statistics[channel] = {"mean": 0.0, "std": 0.0, "median": 0.0}
            continue
        bins = np.arange(len(hist), dtype=np.float64)
        mean = float((bins * hist).sum() / total)
        variance = float((hist * (bins - mean) ** 2).sum() / total)
        cdf = np.cumsum(hist).astype(np.float64) / total
        statistics[channel] = {
            "mean": mean,
            "std": float(np.sqrt(variance)),
            "median": float(np.searchsorted(cdf, 0.5)),
        }
    return statistics


def _write_intensity_metadata(
    index,
    windows: dict[int, tuple[float, float]],
    statistics: dict[int, dict[str, float]] | None = None,
) -> int:
    """Write the display window and statistics into every image-level zarr.json."""
    for rel, meta in index.groups():
        if "labels" in rel.split("/"):
            continue
        channels = (
            meta.get("attributes", {}).get("ome", {}).get("omero", {}).get("channels")
        )
        if not channels:
            continue
        changed = False
        for i, channel in enumerate(channels):
            if i not in windows:
                continue
            start, end = windows[i]
            channel["window"] = {
                "start": start,
                "end": end,
                "min": 0.0,
                "max": 65535.0,
            }
            if statistics is not None and i in statistics:
                channel["statistics"] = statistics[i]
            changed = True
        if changed:
            index.mark_dirty(rel)
    return index.flush()
