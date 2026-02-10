"""HCS (High Content Screening) OME-NGFF metadata-only fusion utilities.

After Snakemake jobs write zarr stores directly into the HCS plate hierarchy
(e.g., {plate}.zarr/{row}/{col}/{tile}/{image_type}.zarr), these functions
discover what was written and layer the OME-NGFF metadata on top.

No symlinks or data copies — only zarr.json metadata files.
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# High-level API (used by Snakemake scripts)
# ---------------------------------------------------------------------------


def write_hcs_metadata(plate_zarr_path):
    """Write OME-NGFF HCS metadata for an existing plate zarr directory.

    Walks the plate zarr to discover rows, columns, fields, and image/label
    types, then writes zarr.json metadata at each level of the hierarchy.

    Args:
        plate_zarr_path: Path to the plate zarr directory (e.g., sbs/1.zarr).
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

    # Write plate-level metadata
    _write_plate_metadata(plate_path, wells_by_row_col)

    # Write row-level group metadata
    for row in sorted(set(rc[0] for rc in wells_by_row_col)):
        _write_zarr_v3_group_metadata(plate_path / row)

    # Group fields by well
    fields_by_well = {}
    for row, col, tile in structure:
        fields_by_well.setdefault((row, col), []).append(tile)

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
    """Walk a plate zarr directory to discover the row/col/tile structure.

    Expected layout: plate.zarr/{row}/{col}/{tile}/{image_type}.zarr/

    Args:
        plate_zarr_path: Path to the plate zarr directory.

    Returns:
        list of (row, col, tile) tuples found in the directory.
    """
    plate_path = Path(plate_zarr_path)
    results = []
    seen = set()

    for zarr_dir in sorted(plate_path.rglob("*.zarr")):
        if not zarr_dir.is_dir():
            continue
        # Skip the plate zarr itself if it matches *.zarr
        if zarr_dir == plate_path:
            continue

        rel = zarr_dir.relative_to(plate_path)
        parts = list(rel.parts)

        # Expected: {row}/{col}/{tile}/[{cycle}/]{image_type}.zarr
        # or inside labels/: {row}/{col}/{tile}/labels/{label_type}.zarr
        if len(parts) < 4:
            continue

        row, col, tile = parts[0], parts[1], parts[2]
        key = (row, col, tile)
        if key not in seen:
            seen.add(key)
            results.append(key)

    return results


# ---------------------------------------------------------------------------
# Helpers — well parsing
# ---------------------------------------------------------------------------


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
    return "image-label" in meta.get("attributes", {})


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


def _write_plate_metadata(plate_zarr_path, wells_by_row_col):
    """Write HCS plate-level zarr.json with OME-NGFF plate metadata."""
    plate_path = Path(plate_zarr_path)
    plate_path.mkdir(parents=True, exist_ok=True)

    rows = sorted(set(rc[0] for rc in wells_by_row_col.keys()))
    cols = sorted(set(rc[1] for rc in wells_by_row_col.keys()), key=lambda x: int(x))

    plate_metadata = {
        "zarr_format": 3,
        "node_type": "group",
        "attributes": {
            "ome": {
                "version": "0.5",
                "plate": {
                    "acquisitions": [{"id": 0, "name": "default"}],
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
